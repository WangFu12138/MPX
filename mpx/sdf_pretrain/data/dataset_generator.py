"""Online dataset generator for SDF pretraining.

This module generates training data on-the-fly:
1. Sample random LiDAR poses over terrain
2. Generate heightmaps via ray casting (mjx.ray)
3. Sample query points in local view frustum
4. Compute ground truth SDF values

Key class:
- SDFOnlineGenerator: Main data generator class
"""

import os
import numpy as np
from typing import Tuple, Optional, List, Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mpx.sdf_pretrain.data.transforms import (
    get_camera_transforms,
    local_to_global_points_jax,
    global_to_local_points_numpy,
)
from mpx.sdf_pretrain.data.analytical_sdf import (
    parse_xml_to_boxes,
    get_ground_truth_sdf_with_floor,
)
from mpx.terrain.heightmap import create_sensor_matrix, create_sensor_matrix_with_mask


class SDFOnlineGenerator:
    """Online data generator for SDF pretraining.

    This class:
    1. Loads a MuJoCo terrain scene
    2. Parses box geometry for ground truth SDF
    3. Generates heightmaps and SDF labels on-the-fly

    Example:
        >>> generator = SDFOnlineGenerator("scene_terrain_test.xml")
        >>> batch = generator.generate_batch(key, batch_size=32)
        >>> heightmaps = batch['heightmap']  # (32, H, W, 3)
        >>> queries = batch['queries_local']  # (32, N, 3)
        >>> sdfs = batch['sdf']  # (32, N)
    """

    def __init__(self,
                 xml_path: str,
                 heightmap_size: Tuple[int, int] = (21, 21),
                 heightmap_resolution: float = 0.05,
                 num_queries_per_sample: int = 1024,
                 lidar_height_range: Tuple[float, float] = (0.4, 0.8),
                 view_bounds: Tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.3, 0.5),
                 floor_height: float = 0.0):
        """
        Initialize the SDF online generator.

        Args:
            xml_path: Path to MuJoCo XML scene file
            heightmap_size: (H, W) grid size for heightmap
            heightmap_resolution: Distance between heightmap samples (meters)
            num_queries_per_sample: Number of query points per sample
            lidar_height_range: (min, max) height for random LiDAR placement
            view_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) for query sampling
            floor_height: Height of the floor plane
        """
        self.xml_path = xml_path
        self.heightmap_size = heightmap_size
        self.heightmap_resolution = heightmap_resolution
        self.num_queries_per_sample = num_queries_per_sample
        self.lidar_height_range = lidar_height_range
        self.view_bounds = view_bounds
        self.floor_height = floor_height

        # Load MuJoCo model (不含机器人)
        terrain_xml = self._get_terrain_only_path()
        if not os.path.exists(terrain_xml):
            raise FileNotFoundError(f"Terrain scene not found: {terrain_xml}")

        self.mj_model = mujoco.MjModel.from_xml_path(terrain_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        # CRITICAL: must call mj_forward to initialize geometry positions,
        # collision tree, and kinematics BEFORE transferring to MJX.
        # Without this, mjx.ray operates on zeroed-out geometry data,
        # producing garbage intersection distances (~4e13).
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Parse initial box geometry for ground truth SDF
        self._update_boxes_info()

        print(f"SDFOnlineGenerator initialized:")
        print(f"  XML path: {xml_path}")
        print(f"  Heightmap size: {heightmap_size}")
        print(f"  Real terrain boxes: {len(self.boxes_info)}")

    def _update_boxes_info(self):
        """Update self.boxes_info from the current MuJoCo model state."""
        boxes = []
        for i in range(self.mj_model.ngeom):
            # Check if this geom is a box
            if self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                body_id = self.mj_model.geom_bodyid[i]
                # Filter out boxes too far away (e.g., walls/placeholders)
                pos = self.mj_model.geom_pos[i]
                if pos[0] > 50 or pos[1] > 50:
                    continue

                boxes.append({
                    'pos': pos.tolist(),
                    'quat': self.mj_model.geom_quat[i].tolist(),
                    'size': self.mj_model.geom_size[i].tolist()
                })
        self.boxes_info = boxes

    def _get_terrain_only_path(self) -> str:
        """Get the path to the terrain-only XML file.
        Returns the path to scene_terrain_only.xml in the same directory as xml_path,
        or falls back to xml_path if not found.
        """
        dir_name = os.path.dirname(self.xml_path)
        terrain_only = os.path.join(dir_name, "scene_terrain_only.xml")
        if os.path.exists(terrain_only):
            return terrain_only
        return self.xml_path

    def batch_ray_cast_to_surface(self, xy_points: jnp.ndarray) -> jnp.ndarray:
        """Batch ray cast to get surface height at multiple xy locations.

        Uses vmap for parallel ray casting instead of slow loops.
        Uses same geomgroup as heightmap to exclude robot parts.

        Args:
            xy_points: (N, 2) array of (x, y) coordinates in world frame

        Returns:
            (N,) array of Z heights (surface height at each xy)
        """
        N = xy_points.shape[0]
        ray_origin_z = 5.0  # Start rays from high above

        # Create ray origins: (N, 3) with z = 5.0
        ray_origins = jnp.concatenate([
            xy_points,
            jnp.full((N, 1), ray_origin_z)
        ], axis=-1)

        # Ray direction: straight down
        ray_dir = jnp.array([0.0, 0.0, -1.0])

        # Use geomgroup that ONLY checks group 0 (terrain boxes)
        # This excludes robot parts (group 2) and other objects
        geomgroup = (1, 0, 0, 0, 0, 0)

        # vmap over ray origins for batch ray casting
        def single_ray(origin):
            dist, _ = mjx.ray(self.mjx_model, self.mjx_data, origin, ray_dir, geomgroup)
            return dist

        dists = jax.vmap(single_ray)(ray_origins)

        # Compute surface z: ray_origin_z - dist
        # If dist < 0 (no hit), use floor_height
        surface_z = jnp.where(
            dists > 0,
            ray_origin_z - dists,
            self.floor_height
        )

        return surface_z

    def sample_queries_with_terrain(self, key: jax.Array, lidar_pose: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample query points based on actual terrain height.

        3-tier sampling strategy (uniform, near-surface, edge):
        1. Sample xy locations (different strategies)
        2. Ray cast to get surface height z0 at each xy
        3. Sample z above the surface (z0 + margin to sensor_z)

        Args:
            key: JAX random key
            lidar_pose: (6,) array [x, y, z, roll, pitch, yaw]

        Returns:
            Tuple of (queries_local, queries_global), each (N, 3)
        """
        num_points = self.num_queries_per_sample
        x_min, x_max, y_min, y_max, z_min, z_max = self.view_bounds

        # Split points among three strategies
        n_uniform = int(num_points * 0.2)
        n_surface = int(num_points * 0.5)
        n_edge = num_points - n_uniform - n_surface

        # Generate random keys
        keys = jax.random.split(key, 15)

        # ========== 1. Sample LOCAL xy (三种策略) ==========

        # Uniform xy (20%)
        ux = jax.random.uniform(keys[0], (n_uniform,), minval=x_min, maxval=x_max)
        uy = jax.random.uniform(keys[1], (n_uniform,), minval=y_min, maxval=y_max)

        # Near-surface xy (50%) - uniform distribution
        sx = jax.random.uniform(keys[2], (n_surface,), minval=x_min, maxval=x_max)
        sy = jax.random.uniform(keys[3], (n_surface,), minval=y_min, maxval=y_max)

        # Edge xy (30%) - near boundaries
        edge_side_x = jax.random.bernoulli(keys[4], 0.5, (n_edge,))
        ex = jnp.where(edge_side_x, x_max - 0.05, x_min + 0.05)
        ex = ex + jax.random.normal(keys[5], (n_edge,)) * 0.05
        ex = jnp.clip(ex, x_min, x_max)

        edge_side_y = jax.random.bernoulli(keys[6], 0.5, (n_edge,))
        ey = jnp.where(edge_side_y, y_max - 0.05, y_min + 0.05)
        ey = ey + jax.random.normal(keys[7], (n_edge,)) * 0.05
        ey = jnp.clip(ey, y_min, y_max)

        # Concatenate all local xy
        all_local_xy = jnp.concatenate([
            jnp.stack([ux, uy], axis=-1),
            jnp.stack([sx, sy], axis=-1),
            jnp.stack([ex, ey], axis=-1)
        ], axis=0)  # (N, 2)

        # ========== 2. 转换为全局 xy ==========
        T_lidar_to_world = get_camera_transforms(np.array(lidar_pose))[0]

        # Transform local xy to global xy
        # local_xy is (N, 2), need to add z=0 for transformation
        local_xy_homo = jnp.concatenate([
            all_local_xy,
            jnp.zeros((num_points, 1)),
            jnp.ones((num_points, 1))
        ], axis=-1)  # (N, 4) homogeneous

        global_xy_homo = (T_lidar_to_world @ local_xy_homo.T).T  # (N, 4)
        global_xy = global_xy_homo[:, :2]  # (N, 2)

        # ========== 3. 批量 ray cast 获取表面高度 ==========
        surface_z = self.batch_ray_cast_to_surface(global_xy)  # (N,)

        # ========== 4. 基于 surface_z 采样 z ==========
        sensor_z = lidar_pose[2]
        margin = 0.02  # 2cm above surface

        # Uniform sampling: surface to sensor (full range)
        uz = jax.random.uniform(
            keys[8], (n_uniform,),
            minval=jnp.clip(surface_z[:n_uniform] + margin, z_min, sensor_z),
            maxval=sensor_z
        )

        # Near-surface sampling: dense near surface (0.1m range)
        sz = jax.random.uniform(
            keys[9], (n_surface,),
            minval=jnp.clip(surface_z[n_uniform:n_uniform+n_surface] + margin, z_min, sensor_z),
            maxval=jnp.clip(surface_z[n_uniform:n_uniform+n_surface] + 0.1, z_min, sensor_z)
        )

        # Edge sampling: near surface with larger range (0.15m)
        ez = jax.random.uniform(
            keys[10], (n_edge,),
            minval=jnp.clip(surface_z[n_uniform+n_surface:] + margin, z_min, sensor_z),
            maxval=jnp.clip(surface_z[n_uniform+n_surface:] + 0.15, z_min, sensor_z)
        )

        # Concatenate all z
        all_z = jnp.concatenate([uz, sz, ez])  # (N,)

        # ========== 5. 组合查询点 ==========
        # Global queries
        queries_global = jnp.concatenate([global_xy, all_z[:, None]], axis=-1)  # (N, 3)

        # Transform to local frame
        T_world_to_lidar = get_camera_transforms(np.array(lidar_pose))[1]
        queries_local = global_to_local_points_numpy(np.array(queries_global), T_world_to_lidar)
        queries_local = jnp.array(queries_local)

        return queries_local, queries_global

    def randomize_terrain(self, key: jax.Array):
        """Randomize the terrain by modifying box sizes and positions.

        This ensures the network doesn't overfit to a single fixed terrain layout.
        The layout is updated in the CPU MjModel, then pushed to MJX.
        """
        # Find all boxes that are part of the terrain (skip walls)
        box_geom_ids = []
        for i in range(self.mj_model.ngeom):
            if self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                pos = self.mj_model.geom_pos[i]
                if pos[0] < 50 and pos[1] < 50:
                    box_geom_ids.append(i)

        n_boxes = len(box_geom_ids)
        if n_boxes == 0:
            return

        # Generate random parameters
        np.random.seed(int(jax.random.randint(key, (), 0, 1000000)))

        for geom_id in box_geom_ids:
            # Randomize Size
            sx = np.random.uniform(0.2, 1.0)
            sy = np.random.uniform(0.2, 1.0)
            sz = np.random.uniform(0.01, 0.1)
            self.mj_model.geom_size[geom_id] = [sx, sy, sz]

            # Randomize Position (Keep tight within lidar's central view)
            px = np.random.uniform(-3.0, 3.0)
            py = np.random.uniform(-3.0, 3.0)
            pz = sz
            self.mj_model.geom_pos[geom_id] = [px, py, pz]

            # Random yaw rotation
            yaw = np.random.uniform(-np.pi/4, np.pi/4)
            qw = np.cos(yaw/2)
            qz = np.sin(yaw/2)
            self.mj_model.geom_quat[geom_id] = [qw, 0, 0, qz]

        # Update kinematics and collision trees
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Push to MJX
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Update ground truth SDF query boxes
        self._update_boxes_info()

    def sample_lidar_poses(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample random LiDAR poses.

        Args:
            key: JAX random key
            batch_size: Number of poses to sample

        Returns:
            LiDAR poses, shape (batch_size, 6) with [x, y, z, roll, pitch, yaw]
        """
        keys = jax.random.split(key, 6)

        # Force LiDAR to spawn directly above the generated terrain cluster
        # The randomize_terrain function places boxes within [-3.0, 3.0]
        x_range = 2.5
        y_range = 2.5
        
        x = jax.random.uniform(keys[0], (batch_size,), minval=-x_range, maxval=x_range)
        y = jax.random.uniform(keys[1], (batch_size,), minval=-y_range, maxval=y_range)
        z = jax.random.uniform(keys[2], (batch_size,),
                              minval=self.lidar_height_range[0],
                              maxval=self.lidar_height_range[1])

        roll = jax.random.uniform(keys[3], (batch_size,), minval=-0.1, maxval=0.1)
        pitch = jax.random.uniform(keys[4], (batch_size,), minval=-0.1, maxval=0.1)
        yaw = jax.random.uniform(keys[5], (batch_size,), minval=-np.pi, maxval=np.pi)

        poses = jnp.stack([x, y, z, roll, pitch, yaw], axis=-1)
        return poses

    def generate_heightmaps(self, key: jax.Array,
                            lidar_poses: jnp.ndarray) -> jnp.ndarray:
        """Generate heightmaps using ray casting with proper ray-miss handling.

        Args:
            key: JAX random key (unused, for potential noise injection)
            lidar_poses: LiDAR poses, shape (batch_size, 6)

        Returns:
            Heightmaps, shape (batch_size, H, W, 3) with 3D intersection points
        """
        batch_size = lidar_poses.shape[0]
        H, W = self.heightmap_size
        dist = self.heightmap_resolution

        heightmaps = []
        for i in range(batch_size):
            pose = np.array(lidar_poses[i])
            center = pose[:3]
            yaw = pose[5]

            # Generate heightmap WITH hit/miss mask
            hm, hit_mask = create_sensor_matrix_with_mask(
                self.mjx_model, self.mjx_data, center,
                yaw=yaw,
                key=None,
                dist_x=dist,
                dist_y=dist,
                num_heightscans=H,
                num_widthscans=W
            )

            # For missed rays: replace the z-coordinate with sensor height
            fallback_z = center[2]
            miss_mask = ~hit_mask
            hm_z = jnp.where(miss_mask, fallback_z, hm[..., 2])
            hm = hm.at[..., 2].set(hm_z)

            heightmaps.append(hm)

        return jnp.array(heightmaps)

    def generate_batch(self, key: jax.Array, batch_size: int) -> Dict[str, jnp.ndarray]:
        """
        Generate a complete training batch.

        Uses terrain-aware sampling: queries are sampled ABOVE the actual surface,
        guaranteeing positive SDF values (free space).

        Args:
            key: JAX random key
            batch_size: Number of samples in batch

        Returns:
            Dict containing:
                - 'heightmap': (B, H, W, 3)
                - 'queries_local': (B, N, 3)
                - 'sdf': (B, N) ground truth
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # 1. Sample LiDAR poses
        lidar_poses = self.sample_lidar_poses(key1, batch_size)

        # 2. Generate heightmaps
        heightmaps = self.generate_heightmaps(key2, lidar_poses)

        # 3. Sample query points using terrain-aware strategy
        queries_local_list = []
        queries_global_list = []
        keys = jax.random.split(key3, batch_size)

        for i in range(batch_size):
            q_local, q_global = self.sample_queries_with_terrain(keys[i], lidar_poses[i])
            queries_local_list.append(q_local)
            queries_global_list.append(q_global)

        queries_local = jnp.stack(queries_local_list, axis=0)
        queries_global = jnp.stack(queries_global_list, axis=0)

        # 4. Compute ground truth SDF
        sdfs = get_ground_truth_sdf_with_floor(
            queries_global.reshape(-1, 3),
            self.boxes_info,
            self.floor_height
        )
        sdfs = sdfs.reshape(batch_size, self.num_queries_per_sample)

        return {
            'heightmap': heightmaps,
            'queries_local': queries_local,
            'sdf': sdfs,
            'queries_global': queries_global,
            'lidar_pose': lidar_poses,
        }


def create_dataloader(generator: SDFOnlineGenerator,
                      batch_size: int = 32,
                      seed: int = 42):
    """Create an infinite data generator for training.

    Args:
        generator: SDFOnlineGenerator instance
        batch_size: Batch size
        seed: Random seed

    Yields:
        Dict with batch data
    """
    key = jax.random.PRNGKey(seed)

    while True:
        key, subkey = jax.random.split(key)
        batch = generator.generate_batch(subkey, batch_size)
        yield batch
