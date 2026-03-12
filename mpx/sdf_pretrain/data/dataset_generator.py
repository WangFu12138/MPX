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
    sample_local_queries,
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

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
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

            # Randomize Position
            px = np.random.uniform(-8.0, 8.0)
            py = np.random.uniform(-8.0, 8.0)
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

        x = jax.random.uniform(keys[0], (batch_size,), minval=-1.0, maxval=1.0)
        y = jax.random.uniform(keys[1], (batch_size,), minval=-1.0, maxval=1.0)
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

    def get_surface_height_at_xy(self, xy_points: np.ndarray) -> np.ndarray:
        """Get the highest obstacle Z at given (x, y) coordinates using MuJoCo ray cast.

        Uses mjx.ray to cast rays from above straight down to find the surface height.

        Args:
            xy_points: (N, 2) array of (x, y) coordinates in world frame

        Returns:
            (N,) array of Z heights (highest obstacle at each xy)
        """
        N = xy_points.shape[0]
        z_heights = np.zeros(N)

        # Ray cast from high above (z=5.0) straight down
        ray_origin_z = 5.0
        ray_dir = jnp.array([0.0, 0.0, -1.0])  # pointing down

        for i in range(N):
            x, y = xy_points[i]
            ray_origin = jnp.array([x, y, ray_origin_z])

            # Use mjx.ray to find intersection
            # mjx.ray returns (dist, geom_id) tuple
            dist, geom_id = mjx.ray(
                self.mjx_model,
                self.mjx_data,
                ray_origin,
                ray_dir,
                None  # geomgroup=None means check all geoms
            )

            # dist is the distance along ray direction to the first intersection
            # Since ray points down, intersection z = ray_origin_z - dist
            # dist < 0 means no intersection
            if float(dist) > 0:
                z_heights[i] = float(ray_origin_z - dist)
            else:
                # No intersection, use floor height
                z_heights[i] = self.floor_height

        return z_heights

    def generate_batch(self, key: jax.Array, batch_size: int) -> Dict[str, jnp.ndarray]:
        """
        Generate a complete training batch.

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

        # 3. Sample query points (Local frame)
        # Using the smart sampling strategy (uniform + near-surface + edges)
        queries_local_list = []
        keys = jax.random.split(key3, batch_size)
        for i in range(batch_size):
            q_local = sample_local_queries(keys[i], self.num_queries_per_sample, self.view_bounds)
            queries_local_list.append(q_local)
        
        queries_local = jnp.stack(queries_local_list, axis=0)

        # 4. Transform queries to global frame for ground truth calculation
        all_queries_global = []
        # 把局部的采样点转换到全局坐标系下计算SDF
        for i in range(batch_size):
            T = get_camera_transforms(np.array(lidar_poses[i]))[0]
            q_global = local_to_global_points_jax(jnp.array(queries_local[i]), T)
            all_queries_global.append(q_global)

        queries_global = jnp.stack(all_queries_global, axis=0)

        # 5. Compute structural SDF
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
            'queries_global': queries_global, # Optional, for debugging
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
