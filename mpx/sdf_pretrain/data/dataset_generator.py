"""Online dataset generator for SDF pretraining.

This module generates training data on-the-fly using dynamic terrain generation:
1. Generate terrain boxes in memory using WFC algorithm
2. Update MJX model geometry directly (no XML needed)
3. Sample LiDAR poses over terrain
4. Generate heightmaps via ray casting (mjx.ray)
5. Sample query points and compute ground truth SDF

Key changes from previous version:
- No XML file dependency
- Dynamic terrain generation using WFC
- JAX-based geometry updates
- Consistent SDF ground truth

Key class:
- SDFDynamicGenerator: Main data generator class
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
    get_ground_truth_sdf_with_floor,
)
from mpx.sdf_pretrain.data.dynamic_terrain import (
    DynamicTerrainManager,
    TerrainConfig,
    CurriculumTerrainManager,
)
from mpx.terrain.heightmap import create_sensor_matrix_with_mask


class SDFDynamicGenerator:
    """Dynamic data generator for SDF pretraining.

    This class generates terrain in memory and creates training data without
    relying on static XML files. It uses WFC algorithm for diverse terrains.

    Example:
        >>> generator = SDFDynamicGenerator()
        >>> generator.regenerate_terrain(key)  # Generate new terrain
        >>> batch = generator.generate_batch(key, batch_size=32)
        >>> heightmaps = batch['heightmap']  # (32, H, W, 3)
        >>> queries = batch['queries_local']  # (32, N, 3)
        >>> sdfs = batch['sdf']  # (32, N)
    """

    def __init__(self,
                 max_boxes: int = 200,
                 heightmap_size: Tuple[int, int] = (21, 21),
                 heightmap_resolution: float = 0.05,
                 num_queries_per_sample: int = 1024,
                 lidar_height_range: Tuple[float, float] = (0.4, 0.8),
                 view_bounds: Tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.3, 0.5),
                 floor_height: float = 0.0,
                 terrain_config: Optional[TerrainConfig] = None,
                 use_curriculum: bool = False):
        """
        Initialize the SDF dynamic generator.

        Args:
            max_boxes: Maximum number of terrain boxes
            heightmap_size: (H, W) grid size for heightmap
            heightmap_resolution: Distance between heightmap samples (meters)
            num_queries_per_sample: Number of query points per sample
            lidar_height_range: (min, max) height for random LiDAR placement
            view_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) for query sampling
            floor_height: Height of the floor plane
            terrain_config: Configuration for terrain generation
            use_curriculum: Whether to use curriculum learning
        """
        self.max_boxes = max_boxes
        self.heightmap_size = heightmap_size
        self.heightmap_resolution = heightmap_resolution
        self.num_queries_per_sample = num_queries_per_sample
        self.lidar_height_range = lidar_height_range
        self.view_bounds = view_bounds
        self.floor_height = floor_height

        # Initialize terrain manager
        if use_curriculum:
            self.terrain_manager = CurriculumTerrainManager(
                config=terrain_config,
                curriculum_levels=5
            )
        else:
            self.terrain_manager = DynamicTerrainManager(config=terrain_config)

        # Create minimal MuJoCo model with placeholder boxes
        self.mj_model, self.body_id_offset, self.geom_id_offset = self._create_template_model()
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Current terrain boxes info for SDF computation
        self.boxes_info: List[Dict] = []

        print(f"SDFDynamicGenerator initialized:")
        print(f"  Max boxes: {max_boxes}")
        print(f"  Heightmap size: {heightmap_size}")
        print(f"  Template model created with {self.mj_model.nbody} bodies")

    def _create_template_model(self) -> Tuple[mujoco.MjModel, int, int]:
        """Create a minimal MuJoCo model with floor and placeholder boxes.

        Returns:
            Tuple of (model, body_id_offset, geom_id_offset)
        """
        # Create placeholder boxes XML
        boxes_xml = []
        for i in range(self.max_boxes):
            boxes_xml.append(f'<body name="box_{i}" pos="{100+i} {100+i} 10"><geom type="box" size="0.1 0.1 0.1" contype="2" conaffinity="1" rgba="0.6 0.4 0.2 1"/></body>')

        # Create XML string for template model with proper asset definitions
        # Add a dummy free joint to worldbody so MJX forward() works
        xml_template = """<mujoco model="terrain_template">
  <option timestep="0.001" iterations="1" solver="Newton"/>
  <size nconmax="500" njmax="1000" />
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" material="matplane"/>
    <body name="dummy"><freejoint/><geom type="sphere" size="0.001" pos="0 0 100"/></body>
    """ + "\n".join(boxes_xml) + """
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
  </worldbody>
</mujoco>"""

        model = mujoco.MjModel.from_xml_string(xml_template)
        data = mujoco.MjData(model)

        # Find offsets (skip floor geom)
        body_id_offset = 1  # worldbody is 0
        geom_id_offset = 1  # floor geom is 0

        return model, body_id_offset, geom_id_offset

    def regenerate_terrain(self, key: jax.Array):
        """Regenerate terrain with new random layout.

        This updates both the MJX model geometry and the boxes_info for SDF.

        Args:
            key: JAX random key
        """
        # Generate new terrain
        boxes_info = self.terrain_manager.generate_new_terrain(key)
        terrain_matrix = self.terrain_manager.get_terrain_matrix()

        # Update boxes_info for SDF computation
        self.boxes_info = boxes_info

        # Update MJX model geometry
        self._update_mjx_geometry(terrain_matrix[0])

        print(f"Regenerated terrain: {len(boxes_info)} boxes")

    def _update_mjx_geometry(self, terrain_data: np.ndarray):
        """Update MJX model geometry with new terrain data.

        Args:
            terrain_data: Array of shape (num_boxes, 10) with [pos, quat, size]
        """
        num_boxes = min(terrain_data.shape[0], self.max_boxes)
        body_ids = np.arange(self.body_id_offset, self.body_id_offset + num_boxes)
        geom_ids = np.arange(self.geom_id_offset, self.geom_id_offset + num_boxes)

        # Update CPU model first
        for i, (body_id, geom_id) in enumerate(zip(body_ids, geom_ids)):
            if i >= len(terrain_data):
                break
            pos = terrain_data[i, :3]
            quat = terrain_data[i, 3:7]
            size = terrain_data[i, 7:]

            self.mj_model.body_pos[body_id] = pos
            self.mj_model.body_quat[body_id] = quat
            self.mj_model.geom_size[geom_id] = size

        # Run forward kinematics on CPU
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Re-create MJX model and data with updated geometry
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

    def set_curriculum_level(self, level: int):
        """Set curriculum level for terrain difficulty.

        Args:
            level: Curriculum level (0 to max_levels-1)
        """
        if isinstance(self.terrain_manager, CurriculumTerrainManager):
            self.terrain_manager.set_level(level)
        else:
            print("Warning: Curriculum learning not enabled")

    def batch_ray_cast_to_surface(self, xy_points: jnp.ndarray) -> jnp.ndarray:
        """Batch ray cast to get surface height at multiple xy locations.

        Args:
            xy_points: (N, 2) array of (x, y) coordinates in world frame

        Returns:
            (N,) array of Z heights (surface height at each xy)
        """
        N = xy_points.shape[0]
        ray_origin_z = 5.0

        ray_origins = jnp.concatenate([
            xy_points,
            jnp.full((N, 1), ray_origin_z)
        ], axis=-1)

        ray_dir = jnp.array([0.0, 0.0, -1.0])

        # Ray cast against terrain boxes (geomgroup for terrain)
        geomgroup = (1, 0, 0, 0, 0, 0)

        def single_ray(origin):
            # Use mjx_data directly - kinematics already computed when model was created
            dist, _ = mjx.ray(self.mjx_model, self.mjx_data, origin, ray_dir, geomgroup)
            return dist

        dists = jax.vmap(single_ray)(ray_origins)

        surface_z = jnp.where(
            dists > 0,
            ray_origin_z - dists,
            self.floor_height
        )

        return surface_z

    def sample_queries_with_terrain(self, key: jax.Array, lidar_pose: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample query points based on actual terrain height.

        Args:
            key: JAX random key
            lidar_pose: (6,) array [x, y, z, roll, pitch, yaw]

        Returns:
            Tuple of (queries_local, queries_global), each (N, 3)
        """
        num_points = self.num_queries_per_sample
        x_min, x_max, y_min, y_max, z_min, z_max = self.view_bounds

        # Split points among strategies
        n_uniform = int(num_points * 0.2)
        n_surface = int(num_points * 0.5)
        n_edge = num_points - n_uniform - n_surface

        keys = jax.random.split(key, 15)

        # 1. Sample LOCAL xy (three strategies)
        ux = jax.random.uniform(keys[0], (n_uniform,), minval=x_min, maxval=x_max)
        uy = jax.random.uniform(keys[1], (n_uniform,), minval=y_min, maxval=y_max)

        sx = jax.random.uniform(keys[2], (n_surface,), minval=x_min, maxval=x_max)
        sy = jax.random.uniform(keys[3], (n_surface,), minval=y_min, maxval=y_max)

        edge_side_x = jax.random.bernoulli(keys[4], 0.5, (n_edge,))
        ex = jnp.where(edge_side_x, x_max - 0.05, x_min + 0.05)
        ex = ex + jax.random.normal(keys[5], (n_edge,)) * 0.05
        ex = jnp.clip(ex, x_min, x_max)

        edge_side_y = jax.random.bernoulli(keys[6], 0.5, (n_edge,))
        ey = jnp.where(edge_side_y, y_max - 0.05, y_min + 0.05)
        ey = ey + jax.random.normal(keys[7], (n_edge,)) * 0.05
        ey = jnp.clip(ey, y_min, y_max)

        all_local_xy = jnp.concatenate([
            jnp.stack([ux, uy], axis=-1),
            jnp.stack([sx, sy], axis=-1),
            jnp.stack([ex, ey], axis=-1)
        ], axis=0)

        # 2. Transform to global xy
        T_lidar_to_world = get_camera_transforms(np.array(lidar_pose))[0]

        local_xy_homo = jnp.concatenate([
            all_local_xy,
            jnp.zeros((num_points, 1)),
            jnp.ones((num_points, 1))
        ], axis=-1)

        global_xy_homo = (T_lidar_to_world @ local_xy_homo.T).T
        global_xy = global_xy_homo[:, :2]

        # 3. Batch ray cast for surface heights
        surface_z = self.batch_ray_cast_to_surface(global_xy)

        # 4. Sample z based on surface_z
        sensor_z = lidar_pose[2]
        margin = 0.02

        uz = jax.random.uniform(
            keys[8], (n_uniform,),
            minval=jnp.clip(surface_z[:n_uniform] + margin, z_min, sensor_z),
            maxval=sensor_z
        )

        sz = jax.random.uniform(
            keys[9], (n_surface,),
            minval=jnp.clip(surface_z[n_uniform:n_uniform+n_surface] + margin, z_min, sensor_z),
            maxval=jnp.clip(surface_z[n_uniform:n_uniform+n_surface] + 0.1, z_min, sensor_z)
        )

        ez = jax.random.uniform(
            keys[10], (n_edge,),
            minval=jnp.clip(surface_z[n_uniform+n_surface:] + margin, z_min, sensor_z),
            maxval=jnp.clip(surface_z[n_uniform+n_surface:] + 0.15, z_min, sensor_z)
        )

        all_z = jnp.concatenate([uz, sz, ez])

        # 5. Combine queries
        queries_global = jnp.concatenate([global_xy, all_z[:, None]], axis=-1)

        T_world_to_lidar = get_camera_transforms(np.array(lidar_pose))[1]
        queries_local = global_to_local_points_numpy(np.array(queries_global), T_world_to_lidar)
        queries_local = jnp.array(queries_local)

        return queries_local, queries_global

    def sample_lidar_poses(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample random LiDAR poses over the terrain center.

        Args:
            key: JAX random key
            batch_size: Number of poses to sample

        Returns:
            LiDAR poses, shape (batch_size, 6) with [x, y, z, roll, pitch, yaw]
        """
        keys = jax.random.split(key, 6)

        # Sample near terrain center (assuming terrain is centered around origin)
        x_range = 1.5
        y_range = 1.5

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
        """Generate heightmaps using ray casting.

        Args:
            key: JAX random key
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

            hm, hit_mask = create_sensor_matrix_with_mask(
                self.mjx_model, self.mjx_data, center,
                yaw=yaw,
                key=None,
                dist_x=dist,
                dist_y=dist,
                num_heightscans=H,
                num_widthscans=W
            )

            fallback_z = center[2]
            miss_mask = ~hit_mask
            hm_z = jnp.where(miss_mask, fallback_z, hm[..., 2])
            hm = hm.at[..., 2].set(hm_z)

            heightmaps.append(hm)

        return jnp.array(heightmaps)

    def generate_batch(self, key: jax.Array, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate a complete training batch.

        Args:
            key: JAX random key
            batch_size: Number of samples in batch

        Returns:
            Dict containing:
                - 'heightmap': (B, H, W, 3)
                - 'queries_local': (B, N, 3)
                - 'sdf': (B, N) ground truth
                - 'queries_global': (B, N, 3)
                - 'lidar_pose': (B, 6)
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # 1. Sample LiDAR poses
        lidar_poses = self.sample_lidar_poses(key1, batch_size)

        # 2. Generate heightmaps
        heightmaps = self.generate_heightmaps(key2, lidar_poses)

        # 3. Sample query points
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


# Backward compatibility alias
SDFOnlineGenerator = SDFDynamicGenerator


def create_dataloader(generator: SDFDynamicGenerator,
                      batch_size: int = 32,
                      seed: int = 42):
    """Create an infinite data generator for training.

    Args:
        generator: SDFDynamicGenerator instance
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


if __name__ == "__main__":
    # Test the generator
    print("Testing SDFDynamicGenerator...")

    generator = SDFDynamicGenerator(
        max_boxes=100,
        heightmap_size=(11, 11),
        num_queries_per_sample=64,
    )

    key = jax.random.PRNGKey(42)

    # Generate initial terrain
    key, subkey = jax.random.split(key)
    generator.regenerate_terrain(subkey)

    # Generate a batch
    key, subkey = jax.random.split(key)
    batch = generator.generate_batch(subkey, batch_size=2)

    print(f"Batch generated:")
    print(f"  heightmap: {batch['heightmap'].shape}")
    print(f"  queries_local: {batch['queries_local'].shape}")
    print(f"  sdf: {batch['sdf'].shape}")
    print(f"  SDF range: [{float(batch['sdf'].min()):.4f}, {float(batch['sdf'].max()):.4f}]")

    # Regenerate terrain
    key, subkey = jax.random.split(key)
    generator.regenerate_terrain(subkey)

    # Generate another batch
    key, subkey = jax.random.split(key)
    batch2 = generator.generate_batch(subkey, batch_size=2)

    print(f"\nSecond batch (new terrain):")
    print(f"  SDF range: [{float(batch2['sdf'].min()):.4f}, {float(batch2['sdf'].max()):.4f}]")

    print("\nAll tests passed!")
