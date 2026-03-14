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
                 view_bounds: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, 0.01, 1.5),
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

    def sample_queries_batched(self, key: jax.Array, lidar_poses: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Batch sample query points for multiple LiDAR poses.

        This method processes all samples in parallel using JAX vmap,
        avoiding Python loops and enabling single JIT compilation.

        采样策略 (表面重要性采样):
          - 10% 内部:  Z = surface_z - [0, 0.10]     → SDF < 0 (穿模检测)
          - 70% 表面:  Z = surface_z ± 0.05          → SDF ≈ 0 (核心区域)
          - 20% 外部:  Z = surface_z + [0.05, 0.50]  → SDF > 0 (全局引导)

        Args:
            key: JAX random key
            lidar_poses: (batch_size, 6) array of LiDAR poses

        Returns:
            Tuple of (queries_local, queries_global), each (batch_size, N, 3)
        """
        batch_size = lidar_poses.shape[0]
        num_points = self.num_queries_per_sample
        x_min, x_max, y_min, y_max, z_min, z_max = self.view_bounds

        # 采样比例: 10% 内部, 70% 表面, 20% 外部
        n_inside = int(num_points * 0.10)    # 10% 内部点
        n_surface = int(num_points * 0.70)   # 70% 表面附近点
        n_outside = num_points - n_inside - n_surface  # 20% 外部点

        # Split keys for different sampling strategies
        key, *subkeys = jax.random.split(key, 12)

        # 1. Batch sample LOCAL xy coordinates (pure JAX, no loops)
        # 内部采样 XY: (batch, n_inside)
        inside_x = jax.random.uniform(subkeys[0], (batch_size, n_inside), minval=x_min, maxval=x_max)
        inside_y = jax.random.uniform(subkeys[1], (batch_size, n_inside), minval=y_min, maxval=y_max)

        # 表面采样 XY: (batch, n_surface)
        surface_x = jax.random.uniform(subkeys[2], (batch_size, n_surface), minval=x_min, maxval=x_max)
        surface_y = jax.random.uniform(subkeys[3], (batch_size, n_surface), minval=y_min, maxval=y_max)

        # 外部采样 XY: (batch, n_outside)
        outside_x = jax.random.uniform(subkeys[4], (batch_size, n_outside), minval=x_min, maxval=x_max)
        outside_y = jax.random.uniform(subkeys[5], (batch_size, n_outside), minval=y_min, maxval=y_max)

        # Combine: (batch, N, 2)
        all_local_xy = jnp.concatenate([
            jnp.stack([inside_x, inside_y], axis=-1),
            jnp.stack([surface_x, surface_y], axis=-1),
            jnp.stack([outside_x, outside_y], axis=-1)
        ], axis=1)

        # 2. Batch transform to global xy
        # Pre-compute all transform matrices
        T_lidar_to_world_list = []
        T_world_to_lidar_list = []
        for i in range(batch_size):
            T_l2w, T_w2l = get_camera_transforms(np.array(lidar_poses[i]))
            T_lidar_to_world_list.append(T_l2w)
            T_world_to_lidar_list.append(T_w2l)

        # Transform each sample's local xy to global xy
        global_xy_list = []
        for i in range(batch_size):
            local_xy = all_local_xy[i]  # (N, 2)
            T = T_lidar_to_world_list[i]

            local_xy_homo = jnp.concatenate([
                local_xy,
                jnp.zeros((num_points, 1)),
                jnp.ones((num_points, 1))
            ], axis=-1)

            global_xy_homo = (T @ local_xy_homo.T).T
            global_xy_list.append(global_xy_homo[:, :2])

        global_xy = jnp.stack(global_xy_list, axis=0)  # (batch, N, 2)

        # 3. Batch ray cast for surface heights
        global_xy_flat = global_xy.reshape(-1, 2)  # (batch*N, 2)
        surface_z_flat = self.batch_ray_cast_to_surface(global_xy_flat)
        surface_z = surface_z_flat.reshape(batch_size, num_points)  # (batch, N)

        # 4. Batch sample z coordinates (新的采样策略)
        sensor_z = lidar_poses[:, 2]  # (batch,)

        # 内部点 Z: surface_z - [0, 0.10] → SDF < 0
        inside_z_offset = jax.random.uniform(subkeys[6], (batch_size, n_inside), minval=0.0, maxval=0.10)
        inside_z = surface_z[:, :n_inside] - inside_z_offset

        # 表面附近 Z: surface_z ± 0.05 → SDF ≈ 0
        surface_z_offset = jax.random.uniform(subkeys[7], (batch_size, n_surface), minval=-0.05, maxval=0.05)
        surface_z_pts = surface_z[:, n_inside:n_inside+n_surface] + surface_z_offset

        # 外部点 Z: surface_z + [0.05, 0.50] → SDF > 0
        outside_z_offset = jax.random.uniform(subkeys[8], (batch_size, n_outside), minval=0.05, maxval=0.50)
        outside_z = surface_z[:, n_inside+n_surface:] + outside_z_offset
        # 限制不超过传感器高度
        outside_z = jnp.minimum(outside_z, jnp.broadcast_to(sensor_z[:, None], (batch_size, n_outside)))

        # Combine z: (batch, N)
        all_z = jnp.concatenate([inside_z, surface_z_pts, outside_z], axis=1)

        # 5. Combine queries: (batch, N, 3)
        queries_global = jnp.concatenate([global_xy, all_z[:, :, None]], axis=-1)

        # 6. Batch convert to local coordinates (不再做SDF正数校验)
        queries_local_list = []
        for i in range(batch_size):
            queries_l = global_to_local_points_numpy(
                np.array(queries_global[i]),
                T_world_to_lidar_list[i]
            )
            queries_local_list.append(jnp.array(queries_l))

        queries_local = jnp.stack(queries_local_list, axis=0)

        return queries_local, queries_global

    def sample_queries_with_terrain(self, key: jax.Array, lidar_pose: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample query points based on actual terrain height.

        采样策略 (表面重要性采样):
          - 10% 内部:  Z = surface_z - [0, 0.10]     → SDF < 0 (穿模检测)
          - 70% 表面:  Z = surface_z ± 0.05          → SDF ≈ 0 (核心区域)
          - 20% 外部:  Z = surface_z + [0.05, 0.50]  → SDF > 0 (全局引导)

        Args:
            key: JAX random key
            lidar_pose: (6,) array [x, y, z, roll, pitch, yaw]

        Returns:
            Tuple of (queries_local, queries_global), each (N, 3)
        """
        num_points = self.num_queries_per_sample
        x_min, x_max, y_min, y_max, z_min, z_max = self.view_bounds

        # 采样比例: 10% 内部, 70% 表面, 20% 外部
        n_inside = int(num_points * 0.10)    # 10% 内部点
        n_surface = int(num_points * 0.70)   # 70% 表面附近点
        n_outside = num_points - n_inside - n_surface  # 20% 外部点

        keys = jax.random.split(key, 15)

        # 1. Sample LOCAL xy (三种采样策略)
        # 内部采样 XY
        inside_x = jax.random.uniform(keys[0], (n_inside,), minval=x_min, maxval=x_max)
        inside_y = jax.random.uniform(keys[1], (n_inside,), minval=y_min, maxval=y_max)

        # 表面采样 XY
        surface_x = jax.random.uniform(keys[2], (n_surface,), minval=x_min, maxval=x_max)
        surface_y = jax.random.uniform(keys[3], (n_surface,), minval=y_min, maxval=y_max)

        # 外部采样 XY
        outside_x = jax.random.uniform(keys[4], (n_outside,), minval=x_min, maxval=x_max)
        outside_y = jax.random.uniform(keys[5], (n_outside,), minval=y_min, maxval=y_max)

        all_local_xy = jnp.concatenate([
            jnp.stack([inside_x, inside_y], axis=-1),
            jnp.stack([surface_x, surface_y], axis=-1),
            jnp.stack([outside_x, outside_y], axis=-1)
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

        # 4. Sample z coordinates (新的采样策略)
        sensor_z = lidar_pose[2]

        # 内部点 Z: surface_z - [0, 0.10] → SDF < 0
        inside_z_offset = jax.random.uniform(keys[6], (n_inside,), minval=0.0, maxval=0.10)
        inside_z = surface_z[:n_inside] - inside_z_offset

        # 表面附近 Z: surface_z ± 0.05 → SDF ≈ 0
        surface_z_offset = jax.random.uniform(keys[7], (n_surface,), minval=-0.05, maxval=0.05)
        surface_z_pts = surface_z[n_inside:n_inside+n_surface] + surface_z_offset

        # 外部点 Z: surface_z + [0.05, 0.50] → SDF > 0
        outside_z_offset = jax.random.uniform(keys[8], (n_outside,), minval=0.05, maxval=0.50)
        outside_z = surface_z[n_inside+n_surface:] + outside_z_offset
        # 限制不超过传感器高度
        outside_z = jnp.minimum(outside_z, sensor_z)

        all_z = jnp.concatenate([inside_z, surface_z_pts, outside_z])

        # 5. Combine queries
        queries_global = jnp.concatenate([global_xy, all_z[:, None]], axis=-1)

        # 6. Convert to local coordinates (不再做SDF正数校验)
        T_world_to_lidar = get_camera_transforms(np.array(lidar_pose))[1]
        queries_local = global_to_local_points_numpy(np.array(queries_global), T_world_to_lidar)
        queries_local = jnp.array(queries_local)

        return queries_local, queries_global

    def sample_lidar_poses(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample random LiDAR poses over the terrain center.

        The z height is based on terrain surface height at each (x, y) position,
        ensuring the sensor is always above the terrain.

        Args:
            key: JAX random key
            batch_size: Number of poses to sample

        Returns:
            LiDAR poses, shape (batch_size, 6) with [x, y, z, roll, pitch, yaw]
        """
        keys = jax.random.split(key, 7)

        # Sample near terrain center (assuming terrain is centered around origin)
        # IMPORTANT: LiDAR position range should allow heightmap to cover terrain!
        # - Terrain xy range: ±3.15m (WFC generated)
        # - Heightmap scan range: ±0.5m (view_bounds)
        # - LiDAR position range: ±2.0m (so heightmap covers ±0.5m around LiDAR)
        # - This ensures heightmap (±0.5m) never exceeds terrain bounds (±3.15m)
        x_range = 2.0  # LiDAR position range: ±2m
        y_range = 2.0  # Heightmap will cover ±0.5m around this position

        x = jax.random.uniform(keys[0], (batch_size,), minval=-x_range, maxval=x_range)
        y = jax.random.uniform(keys[1], (batch_size,), minval=-y_range, maxval=y_range)

        # Get terrain height at each (x, y) position
        xy_points = jnp.stack([x, y], axis=-1)
        surface_z = self.batch_ray_cast_to_surface(xy_points)

        # Sample z based on terrain height + offset
        # Sensor should be at least 0.2m above terrain surface
        z_offset_min = self.lidar_height_range[0]  # 0.4m minimum offset
        z_offset_max = self.lidar_height_range[1]  # 0.8m maximum offset
        z_offset = jax.random.uniform(keys[2], (batch_size,),
                                       minval=z_offset_min,
                                       maxval=z_offset_max)
        z = surface_z + z_offset

        roll = jax.random.uniform(keys[3], (batch_size,), minval=-0.1, maxval=0.1)
        pitch = jax.random.uniform(keys[4], (batch_size,), minval=-0.1, maxval=0.1)
        yaw = jax.random.uniform(keys[5], (batch_size,), minval=-np.pi, maxval=np.pi)

        poses = jnp.stack([x, y, z, roll, pitch, yaw], axis=-1)
        return poses

    def generate_heightmaps(self, key: jax.Array,
                            lidar_poses: jnp.ndarray) -> jnp.ndarray:
        """Generate heightmaps using ray casting.

        Returns heightmaps in LOCAL coordinates:
        - XY: relative to LiDAR center (with yaw rotation)
        - Z: relative to GROUND height (0 = ground level, +0.1 = 10cm step up)

        This is the standard representation for legged robot terrain perception:
        - Z=0 means "at the same height as robot's base/feet"
        - Z=+0.1 means "10cm obstacle to step over"
        - Z=-0.1 means "10cm hole to avoid"

        Uses "Shadow Base Frame" approach:
        - XY and Yaw follow LiDAR position
        - Z-axis is pinned at world Z=0 (sea level)
        - This ensures terrain height is independent of LiDAR mounting height

        Args:
            key: JAX random key
            lidar_poses: LiDAR poses, shape (batch_size, 6)
                pose[:3] = (x, y, sensor_z) - LiDAR absolute position
                pose[5] = yaw angle

        Returns:
            Heightmaps in local coords, shape (batch_size, H, W, 3)
            XY: relative to LiDAR center (range: ±0.5 meters)
            Z: relative to ground height (ground = 0, step up = positive)
        """
        batch_size = lidar_poses.shape[0]
        H, W = self.heightmap_size
        dist = self.heightmap_resolution

        heightmaps = []
        for i in range(batch_size):
            pose = np.array(lidar_poses[i])
            center = pose[:3]  # (x, y, sensor_z) - LiDAR absolute position
            yaw = pose[5]

            # 1. Ray casting to get world coordinate hit points
            hm_world, hit_mask = create_sensor_matrix_with_mask(
                self.mjx_model, self.mjx_data, center,
                yaw=yaw,
                key=None,
                dist_x=dist,
                dist_y=dist,
                num_heightscans=H,
                num_widthscans=W
            )

            # ==================== Core Fix: Shadow Base Frame ====================
            # 2. Build "Shadow Base Frame" (Gravity-Aligned Base Frame)
            #    - XY and Yaw align with LiDAR
            #    - Z-axis is FORCED to 0 (world sea level)
            #    - This decouples terrain height from LiDAR mounting height
            base_pose = np.array([
                center[0], center[1], 0.0,  # Z=0 is the key!
                0.0, 0.0, yaw  # roll=0, pitch=0, yaw=yaw (gravity-aligned)
            ])
            _, T_world_to_base = get_camera_transforms(base_pose)

            # 3. Transform XY coordinates (hm_local Z still has real world altitude)
            H_W = H * W
            hm_world_flat = np.array(hm_world).reshape(H_W, 3)
            hm_local_flat = global_to_local_points_numpy(hm_world_flat, T_world_to_base)
            hm_local = jnp.array(hm_local_flat.reshape(H, W, 3))

            # 4. Get the TRUE ground height directly under LiDAR
            #    Center of grid (H//2, W//2) is the point right below LiDAR
            ground_z_world = hm_world[H // 2, W // 2, 2]

            # 5. Compute pure terrain relief (relative to ground)
            #    This is what we want: 0=ground, +0.1=step up, -0.1=hole
            hm_z_relative = hm_local[..., 2] - ground_z_world

            # 6. Handle miss points (set to 0 = ground level) and clip extremes
            miss_mask = ~hit_mask
            hm_z_final = jnp.where(miss_mask, 0.0, hm_z_relative)
            hm_z_final = jnp.clip(hm_z_final, -1.0, 1.0)  # Physical limits

            # Combine: XY from transform, Z relative to ground
            hm_local = hm_local.at[..., 2].set(hm_z_final)

            heightmaps.append(hm_local)

        return jnp.array(heightmaps)

    def generate_batch(self, key: jax.Array, batch_size: int, verbose: bool = False) -> Dict[str, jnp.ndarray]:
        """Generate a complete training batch using batched processing.

        This method uses the new batched sampling approach for faster data generation.

        Args:
            key: JAX random key
            batch_size: Number of samples in batch
            verbose: Print detailed timing information

        Returns:
            Dict containing:
                - 'heightmap': (B, H, W, 3)
                - 'queries_local': (B, N, 3)
                - 'sdf': (B, N) ground truth
                - 'queries_global': (B, N, 3)
                - 'lidar_pose': (B, 6)
        """
        import time

        key1, key2, key3 = jax.random.split(key, 3)

        # 1. Sample LiDAR poses
        if verbose:
            print("    [Batch Gen] Step 1/4: Sampling LiDAR poses...", end='', flush=True)
            t0 = time.time()
        lidar_poses = self.sample_lidar_poses(key1, batch_size)
        if verbose:
            print(f" Done ({time.time()-t0:.2f}s)")

        # 2. Generate heightmaps
        if verbose:
            print("    [Batch Gen] Step 2/4: Generating heightmaps...", end='', flush=True)
            t0 = time.time()
        heightmaps = self.generate_heightmaps(key2, lidar_poses)
        if verbose:
            print(f" Done ({time.time()-t0:.2f}s)")

        # 3. Batch sample query points (NEW: uses batched processing)
        if verbose:
            print("    [Batch Gen] Step 3/4: Batch sampling query points...", end='', flush=True)
            t0 = time.time()

        queries_local, queries_global = self.sample_queries_batched(key3, lidar_poses)

        if verbose:
            print(f" Done ({time.time()-t0:.2f}s)")

        # 4. Compute ground truth SDF (already validated in batched method)
        if verbose:
            print("    [Batch Gen] Step 4/4: Computing ground truth SDF...", end='', flush=True)
            t0 = time.time()

        # Final SDF computation for the batch
        queries_flat = queries_global.reshape(-1, 3)
        sdfs_flat = get_ground_truth_sdf_with_floor(
            queries_flat,
            self.boxes_info,
            self.floor_height
        )
        sdfs = sdfs_flat.reshape(batch_size, self.num_queries_per_sample)

        if verbose:
            num_neg = int(jnp.sum(sdfs_flat < 0.0))
            print(f" Done ({time.time()-t0:.2f}s, negative SDFs: {num_neg})")

        # 5. 物理固定缩放 (Fixed Scaling) - 保留绝对物理尺度!
        # ================== 关键修改 ==================
        # 废除单样本 Z-Score normalization!
        # Z-Score 会摧毁物理尺度，导致网络无法预测绝对距离 SDF。
        #
        # 物理范围:
        #   XY 原始是 [-0.5, 0.5] 米，乘以 2 变成 [-1, 1]
        #   Z  原始是 [-1.0, 1.0] 米，乘以 1 保持 [-1, 1]
        # ===============================================

        if verbose:
            print("\n  === Raw Heightmap Statistics (BEFORE fixed scaling, local coords) ===")
            print(f"  Expected: X/Y in [-0.5, 0.5], Z in [-0.15, +0.15] (relative to ground)")
            print(f"  Z=0: ground level, Z=+0.1: 10cm step up, Z=-0.1: 10cm hole")
            print(f"  Actual X: [{float(heightmaps[..., 0].min()):.4f}, {float(heightmaps[..., 0].max()):.4f}], mean: {float(heightmaps[..., 0].mean()):.4f}")
            print(f"  Actual Y: [{float(heightmaps[..., 1].min()):.4f}, {float(heightmaps[..., 1].max()):.4f}], mean: {float(heightmaps[..., 1].mean()):.4f}")
            print(f"  Actual Z: [{float(heightmaps[..., 2].min()):.4f}, {float(heightmaps[..., 2].max()):.4f}], mean: {float(heightmaps[..., 2].mean()):.4f}")

        # 1. 物理截断 (防打空/防自身碰撞)
        heightmaps_clipped = jnp.clip(heightmaps, -1.0, 1.0)

        # 2. 固定系数缩放 (Fixed Scaling) -> 强行映射到 [-1, 1]
        # 构造缩放系数张量 [2.0, 2.0, 1.0]
        scale_factors = jnp.array([2.0, 2.0, 1.0])

        # 直接相乘，保留绝对物理尺度！
        heightmaps_normalized = heightmaps_clipped * scale_factors

        return {
            'heightmap': heightmaps_normalized,
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
