"""Coordinate transforms for SDF pretraining.

This module provides utilities for transforming between different coordinate frames:
- LiDAR/camera frame: Local frame attached to the sensor
- World frame: Global MuJoCo simulation frame

Key functions:
- get_camera_transforms: Compute 4x4 transformation matrices from 6D pose
- local_to_global_points: Transform points from sensor to world frame
- global_to_local_points: Transform points from world to sensor frame
- sample_local_queries: Sample query points in sensor's local view frustum
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert zyx euler angles to quaternion [w, x, y, z].

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        Quaternion [w, x, y, z]
    """
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


def quat_to_rot_matrix_numpy(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (NumPy).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


def quat_to_rot_matrix_jax(quat: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (JAX).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = jnp.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


def get_camera_transforms(lidar_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transform matrices from 6D lidar pose.

    Args:
        lidar_pose: 6D pose [x, y, z, roll, pitch, yaw]

    Returns:
        Tuple of (T_lidar_to_world, T_world_to_lidar):
            - T_lidar_to_world: 4x4 transformation matrix (lidar -> world)
            - T_world_to_lidar: 4x4 transformation matrix (world -> lidar)
    """
    x, y, z, roll, pitch, yaw = lidar_pose

    # Rotation matrix from euler angles
    quat = euler_to_quat(roll, pitch, yaw)
    R = quat_to_rot_matrix_numpy(quat)

    # Translation vector
    t = np.array([x, y, z])

    # 4x4 transformation matrix: lidar -> world
    T_lidar_to_world = np.eye(4)
    T_lidar_to_world[:3, :3] = R
    T_lidar_to_world[:3, 3] = t

    # 4x4 transformation matrix: world -> lidar
    T_world_to_lidar = np.eye(4)
    T_world_to_lidar[:3, :3] = R.T
    T_world_to_lidar[:3, 3] = -R.T @ t

    return T_lidar_to_world, T_world_to_lidar


def local_to_global_points_numpy(queries_local: np.ndarray,
                                  T_lidar_to_world: np.ndarray) -> np.ndarray:
    """Transform local points to global coordinates (NumPy version).

    Args:
        queries_local: Local points, shape (N, 3)
        T_lidar_to_world: 4x4 transformation matrix

    Returns:
        Global points, shape (N, 3)
    """
    if queries_local.ndim == 1:
        queries_local = queries_local.reshape(1, -1)

    n_points = queries_local.shape[0]
    ones = np.ones((n_points, 1))
    queries_homo = np.hstack([queries_local, ones])

    queries_global_homo = (T_lidar_to_world @ queries_homo.T).T

    return queries_global_homo[:, :3]


def local_to_global_points_jax(queries_local: jnp.ndarray,
                                T_lidar_to_world: jnp.ndarray) -> jnp.ndarray:
    """Transform local points to global coordinates (JAX version).

    Args:
        queries_local: Local points, shape (N, 3)
        T_lidar_to_world: 4x4 transformation matrix

    Returns:
        Global points, shape (N, 3)
    """
    if queries_local.ndim == 1:
        queries_local = queries_local.reshape(1, -1)

    n_points = queries_local.shape[0]
    ones = jnp.ones((n_points, 1))
    queries_homo = jnp.hstack([queries_local, ones])

    queries_global_homo = (T_lidar_to_world @ queries_homo.T).T

    return queries_global_homo[:, :3]


def global_to_local_points_numpy(queries_global: np.ndarray,
                                  T_world_to_lidar: np.ndarray) -> np.ndarray:
    """Transform global points to local coordinates (NumPy version).

    Args:
        queries_global: Global points, shape (N, 3)
        T_world_to_lidar: 4x4 transformation matrix

    Returns:
        Local points, shape (N, 3)
    """
    if queries_global.ndim == 1:
        queries_global = queries_global.reshape(1, -1)

    n_points = queries_global.shape[0]
    ones = np.ones((n_points, 1))
    queries_homo = np.hstack([queries_global, ones])

    queries_local_homo = (T_world_to_lidar @ queries_homo.T).T

    return queries_local_homo[:, :3]


def global_to_local_points_jax(queries_global: jnp.ndarray,
                                T_world_to_lidar: jnp.ndarray) -> jnp.ndarray:
    """Transform global points to local coordinates (JAX version).

    Args:
        queries_global: Global points, shape (N, 3)
        T_world_to_lidar: 4x4 transformation matrix

    Returns:
        Local points, shape (N, 3)
    """
    if queries_global.ndim == 1:
        queries_global = queries_global.reshape(1, -1)

    n_points = queries_global.shape[0]
    ones = jnp.ones((n_points, 1))
    queries_homo = jnp.hstack([queries_global, ones])

    queries_local_homo = (T_world_to_lidar @ queries_homo.T).T

    return queries_local_homo[:, :3]


# Aliases for convenience
local_to_global_points = local_to_global_points_numpy
global_to_local_points = global_to_local_points_numpy


def sample_local_queries(key: jax.Array,
                         num_points: int,
                         view_bounds: Tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.3, 0.5)
                         ) -> jnp.ndarray:
    """Sample query points in the sensor's local view frustum using a smart
    3-tier sampling strategy:

      - 20% Uniform:       fills spatial gaps across the entire view frustum
      - 50% Near-surface:  dense sampling near z ≈ 0 (free space just above terrain)
      - 30% Edge-focused:  concentrated near estimated box edges

    All points are in LOCAL coordinates relative to the sensor (above terrain).

    Args:
        key: JAX random key
        num_points: Total number of points to sample
        view_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) in local frame

    Returns:
        Query points in local frame, shape (num_points, 3)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = view_bounds

    n_uniform = int(num_points * 0.2)
    n_surface = int(num_points * 0.5)
    n_edge = num_points - n_uniform - n_surface  # Remainder (~30%)

    # ---- 1. Uniform sampling (20%) ----
    key, k1, k2, k3 = jax.random.split(key, 4)
    ux = jax.random.uniform(k1, (n_uniform,), minval=x_min, maxval=x_max)
    uy = jax.random.uniform(k2, (n_uniform,), minval=y_min, maxval=y_max)
    uz = jax.random.uniform(k3, (n_uniform,), minval=z_min, maxval=z_max)
    uniform_pts = jnp.stack([ux, uy, uz], axis=-1)

    # ---- 2. Near-surface sampling (50%) ----
    # Dense sampling near z=0 with small Gaussian perturbation (σ=0.02m)
    key, k1, k2, k3 = jax.random.split(key, 4)
    sx = jax.random.uniform(k1, (n_surface,), minval=x_min, maxval=x_max)
    sy = jax.random.uniform(k2, (n_surface,), minval=y_min, maxval=y_max)
    # z centered around 0 (just above terrain) with small variance
    sz = jax.random.normal(k3, (n_surface,)) * 0.02
    sz = jnp.clip(sz, z_min, z_max)
    surface_pts = jnp.stack([sx, sy, sz], axis=-1)

    # ---- 3. Edge-focused sampling (30%) ----
    # Sample near the xy-boundaries of the view frustum (where box edges likely are)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    # x near edges: randomly pick left or right edge, then add small perturbation
    edge_side_x = jax.random.bernoulli(k1, 0.5, (n_edge,))
    ex = jnp.where(edge_side_x, x_max, x_min) + jax.random.normal(k2, (n_edge,)) * 0.05
    ex = jnp.clip(ex, x_min, x_max)
    ey = jax.random.uniform(k3, (n_edge,), minval=y_min, maxval=y_max)
    # z near surface with slightly larger variance
    ez = jax.random.normal(k4, (n_edge,)) * 0.05
    ez = jnp.clip(ez, z_min, z_max)
    edge_pts = jnp.stack([ex, ey, ez], axis=-1)

    # ---- Concatenate all ----
    queries_local = jnp.concatenate([uniform_pts, surface_pts, edge_pts], axis=0)

    return queries_local
