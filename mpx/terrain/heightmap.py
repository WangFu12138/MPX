"""Heightmap utilities for terrain perception using ray casting.

This module provides JAX-based ray casting for generating local heightmaps
from MuJoCo/MJX simulation environments.
"""
import mujoco
import numpy as np
from mujoco import mjx
import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def raycast_sensor(mjx_model, mjx_data, pos):
    """
    Perform a single ray cast from a position downward.

    Args:
        mjx_model: MJX model
        mjx_data: MJX data
        pos: 3D position [x, y, z] to cast ray from

    Returns:
        Intersection point [x, y, z] on the terrain
    """
    ray_sensor_site = jnp.array([pos[0], pos[1], pos[2]])
    direction_vector = jnp.array([0, 0, -1.])
    geomgroup_mask = (1, 0, 0, 0, 1, 1)

    f_ray = partial(mjx.ray, vec=direction_vector, geomgroup=geomgroup_mask)

    z = f_ray(mjx_model, mjx_data, ray_sensor_site)
    intersection_point = ray_sensor_site + direction_vector * z[0]

    return intersection_point


@jax.jit
def raycast_sensor_with_dist(mjx_model, mjx_data, pos):
    """
    Perform a single ray cast from a position downward.
    Returns BOTH the intersection point AND the raw distance.

    When the ray misses all geometry, mjx.ray returns distance = -1.
    Callers should check `dist < 0` to detect misses.

    Args:
        mjx_model: MJX model
        mjx_data: MJX data
        pos: 3D position [x, y, z] to cast ray from

    Returns:
        Tuple of (intersection_point [x, y, z], distance scalar)
    """
    ray_sensor_site = jnp.array([pos[0], pos[1], pos[2]])
    direction_vector = jnp.array([0, 0, -1.])
    geomgroup_mask = (1, 0, 0, 0, 1, 1)

    f_ray = partial(mjx.ray, vec=direction_vector, geomgroup=geomgroup_mask)

    result = f_ray(mjx_model, mjx_data, ray_sensor_site)
    dist = result[0]
    intersection_point = ray_sensor_site + direction_vector * dist

    return intersection_point, dist


def create_sensor_matrix_with_mask(mx, dx, center, yaw=0., key=None, noise_range=0.005,
                                    *, dist_x, dist_y, num_heightscans, num_widthscans):
    """
    Create a grid map using ray sensor data, WITH a hit/miss mask.

    Same as create_sensor_matrix but additionally returns a boolean mask
    indicating which rays successfully hit geometry.

    Args:
        Same as create_sensor_matrix.

    Returns:
        Tuple of:
            - points: Heightmap matrix of shape (num_heightscans, num_widthscans, 3)
            - hit_mask: Boolean mask of shape (num_heightscans, num_widthscans),
                        True where ray hit geometry, False where it missed.
    """
    R_W2H = jnp.array([jnp.cos(yaw), jnp.sin(yaw), -jnp.sin(yaw), jnp.cos(yaw)])
    R_W2H = R_W2H.reshape((2, 2))

    c_h = (num_heightscans - 1) / 2
    c_w = (num_widthscans - 1) / 2

    ref_robot = jnp.array([center[0], center[1], center[2] + 0.6])

    idx_h = jnp.arange(num_heightscans)
    idx_w = jnp.arange(num_widthscans)

    p, k = jnp.meshgrid(c_h - idx_h, c_w - idx_w, indexing="ij")
    offsets = jnp.stack([p * dist_x, k * dist_y], axis=-1)

    if key is not None:
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, shape=offsets.shape, minval=-noise_range, maxval=noise_range)
        offsets = offsets + noise

    offsets = offsets @ R_W2H

    grid_positions = jnp.concatenate([
        ref_robot[:2] + offsets,
        jnp.full((num_heightscans, num_widthscans, 1), ref_robot[2])
    ], axis=-1)

    center_row = jnp.round(c_h).astype(jnp.int32)
    center_col = jnp.round(c_w).astype(jnp.int32)

    sensor_matrix = grid_positions.at[center_row, center_col].set(ref_robot)

    points, dists = jax.vmap(
        jax.vmap(raycast_sensor_with_dist, in_axes=(None, None, 0)),
        in_axes=(None, None, 0)
    )(mx, dx, sensor_matrix)

    hit_mask = dists >= 0  # True = hit, False = miss

    return points, hit_mask


def create_sensor_matrix(mx, dx, center, yaw=0., key=None, noise_range=0.005,
                         *, dist_x, dist_y, num_heightscans, num_widthscans):
    """
    Create a grid map using ray sensor data.

    This is the main function used to create the grid map using the ray sensor data.

    Args:
        mx: MJX model
        dx: MJX data
        center: Center position [x, y, z] of the robot
        yaw: Yaw angle of the robot
        key: JAX random key for noise
        noise_range: Range of noise to add to sensor positions
        dist_x: Distance between scans in x direction (robot-specific)
        dist_y: Distance between scans in y direction (robot-specific)
        num_heightscans: Number of height scans (robot-specific)
        num_widthscans: Number of width scans (robot-specific)

    Returns:
        Heightmap matrix of shape (num_heightscans, num_widthscans, 3)
    """
    R_W2H = jnp.array([jnp.cos(yaw), jnp.sin(yaw), -jnp.sin(yaw), jnp.cos(yaw)])
    R_W2H = R_W2H.reshape((2, 2))

    c_h = (num_heightscans - 1) / 2
    c_w = (num_widthscans - 1) / 2

    ref_robot = jnp.array([center[0], center[1], center[2] + 0.6])

    idx_h = jnp.arange(num_heightscans)
    idx_w = jnp.arange(num_widthscans)

    p, k = jnp.meshgrid(c_h - idx_h, c_w - idx_w, indexing="ij")
    offsets = jnp.stack([p * dist_x, k * dist_y], axis=-1)  # (H, W, 2)

    if key is not None:
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, shape=offsets.shape, minval=-noise_range, maxval=noise_range)
        offsets = offsets + noise

    offsets = offsets @ R_W2H

    grid_positions = jnp.concatenate([
        ref_robot[:2] + offsets,
        jnp.full((num_heightscans, num_widthscans, 1), ref_robot[2])
    ], axis=-1)

    center_row = jnp.round(c_h).astype(jnp.int32)
    center_col = jnp.round(c_w).astype(jnp.int32)

    sensor_matrix = grid_positions.at[center_row, center_col].set(ref_robot)

    get_data = jax.vmap(
        jax.vmap(raycast_sensor, in_axes=(None, None, 0)),
        in_axes=(None, None, 0)
    )(mx, dx, sensor_matrix)

    return get_data

# JIT-compile a default version for easy use
# Using a 21x21 grid, with points spaced 5cm apart, covering ~1x1 meters
_default_create_matrix = partial(
    create_sensor_matrix,
    dist_x=0.05, dist_y=0.05,
    num_heightscans=21, num_widthscans=21
)
_jit_create_matrix = jax.jit(_default_create_matrix)


def get_heightmap(model: mujoco.MjModel, data: mujoco.MjData, center_pos, yaw=0.0):
    """
    Convenience function to get a local heightmap from CPU MuJoCo data.
    
    This function converts the CPU model/data to MJX, performs the batched raycast,
    and returns the result back as a standard NumPy array. 
    It is ideal for visualization and debugging.

    Args:
        model: Standard mujoco.MjModel
        data: Standard mujoco.MjData
        center_pos: The (x, y, z) position of the robot to center the heightmap around
        yaw: The heading of the robot, to align the grid's forward direction

    Returns:
        points: (21, 21, 3) numpy array containing the 3D surface intersection points
    """
    # 1. Convert to MJX
    # Note: MJX does not support all the flags of standard MuJoCo, e.g. energy
    original_flags = model.opt.enableflags
    model.opt.enableflags = int(model.opt.enableflags) & ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    
    try:
        mx = mjx.put_model(model)
        dx = mjx.put_data(model, data)
    finally:
        # Restore flags
        model.opt.enableflags = original_flags

    # 2. Convert inputs to JAX arrays
    center_jnp = jnp.array(center_pos)
    yaw_jnp = jnp.array(yaw)

    # 3. Call the JIT-compiled scanner
    points_jnp = _jit_create_matrix(mx, dx, center_jnp, yaw_jnp, key=None)
    
    # 4. Bring result back to CPU
    return np.array(points_jnp)
