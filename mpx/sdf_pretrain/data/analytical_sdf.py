"""Analytical SDF computation for terrain boxes.

This module provides functions to compute ground truth SDF values for
terrain composed of axis-aligned or oriented boxes.

Key functions:
- parse_xml_to_boxes: Extract box geometry from MuJoCo XML
- compute_sdf_box_numpy: NumPy SDF for axis-aligned boxes
- compute_sdf_box_jax: JAX SDF for axis-aligned boxes
- compute_sdf_box_oriented_jax: JAX SDF for oriented boxes
- get_ground_truth_sdf: Compute SDF for multiple query points
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict

import jax
import jax.numpy as jnp


def parse_xml_to_boxes(xml_path: str, terrain_only: bool = True) -> List[Dict]:
    """
    Parse MuJoCo XML file and extract all box geometries.

    Args:
        xml_path: Path to the MuJoCo XML file
        terrain_only: If True, only extract terrain boxes (names starting with 'box_')
                      If False, extract all boxes including robot parts

    Returns:
        List of dicts, each containing:
            - 'name': box name
            - 'pos': center position [x, y, z]
            - 'quat': quaternion [w, x, y, z]
            - 'size': half-sizes [sx, sy, sz] (MuJoCo convention)
    """
    boxes = []

    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found: {xml_path}")
        return boxes

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        print("Warning: No worldbody found in XML")
        return boxes

    # Iterate through all bodies
    for body in worldbody.findall('.//body'):
        body_name = body.get('name', 'unnamed')

        # Filter: only terrain boxes if terrain_only=True
        if terrain_only and not body_name.startswith('box_'):
            continue

        pos_str = body.get('pos', '0 0 0')
        quat_str = body.get('quat', '1 0 0 0')

        # Parse position and quaternion
        pos = np.array([float(x) for x in pos_str.split()])
        quat = np.array([float(x) for x in quat_str.split()])

        # Find box geometry
        for geom in body.findall('geom'):
            geom_type = geom.get('type', 'sphere')
            if geom_type == 'box':
                size_str = geom.get('size', '0.1 0.1 0.1')
                size = np.array([float(x) for x in size_str.split()])

                boxes.append({
                    'name': body_name,
                    'pos': pos,
                    'quat': quat,
                    'size': size
                })

    return boxes


def compute_sdf_box_numpy(query: np.ndarray, box_center: np.ndarray,
                          box_halfsizes: np.ndarray) -> float:
    """
    Compute signed distance from a point to an axis-aligned box (NumPy version).

    SDF formula: length(max(q,0)) + min(max(q),0)
    - If point is outside: returns positive distance to surface
    - If point is inside: returns negative distance to nearest surface

    Args:
        query: Query point [x, y, z]
        box_center: Box center [x, y, z]
        box_halfsizes: Box half-sizes [sx, sy, sz]

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    # Transform query to box local frame
    local = query - box_center

    # q = |local| - halfsizes
    # Positive q means outside on that axis, negative means inside
    q = np.abs(local) - box_halfsizes

    # Correct SDF formula:
    # - outside_dist: distance in the "outside region" (axes where q > 0)
    # - inside_dist: 0 if any axis is outside, otherwise min(q)
    outside_dist = np.linalg.norm(np.maximum(q, 0))
    inside_dist = np.minimum(np.max(q), 0)

    # Combined SDF
    sdf = outside_dist + inside_dist

    return sdf


def compute_sdf_box_jax(query: jnp.ndarray, box_center: jnp.ndarray,
                        box_halfsizes: jnp.ndarray) -> jnp.ndarray:
    """
    Compute signed distance from a point to an axis-aligned box (JAX version).

    SDF formula: length(max(q,0)) + min(max(q),0)

    Args:
        query: Query point [x, y, z] (jax array)
        box_center: Box center [x, y, z] (jax array)
        box_halfsizes: Box half-sizes [sx, sy, sz] (jax array)

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    # Transform query to box local frame
    local = query - box_center

    # q = |local| - halfsizes
    q = jnp.abs(local) - box_halfsizes

    # Correct SDF formula
    outside_dist = jnp.linalg.norm(jnp.maximum(q, 0))
    inside_dist = jnp.minimum(jnp.max(q), 0)

    sdf = outside_dist + inside_dist

    return sdf


def compute_sdf_box_oriented_jax(query: jnp.ndarray, box_center: jnp.ndarray,
                                  box_quat: jnp.ndarray, box_halfsizes: jnp.ndarray) -> jnp.ndarray:
    """
    Compute signed distance from a point to an oriented box (JAX version).

    SDF formula: length(max(q,0)) + min(max(q),0)

    Args:
        query: Query point [x, y, z] (jax array)
        box_center: Box center [x, y, z] (jax array)
        box_quat: Box quaternion [w, x, y, z] (jax array)
        box_halfsizes: Box half-sizes [sx, sy, sz] (jax array)

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    # Convert quaternion to rotation matrix
    w, x, y, z = box_quat

    # Normalize quaternion
    norm = jnp.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Rotation matrix from quaternion
    R = jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    # Transform query to box local frame
    local = R.T @ (query - box_center)

    # Compute SDF for axis-aligned box in local frame
    q = jnp.abs(local) - box_halfsizes
    outside_dist = jnp.linalg.norm(jnp.maximum(q, 0))
    inside_dist = jnp.minimum(jnp.max(q), 0)
    sdf = outside_dist + inside_dist

    return sdf


def get_ground_truth_sdf(queries_global: jnp.ndarray, boxes_info: List[Dict]) -> jnp.ndarray:
    """
    Compute ground truth SDF for multiple query points against all boxes.

    Uses JAX for efficient batched computation with vmap.

    Args:
        queries_global: Query points in world frame, shape (N, 3)
        boxes_info: List of box dicts from parse_xml_to_boxes

    Returns:
        SDF values, shape (N,)
    """
    # Compute SDF for each box and take minimum
    def compute_single_sdf(query):
        min_sdf = jnp.inf
        for box in boxes_info:
            sdf = compute_sdf_box_oriented_jax(
                query,
                jnp.array(box['pos']),
                jnp.array(box['quat']),
                jnp.array(box['size'])
            )
            min_sdf = jnp.minimum(min_sdf, sdf)
        return min_sdf

    # Vectorize over all query points
    compute_batch_sdf = jax.vmap(compute_single_sdf)
    sdfs = compute_batch_sdf(queries_global)

    return sdfs


def get_ground_truth_sdf_with_floor(queries_global: jnp.ndarray,
                                     boxes_info: List[Dict],
                                     floor_height: float = 0.0) -> jnp.ndarray:
    """
    Compute ground truth SDF including a floor plane at z=floor_height.

    The floor is modeled as a half-space: SDF = z - floor_height

    Args:
        queries_global: Query points in world frame, shape (N, 3)
        boxes_info: List of box dicts from parse_xml_to_boxes
        floor_height: Height of the floor plane

    Returns:
        SDF values, shape (N,)
    """
    # Compute SDF for boxes
    boxes_sdf = get_ground_truth_sdf(queries_global, boxes_info)

    # Compute SDF for floor (negative below floor, positive above)
    floor_sdf = queries_global[:, 2] - floor_height

    # Take minimum (union of SDFs)
    combined_sdf = jnp.minimum(boxes_sdf, floor_sdf)

    return combined_sdf
