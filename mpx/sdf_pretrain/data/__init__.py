"""Data generation and coordinate transform utilities for SDF pretraining.

This module provides:
- SDFOnlineGenerator: Online data generator for SDF training
- Coordinate transforms: LiDAR/camera to world frame conversions
- SDF computation: Analytical SDF for boxes and terrains
"""

from mpx.sdf_pretrain.data.dataset_generator import SDFOnlineGenerator
from mpx.sdf_pretrain.data.transforms import (
    get_camera_transforms,
    local_to_global_points,
    global_to_local_points,
    sample_local_queries,
)
from mpx.sdf_pretrain.data.analytical_sdf import (
    parse_xml_to_boxes,
    compute_sdf_box_numpy,
    compute_sdf_box_jax,
    compute_sdf_box_oriented_jax,
    get_ground_truth_sdf,
)

__all__ = [
    # Dataset generator
    "SDFOnlineGenerator",
    # Transforms
    "get_camera_transforms",
    "local_to_global_points",
    "global_to_local_points",
    "sample_local_queries",
    # SDF computation
    "parse_xml_to_boxes",
    "compute_sdf_box_numpy",
    "compute_sdf_box_jax",
    "compute_sdf_box_oriented_jax",
    "get_ground_truth_sdf",
]
