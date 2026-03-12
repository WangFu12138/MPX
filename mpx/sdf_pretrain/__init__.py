"""SDF Pretraining module for visual perception in legged locomotion.

This module provides end-to-end SDF (Signed Distance Field) prediction from
heightmap images, enabling terrain-aware control for bipedal robots.

Key components:
- data: Online dataset generation and coordinate transforms
- models: Neural network architectures (HeightmapEncoder, ImplicitSDFDecoder)
- train: Training loop with JAX/Flax/Optax
"""

from mpx.sdf_pretrain.data import (
    SDFOnlineGenerator,
    parse_xml_to_boxes,
    get_camera_transforms,
    local_to_global_points,
    global_to_local_points,
    compute_sdf_box_jax,
    compute_sdf_box_oriented_jax,
    get_ground_truth_sdf,
)

__all__ = [
    # Data generation
    "SDFOnlineGenerator",
    "parse_xml_to_boxes",
    "get_camera_transforms",
    "local_to_global_points",
    "global_to_local_points",
    "compute_sdf_box_jax",
    "compute_sdf_box_oriented_jax",
    "get_ground_truth_sdf",
]
