"""Terrain generation module for MPX.

This module provides procedural terrain generation using Wave Function Collapse (WFC)
algorithm and utilities for terrain perception.

Key components:
- generator: Procedural terrain generation with stairs, platforms, rough ground
- heightmap: Ray casting-based local heightmap extraction
- wfc: Wave Function Collapse solver
- get_indexes: Terrain tile indexing utilities
"""

from mpx.terrain.generator import (
    TerrainGenerator,
    generate_14,
    addElement,
    create_centered_grid,
    create_random_matrix,
    random_test_env,
    fill_terrain_template,
)
from mpx.terrain.heightmap import raycast_sensor, create_sensor_matrix

__all__ = [
    # Generator
    "TerrainGenerator",
    "generate_14",
    "addElement",
    "create_centered_grid",
    "create_random_matrix",
    "random_test_env",
    "fill_terrain_template",
    # Heightmap
    "raycast_sensor",
    "create_sensor_matrix",
]
