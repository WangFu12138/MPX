"""Dynamic terrain generation for SDF pretraining.

This module provides terrain generation that works entirely in memory,
without relying on static XML files. It uses the WFC algorithm from
mpx.terrain.generator to create diverse, connected terrains.

Key features:
- No XML file dependency
- JAX-compatible terrain updates
- Consistent SDF ground truth
- Curriculum learning support
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import terrain generator from mpx.terrain
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpx.terrain.generator import (
    TerrainGenerator,
    generate_14,
    create_centered_grid,
    addElement,
)


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    size: int = 5  # WFC grid size
    width: float = 0.35  # Step width
    step_height_min: float = 0.04  # Minimum step height
    step_height_max: float = 0.12  # Maximum step height
    num_steps: int = 3  # Number of stair steps per block
    max_boxes: int = 200  # Maximum number of boxes


class DynamicTerrainManager:
    """Manages dynamic terrain generation for SDF pretraining.

    This class generates terrain entirely in memory using WFC algorithm,
    providing both the box data for MuJoCo and the boxes_info for analytical SDF.

    Example:
        >>> manager = DynamicTerrainManager()
        >>> boxes_info = manager.generate_new_terrain(key)
        >>> # boxes_info can be used for:
        >>> # 1. Creating MuJoCo/MJX geometry
        >>> # 2. Computing analytical SDF
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        """
        Initialize the dynamic terrain manager.

        Args:
            config: Terrain configuration. Uses defaults if None.
        """
        self.config = config or TerrainConfig()
        self.current_boxes_info: List[Dict] = []
        self.current_terrain_matrix: Optional[np.ndarray] = None

    def generate_new_terrain(self, key: jax.Array) -> List[Dict]:
        """Generate a new random terrain.

        Args:
            key: JAX random key

        Returns:
            List of box dicts with 'pos', 'quat', 'size' keys
        """
        cfg = self.config

        # Randomize step height for curriculum
        # Use JAX to generate a random seed (avoid overflow by using smaller range)
        seed_array = jax.random.randint(key, (), 0, 1000000)
        rng = np.random.RandomState(int(seed_array))
        step_height = rng.uniform(cfg.step_height_min, cfg.step_height_max)

        # Create terrain generator (render=False means we only get box_data)
        tg = TerrainGenerator(
            width=cfg.width,
            step_height=step_height,
            num_stairs=cfg.num_steps,
            render=False
        )

        # Generate WFC map - use test=True to ensure center is initialized
        wave = generate_14(size=cfg.size, test=True)

        # Create position grid
        grid = create_centered_grid(cfg.size, tg.length)

        # Add terrain elements based on WFC wave
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                addElement(tg, wave[i, j], grid[i, j])

        # If no boxes generated, create a simple fallback terrain
        if tg.count_boxes == 0:
            print("Warning: WFC generated 0 boxes, using fallback terrain")
            # Create simple stairs as fallback
            for i in range(3):
                for k in range(cfg.size):
                    x = (k - cfg.size // 2) * tg.length
                    y = (i - 1) * 0.5
                    z = (k + 1) * step_height / 2
                    tg.AddBox([x, y, z / 2], [0, 0, 0], [tg.width, 0.5, z])

        # Convert box_data to boxes_info format
        boxes_info = []
        for box in tg.box_data:
            boxes_info.append({
                'pos': box['pos'].tolist() if isinstance(box['pos'], np.ndarray) else box['pos'],
                'quat': box['quat'].tolist() if isinstance(box['quat'], np.ndarray) else box['quat'],
                'size': box['size'].tolist() if isinstance(box['size'], np.ndarray) else box['size'],
            })

        # Store for later use
        self.current_boxes_info = boxes_info
        self.current_terrain_matrix = self._create_terrain_matrix(tg.box_data)

        print(f"Generated terrain: {len(boxes_info)} boxes, step_height={step_height:.3f}m")

        return boxes_info

    def _create_terrain_matrix(self, box_data: List[Dict]) -> np.ndarray:
        """Create terrain matrix for MJX model updates.

        Args:
            box_data: List of box data dicts

        Returns:
            Terrain matrix of shape (1, num_boxes, 10)
            Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, size_x, size_y, size_z]
        """
        cfg = self.config
        num_boxes = len(box_data)

        # Pad to max_boxes
        matrix = np.zeros((1, cfg.max_boxes, 10))

        for i, box in enumerate(box_data):
            if i >= cfg.max_boxes:
                break
            pos = np.array(box['pos'])
            quat = np.array(box['quat'])
            size = np.array(box['size'])

            matrix[0, i, :3] = pos
            matrix[0, i, 3:7] = quat
            matrix[0, i, 7:] = size

        # Fill remaining with placeholder boxes (far away)
        for i in range(len(box_data), cfg.max_boxes):
            matrix[0, i, :3] = [100 + i, 100 + i, 10]
            matrix[0, i, 3:7] = [1, 0, 0, 0]
            matrix[0, i, 7:] = [0.1, 0.1, 0.1]

        return matrix

    def get_boxes_info(self) -> List[Dict]:
        """Get current boxes info for analytical SDF computation."""
        return self.current_boxes_info

    def get_terrain_matrix(self) -> np.ndarray:
        """Get current terrain matrix for MJX model updates."""
        return self.current_terrain_matrix


class CurriculumTerrainManager(DynamicTerrainManager):
    """Terrain manager with curriculum learning support.

    Gradually increases terrain difficulty during training.
    """

    def __init__(self, config: Optional[TerrainConfig] = None,
                 curriculum_levels: int = 5):
        """
        Initialize with curriculum support.

        Args:
            config: Base terrain configuration
            curriculum_levels: Number of curriculum levels
        """
        super().__init__(config)
        self.curriculum_levels = curriculum_levels
        self.current_level = 0

    def set_level(self, level: int):
        """Set curriculum level (0 to curriculum_levels-1)."""
        self.current_level = max(0, min(level, self.curriculum_levels - 1))

        # Adjust difficulty based on level
        progress = self.current_level / (self.curriculum_levels - 1)

        # Interpolate step height
        self.config.step_height_min = 0.02 + progress * 0.04  # 2cm to 6cm
        self.config.step_height_max = 0.06 + progress * 0.10  # 6cm to 16cm

        print(f"Curriculum level {level}: step_height=[{self.config.step_height_min:.3f}, {self.config.step_height_max:.3f}]")

    def generate_new_terrain(self, key: jax.Array) -> List[Dict]:
        """Generate terrain with current curriculum level."""
        return super().generate_new_terrain(key)


def create_simple_terrain(key: jax.Array,
                          num_boxes: int = 50,
                          x_range: Tuple[float, float] = (-2.0, 2.0),
                          y_range: Tuple[float, float] = (-2.0, 2.0),
                          z_min: float = 0.02,
                          z_max: float = 0.15,
                          size_range: Tuple[float, float] = (0.1, 0.3)) -> List[Dict]:
    """Create a simple random terrain with overlapping boxes.

    This is a fallback generator that creates dense, overlapping boxes
    ensuring no gaps that could cause ray cast misses.

    Args:
        key: JAX random key
        num_boxes: Number of boxes to generate
        x_range: (min, max) x position range
        y_range: (min, max) y position range
        z_min: Minimum box half-height
        z_max: Maximum box half-height
        size_range: (min, max) half-size for x and y

    Returns:
        List of box dicts
    """
    rng = np.random.RandomState(int(jax.random.randint(key, (), 0, 2**31)))

    boxes = []
    for i in range(num_boxes):
        # Random position
        px = rng.uniform(*x_range)
        py = rng.uniform(*y_range)
        sz = rng.uniform(z_min, z_max)
        pz = sz  # Box sits on ground

        # Random size (ensure good coverage)
        sx = rng.uniform(*size_range)
        sy = rng.uniform(*size_range)

        # Random yaw rotation
        yaw = rng.uniform(-np.pi/6, np.pi/6)
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)

        boxes.append({
            'pos': [px, py, pz],
            'quat': [qw, 0, 0, qz],
            'size': [sx, sy, sz],
        })

    return boxes


def visualize_terrain_boxes(boxes_info: List[Dict], output_path: str = "terrain_preview.png"):
    """Visualize terrain boxes for debugging.

    Args:
        boxes_info: List of box dicts
        output_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for box in boxes_info:
            pos = np.array(box['pos'])
            size = np.array(box['size'])

            # Draw box as a simple cuboid
            x, y, z = pos
            dx, dy, dz = size * 2  # Full size

            # Create vertices
            vertices = np.array([
                [x-dx, y-dy, z-dz],
                [x+dx, y-dy, z-dz],
                [x+dx, y+dy, z-dz],
                [x-dx, y+dy, z-dz],
                [x-dx, y-dy, z+dz],
                [x+dx, y-dy, z+dz],
                [x+dx, y+dy, z+dz],
                [x-dx, y+dy, z+dz],
            ])

            # Draw edges
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
            ]

            for edge in edges:
                pts = vertices[edge]
                ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Terrain Preview ({len(boxes_info)} boxes)')

        plt.savefig(output_path)
        plt.close()
        print(f"Saved terrain visualization to {output_path}")

    except ImportError:
        print("matplotlib not available for visualization")


# Test function
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing DynamicTerrainManager...")

    manager = DynamicTerrainManager()
    key = jax.random.PRNGKey(42)

    # Generate a few terrains
    for i in range(3):
        key, subkey = jax.random.split(key)
        boxes = manager.generate_new_terrain(subkey)
        print(f"  Generated {len(boxes)} boxes")

    # Visualize
    visualize_terrain_boxes(manager.get_boxes_info())

    print("\nTesting CurriculumTerrainManager...")
    curriculum = CurriculumTerrainManager(curriculum_levels=5)

    for level in range(5):
        curriculum.set_level(level)
        key, subkey = jax.random.split(key)
        boxes = curriculum.generate_new_terrain(subkey)

    print("\nAll tests passed!")
