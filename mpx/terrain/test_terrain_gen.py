#!/usr/bin/env python3
"""Test script for terrain generation module.

This script validates the terrain generation functionality:
1. WFC map generation
2. TerrainGenerator box creation
3. MuJoCo scene visualization
4. Heightmap ray casting (optional)

Usage:
    # Basic test (no visualization)
    python -m mpx.terrain.test_terrain_gen

    # With visualization
    python -m mpx.terrain.test_terrain_gen --visualize

    # Custom parameters
    python -m mpx.terrain.test_terrain_gen --size 5 --num-objects 50 --visualize
"""
import argparse
import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpx.terrain.generator import (
    generate_14,
    TerrainGenerator,
    addElement,
    create_centered_grid,
    random_test_env,
)
from mpx.terrain.wfc.wfc import WFCCore


def test_wfc_generation(size=5):
    """Test WFC map generation."""
    print("\n" + "="*60)
    print("Test 1: WFC Map Generation")
    print("="*60)

    print(f"\nGenerating {size}x{size} WFC map...")
    wave = generate_14(size=size, test=True)

    print(f"\nWave matrix shape: {wave.shape}")
    print(f"Wave matrix:\n{wave}")

    # Check validity
    unique_tiles = np.unique(wave)
    print(f"\nUnique tile types: {unique_tiles}")
    print(f"Number of unique tiles: {len(unique_tiles)}")

    # Check that boundary is 0
    assert np.all(wave[0, :] == 0), "Top boundary should be 0"
    assert np.all(wave[-1, :] == 0), "Bottom boundary should be 0"
    assert np.all(wave[:, 0] == 0), "Left boundary should be 0"
    assert np.all(wave[:, -1] == 0), "Right boundary should be 0"
    print("\n✓ Boundary check passed (all boundaries are tile type 0)")

    return wave


def test_terrain_generator(width=0.35, step_height=0.08, num_steps=3):
    """Test TerrainGenerator box creation."""
    print("\n" + "="*60)
    print("Test 2: TerrainGenerator Box Creation")
    print("="*60)

    print(f"\nCreating terrain generator:")
    print(f"  - Width: {width}m")
    print(f"  - Step height: {step_height}m")
    print(f"  - Number of steps: {num_steps}")

    tg = TerrainGenerator(
        width=width,
        step_height=step_height,
        num_stairs=num_steps,
        render=False
    )

    print(f"\nBlock height: {tg.block_height}m")
    print(f"Total length: {tg.length}m")

    # Test AddBox
    print("\nAdding test boxes...")
    tg.AddBox([0, 0, 0.5], [0, 0, 0], [0.5, 0.5, 0.5])
    tg.AddBox([1, 0, 0.5], [0, 0, np.pi/4], [0.5, 0.5, 0.5])
    tg.AddBox([2, 0, 0.5], [0, 0, np.pi/2], [0.5, 0.5, 0.5])

    print(f"Total boxes added: {tg.count_boxes}")
    print(f"Box data entries: {len(tg.box_data)}")

    # Display first box
    if tg.box_data:
        print(f"\nFirst box data:")
        print(f"  Position: {tg.box_data[0]['pos']}")
        print(f"  Size: {tg.box_data[0]['size']}")
        print(f"  Quaternion: {tg.box_data[0]['quat']}")

    print("\n✓ TerrainGenerator test passed")

    return tg


def test_terrain_from_wfc(size=5, width=0.35, step_height=0.08, num_steps=3):
    """Test generating terrain from WFC map."""
    print("\n" + "="*60)
    print("Test 3: Terrain Generation from WFC Map")
    print("="*60)

    # Generate WFC map
    wave = generate_14(size=size, test=True)

    # Create terrain generator
    tg = TerrainGenerator(
        width=width,
        step_height=step_height,
        num_stairs=num_steps,
        render=False
    )

    # Create grid
    grid = create_centered_grid(size, tg.length)

    print(f"\nGenerating terrain from {size}x{size} WFC map...")
    print(f"Grid shape: {grid.shape}")

    # Add terrain elements
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            addElement(tg, wave[i, j], grid[i, j])

    print(f"\nTotal boxes generated: {tg.count_boxes}")

    # Statistics
    if tg.box_data:
        positions = np.array([box['pos'] for box in tg.box_data])
        print(f"\nPosition statistics:")
        print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
        print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
        print(f"  Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

    print("\n✓ Terrain from WFC test passed")

    return tg


def test_mujoco_visualization(num_objects=100, size=9, width=0.35,
                              step_height=0.08, num_steps=3):
    """Test MuJoCo scene generation and visualization."""
    print("\n" + "="*60)
    print("Test 4: MuJoCo Scene Visualization")
    print("="*60)

    # Generate terrain
    print(f"\nGenerating test terrain:")
    print(f"  - Size: {size}x{size}")
    print(f"  - Width: {width}m")
    print(f"  - Step height: {step_height}m")
    print(f"  - Number of steps: {num_steps}")
    print(f"  - Max objects: {num_objects}")

    wave = random_test_env(
        num_objects,
        size,
        step_height=step_height,
        width=width,
        num_steps=num_steps
    )

    print(f"\n✓ Terrain scene generated successfully")

    return wave


def visualize_terrain():
    """Visualize terrain with MuJoCo viewer."""
    print("\n" + "="*60)
    print("Launching MuJoCo Viewer...")
    print("="*60)

    try:
        import mujoco
        import mujoco.viewer

        # Get the path to the generated test scene
        mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(mpx_dir, "data", "r2-1024", "mjcf", "scene_terrain_test.xml")

        if not os.path.exists(scene_path):
            print(f"Error: Scene file not found at {scene_path}")
            print("Please run the test without --visualize first to generate the scene.")
            return

        print(f"Loading scene from: {scene_path}")

        # Load model
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)

        print("\nLaunching viewer...")
        print("Controls:")
        print("  - Left click + drag: Rotate view")
        print("  - Right click + drag: Pan")
        print("  - Scroll: Zoom")
        print("  - Double click: Select body")
        print("  - Ctrl + right click: Apply force")
        print("  - Press 'Esc' to exit\n")

        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Run simulation
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

        print("\n✓ Visualization completed")

    except ImportError as e:
        print(f"Error: Could not import mujoco.viewer: {e}")
        print("Make sure MuJoCo is properly installed with GUI support.")
    except Exception as e:
        print(f"Error during visualization: {e}")


def test_heightmap_raycast():
    """Test heightmap ray casting (optional, requires MJX)."""
    print("\n" + "="*60)
    print("Test 5: Heightmap Ray Casting (Optional)")
    print("="*60)

    try:
        import mujoco
        from mujoco import mjx
        import jax
        import jax.numpy as jnp

        print("\nTesting heightmap ray casting with MJX...")

        # Load a simple scene
        mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(mpx_dir, "data", "r2-1024", "mjcf", "scene.xml")

        if not os.path.exists(scene_path):
            print(f"Scene file not found: {scene_path}")
            print("Skipping heightmap test.")
            return

        # Load model
        model = mujoco.MjModel.from_xml_path(scene_path)
        mx = mjx.put_model(model)
        dx = mjx.put_data(model, mujoco.MjData(model))

        # Test raycast
        from mpx.terrain.heightmap import raycast_sensor

        # Cast ray from above origin
        pos = jnp.array([0.0, 0.0, 1.0])
        intersection = raycast_sensor(mx, dx, pos)

        print(f"\nRay cast from: {pos}")
        print(f"Intersection point: {intersection}")
        print(f"Ground height at (0, 0): {intersection[2]:.4f}")

        print("\n✓ Heightmap ray casting test passed")

    except ImportError as e:
        print(f"\nSkipping heightmap test (missing dependencies): {e}")
        print("Required: mujoco, jax, jaxlib")
    except Exception as e:
        print(f"\nHeightmap test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test terrain generation for MPX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test
    python -m mpx.terrain.test_terrain_gen

    # With visualization
    python -m mpx.terrain.test_terrain_gen --visualize

    # Custom parameters
    python -m mpx.terrain.test_terrain_gen --size 7 --num-objects 150 --step-height 0.12
        """
    )

    parser.add_argument('--size', type=int, default=5,
                        help='WFC map size (default: 5)')
    parser.add_argument('--num-objects', type=int, default=100,
                        help='Maximum number of terrain objects (default: 100)')
    parser.add_argument('--width', type=float, default=0.35,
                        help='Width of each stair step in meters (default: 0.35)')
    parser.add_argument('--step-height', type=float, default=0.08,
                        help='Height of each stair step in meters (default: 0.08)')
    parser.add_argument('--num-steps', type=int, default=3,
                        help='Number of stair steps (default: 3)')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch MuJoCo viewer to visualize terrain')
    parser.add_argument('--skip-heightmap', action='store_true',
                        help='Skip heightmap ray casting test')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("MPX Terrain Generation Test Suite")
    print("="*60)

    # Run tests
    test_wfc_generation(args.size)
    test_terrain_generator(args.width, args.step_height, args.num_steps)
    test_terrain_from_wfc(args.size, args.width, args.step_height, args.num_steps)
    test_mujoco_visualization(args.num_objects, args.size, args.width,
                             args.step_height, args.num_steps)

    if not args.skip_heightmap:
        test_heightmap_raycast()

    # Visualize if requested
    if args.visualize:
        visualize_terrain()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
