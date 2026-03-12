#!/usr/bin/env python3
"""Test script for SDF online generator.

Usage:
    python test/test_sdf_generator.py
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np


def test_transforms():
    """Test coordinate transforms."""
    print("\n" + "="*60)
    print("Test 1: Coordinate Transforms")
    print("="*60)

    from mpx.sdf_pretrain.data.transforms import (
        get_camera_transforms,
        local_to_global_points_numpy,
        global_to_local_points_numpy,
        sample_local_queries,
    )

    # Test get_camera_transforms
    pose = np.array([1.0, 2.0, 0.5, 0.0, 0.0, np.pi/4])
    T_to_world, T_to_local = get_camera_transforms(pose)

    print(f"\nPose: {pose}")
    print(f"T_to_world shape: {T_to_world.shape}")
    print(f"T_to_local shape: {T_to_local.shape}")

    # Verify inverses
    product = T_to_world @ T_to_local
    assert np.allclose(product, np.eye(4), atol=1e-10), "Transforms should be inverses!"
    print("✓ Transforms are inverses")

    # Test coordinate conversion
    points_local = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    points_global = local_to_global_points_numpy(points_local, T_to_world)
    points_back = global_to_local_points_numpy(points_global, T_to_local)

    assert np.allclose(points_local, points_back, atol=1e-10), "Round-trip should match!"
    print("✓ Coordinate round-trip works")

    # Test sample_local_queries
    key = jax.random.PRNGKey(0)
    queries = sample_local_queries(key, 100)
    print(f"\nSampled queries shape: {queries.shape}")
    print(f"X range: [{queries[:, 0].min():.3f}, {queries[:, 0].max():.3f}]")
    print(f"Y range: [{queries[:, 1].min():.3f}, {queries[:, 1].max():.3f}]")
    print(f"Z range: [{queries[:, 2].min():.3f}, {queries[:, 2].max():.3f}]")

    print("\n✓ Transforms test passed")


def test_analytical_sdf():
    """Test analytical SDF computation."""
    print("\n" + "="*60)
    print("Test 2: Analytical SDF")
    print("="*60)

    from mpx.sdf_pretrain.data.analytical_sdf import (
        compute_sdf_box_jax,
        get_ground_truth_sdf,
    )

    # Test single box SDF
    box_center = jnp.array([0.0, 0.0, 0.25])
    box_halfsizes = jnp.array([0.5, 0.5, 0.25])

    # Test points
    test_points = jnp.array([
        [0.0, 0.0, 0.25],   # Inside
        [0.6, 0.0, 0.25],   # Outside in X
        [0.0, 0.0, 0.5],    # On surface
    ])

    print("\nTest points and SDF values:")
    for i, point in enumerate(test_points):
        sdf = compute_sdf_box_jax(point, box_center, box_halfsizes)
        status = "INSIDE" if sdf < 0 else ("SURFACE" if sdf == 0 else "OUTSIDE")
        print(f"  Point {i}: {point} -> SDF = {sdf:.4f} ({status})")

    # Test gradient
    grad_fn = jax.grad(compute_sdf_box_jax, argnums=0)
    query_outside = jnp.array([0.6, 0.0, 0.25])
    gradient = grad_fn(query_outside, box_center, box_halfsizes)
    grad_norm = jnp.linalg.norm(gradient)
    print(f"\nGradient at outside point: {gradient}")
    print(f"Gradient norm: {grad_norm:.4f} (should be ~1.0)")

    print("\n✓ Analytical SDF test passed")


def test_online_generator():
    """Test SDF online generator."""
    print("\n" + "="*60)
    print("Test 3: SDF Online Generator")
    print("="*60)

    from mpx.sdf_pretrain.data.dataset_generator import SDFOnlineGenerator

    # Find terrain scene
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(script_dir, "mpx", "data", "r2-1024", "mjcf", "scene_terrain_test.xml")

    if not os.path.exists(xml_path):
        print(f"Warning: Scene file not found: {xml_path}")
        print("Skipping online generator test")
        return

    # Create generator with small batch for testing
    generator = SDFOnlineGenerator(
        xml_path=xml_path,
        heightmap_size=(11, 11),  # Smaller for faster test
        num_queries_per_sample=64,  # Fewer queries
    )

    # Generate a small batch
    print("\nGenerating test batch...")
    key = jax.random.PRNGKey(42)
    batch = generator.generate_batch(key, batch_size=2)

    print("\nBatch contents:")
    print(f"  heightmap: shape={batch['heightmap'].shape}, dtype={batch['heightmap'].dtype}")
    print(f"  queries_local: shape={batch['queries_local'].shape}")
    print(f"  queries_global: shape={batch['queries_global'].shape}")
    print(f"  sdf: shape={batch['sdf'].shape}")
    print(f"  lidar_pose: shape={batch['lidar_pose'].shape}")

    # Check SDF statistics
    sdfs = batch['sdf']
    print(f"\nSDF statistics:")
    print(f"  Min: {float(sdfs.min()):.4f}")
    print(f"  Max: {float(sdfs.max()):.4f}")
    print(f"  Mean: {float(sdfs.mean()):.4f}")

    # Count inside/outside
    inside = (sdfs < 0).sum()
    total = sdfs.size
    print(f"  Inside points: {inside}/{total} ({100*inside/total:.1f}%)")

    print("\n✓ Online generator test passed")


def main():
    print("\n" + "="*60)
    print("SDF Pretrain Module Test Suite")
    print("="*60)

    test_transforms()
    test_analytical_sdf()
    test_online_generator()

    print("\n" + "="*60)
    print("All SDF pretrain tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
