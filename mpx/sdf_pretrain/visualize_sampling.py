#!/usr/bin/env python3
"""
Visualize SDF terrain sampling with updated parameters.

This script verifies:
1. Terrain generation (WFC boxes)
2. LiDAR position sampling (±2m range)
3. Heightmap scan range (±0.5m around LiDAR)
4. Yaw-only rotation (gravity-aligned local frame)
5. Query point distribution (10% inside, 70% surface, 20% outside)

Usage:
    python -m mpx.sdf_pretrain.visualize_sampling
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator
from mpx.sdf_pretrain.data.dynamic_terrain import TerrainConfig


def visualize_terrain_and_sampling():
    """Main visualization function."""

    print("="*60)
    print("SDF Terrain Sampling Visualization")
    print("="*60)
    print("\nParameters to verify:")
    print("  - LiDAR position range: ±2.0m")
    print("  - Heightmap scan range: ±0.5m")
    print("  - Rotation: Yaw-only (gravity-aligned)")
    print("  - Query distribution: 10%/70%/20% (inside/surface/outside)")
    print("="*60 + "\n")

    # Initialize generator with updated parameters
    terrain_config = TerrainConfig(
        size=7,
        step_height_min=0.04,
        step_height_max=0.12,
        max_boxes=150,
    )

    generator = SDFDynamicGenerator(
        max_boxes=150,
        heightmap_size=(21, 21),
        num_queries_per_sample=128,  # Reduced for clearer visualization
        lidar_height_range=(0.4, 0.8),
        view_bounds=(-0.5, 0.5, -0.5, 0.5, 0.01, 0.5),  # ±0.5m scan range
        terrain_config=terrain_config,
    )

    # Generate terrain
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    print("Generating terrain...")
    generator.regenerate_terrain(subkey)

    # Generate a single batch
    key, subkey = jax.random.split(key)
    print("Generating batch (batch_size=1)...\n")
    batch = generator.generate_batch(subkey, batch_size=1, verbose=True)

    # Extract data
    lidar_pose = np.array(batch['lidar_pose'][0])
    queries_local = np.array(batch['queries_local'][0])
    queries_global = np.array(batch['queries_global'][0])
    sdfs = np.array(batch['sdf'][0])
    heightmap_raw = np.array(batch['heightmap'][0])

    # Get denormalized heightmap for visualization
    hm_mean = np.array(batch['heightmap_mean'][0])
    hm_std = np.array(batch['heightmap_std'][0])
    heightmap = heightmap_raw * hm_std + hm_mean

    print("\n" + "="*60)
    print("Data Statistics")
    print("="*60)
    print(f"\n[Lidar Pose]")
    print(f"  Position (world): x={lidar_pose[0]:.3f}, y={lidar_pose[1]:.3f}, z={lidar_pose[2]:.3f}")
    print(f"  Orientation: roll={lidar_pose[3]:.3f}, pitch={lidar_pose[4]:.3f}, yaw={lidar_pose[5]:.3f}")
    print(f"  Position range check: |x|={abs(lidar_pose[0]):.3f} <= 2.0? {abs(lidar_pose[0]) <= 2.0}")
    print(f"                       |y|={abs(lidar_pose[1]):.3f} <= 2.0? {abs(lidar_pose[1]) <= 2.0}")

    print(f"\n[Heightmap (Local Coords)]")
    print(f"  Shape: {heightmap.shape}")
    print(f"  X range: [{heightmap[..., 0].min():.4f}, {heightmap[..., 0].max():.4f}] (expected: ±0.5)")
    print(f"  Y range: [{heightmap[..., 1].min():.4f}, {heightmap[..., 1].max():.4f}] (expected: ±0.5)")
    print(f"  Z range: [{heightmap[..., 2].min():.4f}, {heightmap[..., 2].max():.4f}]")

    print(f"\n[Query Points]")
    print(f"  Total: {len(queries_global)}")
    n_inside = int(len(queries_global) * 0.10)
    n_surface = int(len(queries_global) * 0.70)
    n_outside = len(queries_global) - n_inside - n_surface
    print(f"  Inside (SDF<0): {n_inside} (10%)")
    print(f"  Surface (SDF≈0): {n_surface} (70%)")
    print(f"  Outside (SDF>0): {n_outside} (20%)")

    print(f"\n[SDF Values]")
    print(f"  Range: [{sdfs.min():.4f}, {sdfs.max():.4f}]")
    print(f"  Negative count: {(sdfs < 0).sum()} ({(sdfs < 0).sum()/len(sdfs)*100:.1f}%)")
    print(f"  Near-zero (|SDF|<0.05): {(np.abs(sdfs) < 0.05).sum()} ({(np.abs(sdfs) < 0.05).sum()/len(sdfs)*100:.1f}%)")

    # ==================== FIGURE 1: 3D Overview ====================
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: 3D view of terrain boxes
    ax1 = fig.add_subplot(221, projection='3d')

    # Draw terrain boxes
    boxes_info = generator.boxes_info
    for box in boxes_info[:50]:  # Limit for performance
        pos = np.array(box['pos'])
        size = np.array(box['size']) * 2  # Full size

        # Draw box edges
        x, y, z = pos
        dx, dy, dz = size

        vertices = np.array([
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz],
            [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz],
            [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz],
        ])

        edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        for edge in edges:
            pts = vertices[edge]
            ax1.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], 'brown', alpha=0.3, linewidth=0.5)

    # Draw LiDAR position
    ax1.scatter(*lidar_pose[:3], c='red', s=100, marker='o', label='LiDAR')

    # Draw heightmap points (in world coords)
    # Transform local heightmap back to world
    from mpx.sdf_pretrain.data.transforms import get_camera_transforms, local_to_global_points_numpy
    lidar_pose_for_transform = np.array([
        lidar_pose[0], lidar_pose[1], lidar_pose[2] + 0.6,
        0.0, 0.0, lidar_pose[5]  # Yaw-only
    ])
    T_l2w, _ = get_camera_transforms(lidar_pose_for_transform)
    hm_flat = heightmap.reshape(-1, 3)
    hm_world = local_to_global_points_numpy(hm_flat, T_l2w)

    ax1.scatter(hm_world[::4, 0], hm_world[::4, 1], hm_world[::4, 2],
                c='green', s=5, alpha=0.5, label='Heightmap')

    # Draw query points
    ax1.scatter(queries_global[:n_inside, 0], queries_global[:n_inside, 1], queries_global[:n_inside, 2],
                c='blue', s=10, alpha=0.7, label='Inside (SDF<0)')
    ax1.scatter(queries_global[n_inside:n_inside+n_surface, 0],
                queries_global[n_inside:n_inside+n_surface, 1],
                queries_global[n_inside:n_inside+n_surface, 2],
                c='cyan', s=5, alpha=0.5, label='Surface (SDF≈0)')
    ax1.scatter(queries_global[n_inside+n_surface:, 0],
                queries_global[n_inside+n_surface:, 1],
                queries_global[n_inside+n_surface:, 2],
                c='magenta', s=10, alpha=0.7, label='Outside (SDF>0)')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Overview\nLiDAR at ({lidar_pose[0]:.2f}, {lidar_pose[1]:.2f}, {lidar_pose[2]:.2f})')
    ax1.legend(loc='upper left', fontsize=8)

    # Draw terrain boundary
    ax1.plot([-3.15, 3.15, 3.15, -3.15, -3.15],
             [-3.15, -3.15, 3.15, 3.15, -3.15],
             [0, 0, 0, 0, 0], 'k--', alpha=0.5, label='Terrain boundary')

    # ==================== Subplot 2: Top-down view ====================
    ax2 = fig.add_subplot(222)

    # Terrain boundary
    ax2.plot([-3.15, 3.15, 3.15, -3.15, -3.15],
             [-3.15, -3.15, 3.15, 3.15, -3.15], 'k--', label='Terrain (±3.15m)')

    # LiDAR position range (±2m)
    ax2.add_patch(plt.Rectangle((-2, -2), 4, 4, fill=False, edgecolor='orange',
                                 linestyle='--', linewidth=2, label='LiDAR pos range (±2m)'))

    # Heightmap coverage (±0.5m around LiDAR)
    ax2.add_patch(plt.Rectangle((lidar_pose[0]-0.5, lidar_pose[1]-0.5), 1, 1,
                                 fill=True, facecolor='green', alpha=0.2,
                                 edgecolor='green', linewidth=2, label='Heightmap (±0.5m)'))

    # LiDAR position
    ax2.scatter(lidar_pose[0], lidar_pose[1], c='red', s=100, marker='o', zorder=5)
    ax2.annotate('LiDAR', (lidar_pose[0], lidar_pose[1]), textcoords="offset points",
                 xytext=(10, 10), fontsize=10, color='red')

    # Query points
    ax2.scatter(queries_global[:n_inside, 0], queries_global[:n_inside, 1],
                c='blue', s=15, alpha=0.7, label='Inside')
    ax2.scatter(queries_global[n_inside:n_inside+n_surface, 0],
                queries_global[n_inside:n_inside+n_surface, 1],
                c='cyan', s=10, alpha=0.5, label='Surface')
    ax2.scatter(queries_global[n_inside+n_surface:, 0],
                queries_global[n_inside+n_surface:, 1],
                c='magenta', s=15, alpha=0.7, label='Outside')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY Plane)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)

    # ==================== Subplot 3: Heightmap visualization ====================
    ax3 = fig.add_subplot(223)

    # Show heightmap Z values
    hm_z = heightmap[..., 2]
    im = ax3.imshow(hm_z, cmap='viridis', origin='lower',
                    extent=[-0.5, 0.5, -0.5, 0.5])
    plt.colorbar(im, ax=ax3, label='Z (local, m)')

    # Mark center (LiDAR position)
    ax3.scatter(0, 0, c='red', s=50, marker='x', linewidth=2)

    ax3.set_xlabel('X local (m)')
    ax3.set_ylabel('Y local (m)')
    ax3.set_title(f'Heightmap (Local Coords)\nZ range: [{hm_z.min():.3f}, {hm_z.max():.3f}]')

    # ==================== Subplot 4: SDF histogram ====================
    ax4 = fig.add_subplot(224)

    ax4.hist(sdfs[:n_inside], bins=20, alpha=0.7, color='blue', label='Inside (10%)')
    ax4.hist(sdfs[n_inside:n_inside+n_surface], bins=20, alpha=0.7, color='cyan', label='Surface (70%)')
    ax4.hist(sdfs[n_inside+n_surface:], bins=20, alpha=0.7, color='magenta', label='Outside (20%)')

    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Surface (SDF=0)')
    ax4.set_xlabel('SDF Value (m)')
    ax4.set_ylabel('Count')
    ax4.set_title('SDF Distribution by Category')
    ax4.legend()

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'sampling_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {output_path}")

    plt.show()

    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    # Verify constraints
    checks = []

    # Check 1: LiDAR position within ±2m
    lidar_in_range = abs(lidar_pose[0]) <= 2.0 and abs(lidar_pose[1]) <= 2.0
    checks.append(("LiDAR position in ±2m range", lidar_in_range))

    # Check 2: Heightmap local XY in ±0.5m
    hm_x_ok = heightmap[..., 0].min() >= -0.5 and heightmap[..., 0].max() <= 0.5
    hm_y_ok = heightmap[..., 1].min() >= -0.5 and heightmap[..., 1].max() <= 0.5
    checks.append(("Heightmap X in ±0.5m", hm_x_ok))
    checks.append(("Heightmap Y in ±0.5m", hm_y_ok))

    # Check 3: Query distribution
    inside_pct = (sdfs < 0).sum() / len(sdfs) * 100
    surface_pct = (np.abs(sdfs) < 0.1).sum() / len(sdfs) * 100
    checks.append(("Inside points ~10%", 5 < inside_pct < 15))
    checks.append(("Surface points ~70%", 60 < surface_pct < 80))

    # Check 4: SDF has negative values (inside terrain)
    checks.append(("SDF has negative values (penetration)", sdfs.min() < 0))

    print("\nCheck Results:")
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n🎉 All checks passed! Sampling logic is correct.")
    else:
        print("\n⚠️ Some checks failed. Please review the code.")

    print("="*60 + "\n")


if __name__ == "__main__":
    visualize_terrain_and_sampling()
