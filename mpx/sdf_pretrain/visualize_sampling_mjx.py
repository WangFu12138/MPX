#!/usr/bin/env python3
"""
Visualize SDF terrain sampling in MuJoCo viewer.

Verifies:
1. Terrain generation (WFC boxes)
2. LiDAR position sampling (±2m range)
3. Heightmap scan range (±0.5m around LiDAR)
4. Yaw-only rotation (gravity-aligned local frame)
5. Query point distribution (10% inside, 70% surface, 20% outside)
6. Shadow Base Frame: heightmap Z is relative to ground (0=ground, not LiDAR)

Controls:
- Left click + drag: Rotate camera
- Right click + drag: Pan
- Scroll: Zoom
- Press 'Q' to regenerate scene (new terrain + new LiDAR pose)
- Close window to quit

Usage:
    python -m mpx.sdf_pretrain.visualize_sampling_mjx
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator
from mpx.sdf_pretrain.data.dynamic_terrain import TerrainConfig
from mpx.sdf_pretrain.data.transforms import get_camera_transforms, local_to_global_points_numpy


def main():
    """Main visualization function."""

    print("\n" + "="*60)
    print("SDF Terrain Sampling Visualization (MuJoCo)")
    print("="*60)
    print("\nParameters:")
    print("  - LiDAR position range: ±2.0m")
    print("  - Heightmap scan range: ±0.5m")
    print("  - Rotation: Yaw-only (gravity-aligned)")
    print("  - Query distribution: 10%/70%/20% (inside/surface/outside)")
    print("\nControls:")
    print("  Q - Regenerate terrain and sample new LiDAR pose")
    print("  Close window - Quit")
    print("="*60 + "\n")

    # Initialize generator
    terrain_config = TerrainConfig(
        size=7,
        step_height_min=0.04,
        step_height_max=0.12,
        max_boxes=150,
    )

    generator = SDFDynamicGenerator(
        max_boxes=150,
        heightmap_size=(21, 21),
        num_queries_per_sample=128,
        lidar_height_range=(0.4, 0.8),
        view_bounds=(-0.5, 0.5, -0.5, 0.5, 0.01, 0.5),
        terrain_config=terrain_config,
    )

    # Generate initial terrain
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    print("Generating terrain...")
    generator.regenerate_terrain(subkey)

    # Generate initial batch
    key, subkey = jax.random.split(key)
    print("Generating batch...\n")
    batch = generator.generate_batch(subkey, batch_size=1, verbose=True)

    # Extract data
    lidar_pose = np.array(batch['lidar_pose'][0])
    queries_global = np.array(batch['queries_global'][0])
    sdfs = np.array(batch['sdf'][0])
    heightmap_raw = np.array(batch['heightmap'][0])
    hm_mean = np.array(batch['heightmap_mean'][0])
    hm_std = np.array(batch['heightmap_std'][0])
    heightmap = heightmap_raw * hm_std + hm_mean

    # Get MuJoCo model and data
    model = generator.mj_model
    data = generator.mj_data

    print("\n" + "="*60)
    print("Visualization Legend")
    print("="*60)
    print("  🔴 RED sphere: LiDAR sensor position")
    print("  🟢 GREEN spheres: Heightmap points")
    print("     - XY: relative to LiDAR center (±0.5m)")
    print("     - Z: relative to GROUND height (0=ground, +0.1=10cm step)")
    print("     - Uses Shadow Base Frame (Z-axis at world Z=0)")
    print("  🔵 BLUE spheres: Inside query points (SDF < 0)")
    print("  🔷 CYAN spheres: Surface query points (SDF ≈ 0)")
    print("  🟣 MAGENTA spheres: Outside query points (SDF > 0)")
    print("  🟡 YELLOW box: Heightmap coverage area (±0.5m)")
    print("  🟠 ORANGE box: LiDAR position sampling range (±2.0m)")
    print("="*60 + "\n")

    # State for regeneration
    regenerate = False

    def key_callback(keycode):
        """Handle keyboard input."""
        nonlocal regenerate
        # Q key
        if keycode == ord('Q') or keycode == ord('q'):
            regenerate = True
            print("\n[Q pressed] Regenerating scene...")

    def add_sphere(viewer, pos, color, size=0.02):
        """Add a sphere to the scene."""
        if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
            return False
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color
        )
        viewer.user_scn.ngeom += 1
        return True

    def add_box_edges(viewer, center, size, color):
        """Add box edges using cylinders."""
        x, y, z = center
        dx, dy, dz = size

        # 8 vertices
        vertices = np.array([
            [x-dx, y-dy, z], [x+dx, y-dy, z],
            [x+dx, y+dy, z], [x-dx, y+dy, z],
            [x-dx, y-dy, z+dz*2], [x+dx, y-dy, z+dz*2],
            [x+dx, y+dy, z+dz*2], [x-dx, y+dy, z+dz*2],
        ])

        # 12 edges
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # bottom
            [4,5], [5,6], [6,7], [7,4],  # top
            [0,4], [1,5], [2,6], [3,7],  # vertical
        ]

        for edge in edges:
            if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
                break
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 0.01:
                continue

            direction_normalized = direction / length
            mid_point = (start + end) / 2

            # Rotation matrix to align cylinder
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                R = np.eye(3) if np.dot(z_axis, direction_normalized) > 0 else np.diag([1, 1, -1])

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.01, length/2, 0],
                pos=mid_point,
                mat=R.flatten(),
                rgba=color
            )
            viewer.user_scn.ngeom += 1

    # Stats
    frame_count = 0
    last_print_time = time.time()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set camera
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 45
        viewer.cam.lookat[:] = [0, 0, 0.5]

        while viewer.is_running():
            # Check for regeneration request
            if regenerate:
                regenerate = False

                # Regenerate terrain
                key, subkey = jax.random.split(key)
                generator.regenerate_terrain(subkey)

                # Regenerate batch
                key, subkey = jax.random.split(key)
                batch = generator.generate_batch(subkey, batch_size=1, verbose=False)

                # Update data
                lidar_pose = np.array(batch['lidar_pose'][0])
                queries_global = np.array(batch['queries_global'][0])
                sdfs = np.array(batch['sdf'][0])
                heightmap_raw = np.array(batch['heightmap'][0])
                hm_mean = np.array(batch['heightmap_mean'][0])
                hm_std = np.array(batch['heightmap_std'][0])
                heightmap = heightmap_raw * hm_std + hm_mean

                # Update model reference (terrain boxes changed)
                model = generator.mj_model
                data = generator.mj_data

                # Heightmap Z stats (ground-relative)
                hm_z = heightmap[..., 2]
                hm_z_min = float(hm_z.min())
                hm_z_max = float(hm_z.max())
                hm_z_mean = float(hm_z.mean())

                print(f"\n✓ New LiDAR: ({lidar_pose[0]:+.2f}, {lidar_pose[1]:+.2f}, {lidar_pose[2]:.2f}) "
                      f"Yaw: {np.degrees(lidar_pose[5]):.1f}°")
                print(f"  Heightmap Z: [{hm_z_min:+.3f}, {hm_z_max:+.3f}] (relative to ground, 0=ground)")
                print(f"  Mean Z: {hm_z_mean:+.4f} | "
                      f"SDF: [{sdfs.min():.3f}, {sdfs.max():.3f}] Neg: {(sdfs < 0).sum()}")

            # Clear scene
            viewer.user_scn.ngeom = 0

            # ===== 1. LiDAR position range indicator (±2m, orange) =====
            add_box_edges(viewer, [0, 0, 0], [2.0, 2.0, 0.01], [1.0, 0.5, 0.0, 0.5])

            # ===== 2. Heightmap coverage (±0.5m around LiDAR, yellow) =====
            add_box_edges(viewer, lidar_pose[:3], [0.5, 0.5, 0.01], [1.0, 1.0, 0.0, 0.6])

            # ===== 3. LiDAR position (red sphere) =====
            add_sphere(viewer, lidar_pose[:3], [1.0, 0.0, 0.0, 1.0], size=0.06)

            # ===== 4. Heightmap points in world coords (green) =====
            # Use Shadow Base Frame: XY and Yaw follow LiDAR, Z=0 (world sea level)
            base_pose = np.array([
                lidar_pose[0], lidar_pose[1], 0.0,  # Z=0 is the key!
                0.0, 0.0, lidar_pose[5]  # Yaw-only (gravity-aligned)
            ])
            T_l2w, _ = get_camera_transforms(base_pose)
            hm_flat = heightmap.reshape(-1, 3)
            hm_world = local_to_global_points_numpy(hm_flat, T_l2w)

            for pt in hm_world[::4]:
                if not add_sphere(viewer, pt, [0.0, 1.0, 0.0, 0.6], size=0.015):
                    break

            # ===== 5. Query points =====
            n_inside = int(len(queries_global) * 0.10)
            n_surface = int(len(queries_global) * 0.70)

            # Inside (blue)
            for pt in queries_global[:n_inside][::2]:
                if not add_sphere(viewer, pt, [0.0, 0.0, 1.0, 0.8], size=0.02):
                    break

            # Surface (cyan)
            for pt in queries_global[n_inside:n_inside+n_surface][::3]:
                if not add_sphere(viewer, pt, [0.0, 1.0, 1.0, 0.6], size=0.015):
                    break

            # Outside (magenta)
            for pt in queries_global[n_inside+n_surface:][::2]:
                if not add_sphere(viewer, pt, [1.0, 0.0, 1.0, 0.8], size=0.02):
                    break

            viewer.sync()
            time.sleep(0.01)
            frame_count += 1

            # Print stats every 2 seconds
            current_time = time.time()
            if current_time - last_print_time > 2.0:
                # Heightmap Z stats (ground-relative)
                hm_z = heightmap[..., 2]
                hm_z_min = float(hm_z.min())
                hm_z_max = float(hm_z.max())
                hm_z_mean = float(hm_z.mean())

                print(f"\r[Frame {frame_count}] "
                      f"LiDAR: ({lidar_pose[0]:+.2f}, {lidar_pose[1]:+.2f}, {lidar_pose[2]:.2f}) "
                      f"Yaw: {np.degrees(lidar_pose[5]):.1f}° "
                      f"| HM_Z: [{hm_z_min:+.3f}, {hm_z_max:+.3f}] μ={hm_z_mean:+.3f} "
                      f"| SDF: [{sdfs.min():.3f}, {sdfs.max():.3f}] "
                      f"Neg: {(sdfs < 0).sum()}", end='', flush=True)
                last_print_time = current_time

    print("\n" + "="*60)
    print("Visualization ended")
    print("="*60)


if __name__ == "__main__":
    main()
