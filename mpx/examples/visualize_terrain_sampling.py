#!/usr/bin/env python3
"""
Visualize SDF terrain sampling strategy with dynamic terrain generation.
This script visualizes:
1. The dynamically generated terrain (WFC-based boxes)
2. The simulated lidar pose and frustum
3. The heightmap intersection points (green spheres)
4. The uniformly sampled query points (blue)
5. The near-surface sampled query points (cyan)
6. The edge-sampled query points (magenta)
7. Surface rays showing xy->z mapping (yellow lines)
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
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator

def main():
    print("Loading SDF dynamic terrain generator...")
    generator = SDFDynamicGenerator(
        max_boxes=100,
        heightmap_size=(21, 21),
        num_queries_per_sample=512,  # Reduced for better visualization
        lidar_height_range=(0.5, 0.8),
    )

    # Generate dynamic terrain using WFC algorithm
    rng = jax.random.PRNGKey(42)
    rng, subkey = jax.random.split(rng)
    print("Generating terrain...")
    generator.regenerate_terrain(subkey)

    print("Generating a single batch to extract sampling points...")
    rng, subkey = jax.random.split(rng)
    batch = generator.generate_batch(subkey, batch_size=1)

    lidar_pose = np.array(batch['lidar_pose'][0])  # (6,)
    hm = np.array(batch['heightmap'][0])           # (21, 21, 3)
    queries = np.array(batch['queries_global'][0]) # (512, 3)
    sdfs = np.array(batch['sdf'][0])               # (512,)

    print(f"Lidar Pose: {lidar_pose}")
    print(f"  Position: ({lidar_pose[0]:.2f}, {lidar_pose[1]:.2f}, {lidar_pose[2]:.2f})")
    print(f"  Orientation: (roll={lidar_pose[3]:.2f}, pitch={lidar_pose[4]:.2f}, yaw={lidar_pose[5]:.2f})")
    print(f"Query points: {queries.shape[0]}")
    print(f"SDF range: [{sdfs.min():.4f}, {sdfs.max():.4f}]")
    print(f"Negative SDF count: {(sdfs < 0).sum()}")
    print("Starting visualizer...")

    model = generator.mj_model
    data = generator.mj_data

    # Sample some surface rays for visualization (showing xy->z mapping)
    n_rays = 20
    ray_xy = np.random.uniform(-0.5, 0.5, (n_rays, 2))
    ray_xy_global = ray_xy + lidar_pose[:2]
    surface_z = generator.batch_ray_cast_to_surface(jnp.array(ray_xy_global))
    ray_origins = np.concatenate([ray_xy_global, np.full((n_rays, 1), 5.0)], axis=-1)
    ray_ends = np.concatenate([ray_xy_global, np.array(surface_z)[:, None]], axis=-1)

    print(f"\nVisualization Legend:")
    print(f"  🔴 Red sphere: LiDAR sensor position")
    print(f"  🟢 Green spheres: Heightmap intersection points")
    print(f"  🔵 Blue spheres: Uniform sampled query points")
    print(f"  🔷 Cyan spheres: Near-surface sampled query points")
    print(f"  🟣 Magenta spheres: Edge sampled query points")
    print(f"  🟡 Yellow lines: Surface rays (xy->z mapping)")
    print(f"\nPress Ctrl+C to exit\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to get a good overview
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 45
            viewer.cam.lookat[:] = lidar_pose[:3]

            step = 0
            while viewer.is_running():
                viewer.user_scn.ngeom = 0
                max_geoms = len(viewer.user_scn.geoms)

                # 1. Visualize lidar origin (Red sphere)
                if viewer.user_scn.ngeom < max_geoms:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.05, 0, 0],
                        pos=lidar_pose[:3],
                        mat=np.eye(3).flatten(),
                        rgba=[1.0, 0.0, 0.0, 1.0] # Red
                    )
                    viewer.user_scn.ngeom += 1

                # 2. Visualize Heightmap intersections (Green spheres)
                hm_flat = hm.reshape(-1, 3)
                for pt in hm_flat[::4]:  # Skip some points for clarity
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 1.0, 0.0, 0.5] # Green, semi-transparent
                    )
                    viewer.user_scn.ngeom += 1

                # 3. Visualize surface rays (Yellow lines)
                for i in range(n_rays):
                    if viewer.user_scn.ngeom >= max_geoms: break

                    # Draw line from ray origin to surface
                    start = ray_origins[i]
                    end = ray_ends[i]

                    # MuJoCo lines need special handling - use small cylinders instead
                    direction = end - start
                    length = np.linalg.norm(direction)
                    if length > 0.01:
                        direction_normalized = direction / length
                        mid_point = (start + end) / 2

                        # Create rotation matrix to align cylinder with ray direction
                        # Default cylinder is along z-axis
                        z_axis = np.array([0, 0, 1])
                        rotation_axis = np.cross(z_axis, direction_normalized)
                        rotation_axis_norm = np.linalg.norm(rotation_axis)

                        if rotation_axis_norm > 1e-6:
                            rotation_axis = rotation_axis / rotation_axis_norm
                            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))

                            # Rodrigues' rotation formula
                            K = np.array([
                                [0, -rotation_axis[2], rotation_axis[1]],
                                [rotation_axis[2], 0, -rotation_axis[0]],
                                [-rotation_axis[1], rotation_axis[0], 0]
                            ])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                        else:
                            # Parallel or anti-parallel
                            if np.dot(z_axis, direction_normalized) > 0:
                                R = np.eye(3)
                            else:
                                R = np.diag([1, 1, -1])

                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[0.005, length/2, 0],  # radius, half-length, unused
                            pos=mid_point,
                            mat=R.flatten(),
                            rgba=[1.0, 1.0, 0.0, 0.4] # Yellow, semi-transparent
                        )
                        viewer.user_scn.ngeom += 1

                # 4. Visualize sampled query points
                n_uniform = int(512 * 0.2)
                n_surface = int(512 * 0.5)
                n_edge = 512 - n_uniform - n_surface

                # Uniform (Blue)
                for pt in queries[:n_uniform][::2]:  # Skip some for clarity
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.012, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 0.0, 1.0, 0.7] # Blue
                    )
                    viewer.user_scn.ngeom += 1

                # Near Surface (Cyan)
                for pt in queries[n_uniform:n_uniform+n_surface][::3]:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.012, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 1.0, 1.0, 0.7] # Cyan
                    )
                    viewer.user_scn.ngeom += 1

                # Edge (Magenta)
                for pt in queries[n_uniform+n_surface:][::2]:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.012, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[1.0, 0.0, 1.0, 0.7] # Magenta
                    )
                    viewer.user_scn.ngeom += 1

                viewer.sync()
                time.sleep(0.01)
                step += 1

                # Print stats every 100 steps
                if step % 500 == 0:
                    print(f"Step {step}: {viewer.user_scn.ngeom} geoms rendered")

    except KeyboardInterrupt:
        print("\n\nVisualizer stopped by user.")
        print(f"Final SDF stats: min={sdfs.min():.4f}, max={sdfs.max():.4f}, negative_count={(sdfs < 0).sum()}")

if __name__ == "__main__":
    main()

