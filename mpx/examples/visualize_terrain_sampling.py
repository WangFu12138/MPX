#!/usr/bin/env python3
"""
Visualize SDF terrain sampling strategy.
This script sets up the SDF dataset generator and visualizes:
1. The simulated lidar pose and frustum
2. The heightmap intersection points (green spheres)
3. The uniformly sampled query points (blue)
4. The near-surface sampled query points (cyan)
5. The edge-sampled query points (magenta)
"""

import os
import sys
import jax
import numpy as np
import mujoco
import mujoco.viewer
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFOnlineGenerator

def main():
    print("Loading SDF terrain generator...")
    xml_path = os.path.join(dir_path, '../data/r2-1024/mjcf/scene_terrain_test.xml')
    generator = SDFOnlineGenerator(
        xml_path=xml_path,
        heightmap_size=(21, 21),
        num_queries_per_sample=1024
    )
    
    # Randomly arrange blocks just like in training
    rng = jax.random.PRNGKey(42)
    rng, subkey = jax.random.split(rng)
    generator.randomize_terrain(subkey)
    
    print("Generating a single batch to extract sampling points...")
    rng, subkey = jax.random.split(rng)
    batch = generator.generate_batch(subkey, batch_size=1)
    
    lidar_pose = np.array(batch['lidar_pose'][0])  # (6,)
    hm = np.array(batch['heightmap'][0])           # (21, 21, 3)
    queries = np.array(batch['queries_global'][0]) # (1024, 3)
    
    print(f"Lidar Pose: {lidar_pose[:3]}")
    print("Starting visualizer...")
    
    model = generator.mj_model
    data = generator.mj_data
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera to get a good overview
            viewer.cam.distance = 5.0
            viewer.cam.elevation = -45
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = lidar_pose[:3]
            
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
                for pt in hm_flat:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 1.0, 0.0, 0.6] # Green
                    )
                    viewer.user_scn.ngeom += 1
                
                # 3. Visualize sampled query points
                n_uniform = int(1024 * 0.2)
                n_surface = int(1024 * 0.5)
                n_edge = 1024 - n_uniform - n_surface
                
                # Uniform (Blue)
                for pt in queries[:n_uniform]:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 0.0, 1.0, 0.8] # Blue
                    )
                    viewer.user_scn.ngeom += 1
                    
                # Near Surface (Cyan)
                for pt in queries[n_uniform:n_uniform+n_surface]:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 1.0, 1.0, 1.0] # Cyan
                    )
                    viewer.user_scn.ngeom += 1
                    
                # Edge (Magenta)
                for pt in queries[n_uniform+n_surface:]:
                    if viewer.user_scn.ngeom >= max_geoms: break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=pt,
                        mat=np.eye(3).flatten(),
                        rgba=[1.0, 0.0, 1.0, 1.0] # Magenta
                    )
                    viewer.user_scn.ngeom += 1

                viewer.sync()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nVisualizer stopped.")

if __name__ == "__main__":
    main()
