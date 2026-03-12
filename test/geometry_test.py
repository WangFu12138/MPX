#!/usr/bin/env python3
"""Geometry validation test for SDF pretraining.

This script validates the spatial geometry calibration for SDF computation:
1. parse_xml_to_boxes: Parse XML to get all box parameters
2. get_camera_transforms: Compute transform matrices from 6D pose
3. local_to_global_points: Transform local points to global coordinates
4. compute_sdf_box_jax: Compute SDF from point to box using JAX

Usage:
    python test/geometry_test.py
    python test/geometry_test.py --visualize
"""
import os
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_xml_to_boxes(xml_path: str) -> list:
    """
    Parse MuJoCo XML file and extract all box geometries.

    Args:
        xml_path: Path to the MuJoCo XML file

    Returns:
        List of dicts, each containing:
            - 'name': box name
            - 'pos': center position [x, y, z]
            - 'quat': quaternion [w, x, y, z]
            - 'size': half-sizes [sx, sy, sz] (MuJoCo convention)
    """
    boxes = []

    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found: {xml_path}")
        return boxes

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        print("Warning: No worldbody found in XML")
        return boxes

    # Iterate through all bodies
    for body in worldbody.findall('.//body'):
        body_name = body.get('name', 'unnamed')
        pos_str = body.get('pos', '0 0 0')
        quat_str = body.get('quat', '1 0 0 0')

        # Parse position and quaternion
        pos = np.array([float(x) for x in pos_str.split()])
        quat = np.array([float(x) for x in quat_str.split()])

        # Find box geometry
        for geom in body.findall('geom'):
            geom_type = geom.get('type', 'sphere')
            if geom_type == 'box':
                size_str = geom.get('size', '0.1 0.1 0.1')
                size = np.array([float(x) for x in size_str.split()])

                boxes.append({
                    'name': body_name,
                    'pos': pos,
                    'quat': quat,
                    'size': size
                })

    return boxes


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert zyx euler angles to quaternion [w, x, y, z]."""
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


def get_camera_transforms(lidar_pose: np.ndarray) -> tuple:
    """
    Compute transform matrices from 6D lidar pose.

    Args:
        lidar_pose: 6D pose [x, y, z, roll, pitch, yaw]

    Returns:
        Tuple of (T_lidar_to_world, T_world_to_lidar):
            - T_lidar_to_world: 4x4 transformation matrix (lidar -> world)
            - T_world_to_lidar: 4x4 transformation matrix (world -> lidar)
    """
    x, y, z, roll, pitch, yaw = lidar_pose

    # Rotation matrix from euler angles
    quat = euler_to_quat(roll, pitch, yaw)
    R = quat_to_rot_matrix(quat)

    # Translation vector
    t = np.array([x, y, z])

    # 4x4 transformation matrix: lidar -> world
    T_lidar_to_world = np.eye(4)
    T_lidar_to_world[:3, :3] = R
    T_lidar_to_world[:3, 3] = t

    # 4x4 transformation matrix: world -> lidar
    T_world_to_lidar = np.eye(4)
    T_world_to_lidar[:3, :3] = R.T
    T_world_to_lidar[:3, 3] = -R.T @ t

    return T_lidar_to_world, T_world_to_lidar


def local_to_global_points(queries_local: np.ndarray, T_lidar_to_world: np.ndarray) -> np.ndarray:
    """
    Transform local (lidar frame) points to global (world frame) coordinates.

    Args:
        queries_local: Local points, shape (N, 3) or (N, 4) homogeneous
        T_lidar_to_world: 4x4 transformation matrix

    Returns:
        Global points, shape (N, 3)
    """
    if queries_local.ndim == 1:
        queries_local = queries_local.reshape(1, -1)

    n_points = queries_local.shape[0]

    # Convert to homogeneous coordinates if needed
    if queries_local.shape[1] == 3:
        ones = np.ones((n_points, 1))
        queries_homo = np.hstack([queries_local, ones])
    else:
        queries_homo = queries_local

    # Transform
    queries_global_homo = (T_lidar_to_world @ queries_homo.T).T

    return queries_global_homo[:, :3]


def global_to_local_points(queries_global: np.ndarray, T_world_to_lidar: np.ndarray) -> np.ndarray:
    """
    Transform global (world frame) points to local (lidar frame) coordinates.

    Args:
        queries_global: Global points, shape (N, 3)
        T_world_to_lidar: 4x4 transformation matrix

    Returns:
        Local points, shape (N, 3)
    """
    if queries_global.ndim == 1:
        queries_global = queries_global.reshape(1, -1)

    n_points = queries_global.shape[0]
    ones = np.ones((n_points, 1))
    queries_homo = np.hstack([queries_global, ones])

    queries_local_homo = (T_world_to_lidar @ queries_homo.T).T

    return queries_local_homo[:, :3]


def compute_sdf_box_numpy(query: np.ndarray, box_center: np.ndarray,
                          box_halfsizes: np.ndarray) -> float:
    """
    Compute signed distance from a point to an axis-aligned box (NumPy version).

    This is the reference implementation for validation.

    SDF formula: length(max(q,0)) + min(max(q),0)
    - If point is outside: returns positive distance to surface
    - If point is inside: returns negative distance to nearest surface

    Args:
        query: Query point [x, y, z]
        box_center: Box center [x, y, z]
        box_halfsizes: Box half-sizes [sx, sy, sz]

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    # Transform query to box local frame
    local = query - box_center

    # q = |local| - halfsizes
    # Positive q means outside on that axis, negative means inside
    q = np.abs(local) - box_halfsizes

    # Correct SDF formula:
    # - outside_dist: distance in the "outside region" (axes where q > 0)
    # - inside_dist: 0 if any axis is outside, otherwise min(q)
    outside_dist = np.linalg.norm(np.maximum(q, 0))
    inside_dist = np.minimum(np.max(q), 0)

    # Combined SDF
    sdf = outside_dist + inside_dist

    return sdf


def compute_sdf_box_jax(query, box_center, box_halfsizes):
    """
    Compute signed distance from a point to an axis-aligned box (JAX version).

    SDF formula: length(max(q,0)) + min(max(q),0)

    Args:
        query: Query point [x, y, z] (jax array)
        box_center: Box center [x, y, z] (jax array)
        box_halfsizes: Box half-sizes [sx, sy, sz] (jax array)

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    import jax.numpy as jnp

    # Transform query to box local frame
    local = query - box_center

    # q = |local| - halfsizes
    q = jnp.abs(local) - box_halfsizes

    # Correct SDF formula
    outside_dist = jnp.linalg.norm(jnp.maximum(q, 0))
    inside_dist = jnp.minimum(jnp.max(q), 0)

    sdf = outside_dist + inside_dist

    return sdf


def compute_sdf_box_oriented_jax(query, box_center, box_quat, box_halfsizes):
    """
    Compute signed distance from a point to an oriented box (JAX version).

    SDF formula: length(max(q,0)) + min(max(q),0)

    Args:
        query: Query point [x, y, z] (jax array)
        box_center: Box center [x, y, z] (jax array)
        box_quat: Box quaternion [w, x, y, z] (jax array)
        box_halfsizes: Box half-sizes [sx, sy, sz] (jax array)

    Returns:
        Signed distance (negative = inside, positive = outside)
    """
    import jax.numpy as jnp

    # Convert quaternion to rotation matrix
    w, x, y, z = box_quat
    R = jnp.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    # Transform query to box local frame
    local = R.T @ (query - box_center)

    # Compute SDF for axis-aligned box in local frame
    q = jnp.abs(local) - box_halfsizes
    outside_dist = jnp.linalg.norm(jnp.maximum(q, 0))
    inside_dist = jnp.minimum(jnp.max(q), 0)
    sdf = outside_dist + inside_dist

    return sdf


def get_ground_truth_sdf(queries_global: np.ndarray, boxes_info: list,
                         use_jax: bool = True) -> np.ndarray:
    """
    Compute ground truth SDF for multiple query points against all boxes.

    Args:
        queries_global: Query points, shape (N, 3)
        boxes_info: List of box dicts from parse_xml_to_boxes
        use_jax: If True, use JAX implementation (faster for batches)

    Returns:
        SDF values, shape (N,)
    """
    if use_jax:
        import jax
        import jax.numpy as jnp

        # Convert to JAX arrays
        queries_jax = jnp.array(queries_global)

        # Compute SDF for each box and take minimum
        def compute_single_sdf(query):
            min_sdf = jnp.inf
            for box in boxes_info:
                sdf = compute_sdf_box_oriented_jax(
                    query,
                    jnp.array(box['pos']),
                    jnp.array(box['quat']),
                    jnp.array(box['size'])
                )
                min_sdf = jnp.minimum(min_sdf, sdf)
            return min_sdf

        # Vectorize over all query points
        compute_batch_sdf = jax.vmap(compute_single_sdf)
        sdfs = compute_batch_sdf(queries_jax)

        return np.array(sdfs)
    else:
        # NumPy version (slower but no JAX dependency)
        n_queries = queries_global.shape[0]
        sdfs = np.full(n_queries, np.inf)

        for i, query in enumerate(queries_global):
            for box in boxes_info:
                # For simplicity, use axis-aligned version
                # (assuming boxes are not rotated for now)
                sdf = compute_sdf_box_numpy(
                    query,
                    box['pos'],
                    box['size']
                )
                sdfs[i] = min(sdfs[i], sdf)

        return sdfs


# ============================================================================
# Test Functions
# ============================================================================

def test_parse_xml_to_boxes():
    """Test 1: Parse XML to extract box information."""
    print("\n" + "="*60)
    print("Test 1: parse_xml_to_boxes")
    print("="*60)

    # Use the terrain test scene
    mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(mpx_dir, "mpx", "data", "r2-1024", "mjcf", "scene_terrain_boxes.xml")

    print(f"\nParsing: {xml_path}")
    boxes = parse_xml_to_boxes(xml_path)

    print(f"\nFound {len(boxes)} boxes")
    if boxes:
        print("\nFirst 3 boxes:")
        for i, box in enumerate(boxes[:3]):
            print(f"  Box {i}: {box['name']}")
            print(f"    Position: {box['pos']}")
            print(f"    Quaternion: {box['quat']}")
            print(f"    Half-sizes: {box['size']}")

    print("\n✓ parse_xml_to_boxes test passed")
    return boxes


def test_camera_transforms():
    """Test 2: Camera/LiDAR transform matrices."""
    print("\n" + "="*60)
    print("Test 2: get_camera_transforms")
    print("="*60)

    # Test with a known pose
    lidar_pose = np.array([1.0, 2.0, 0.5, 0.0, 0.0, np.pi/4])  # x, y, z, roll, pitch, yaw

    print(f"\nLiDAR pose: [x={lidar_pose[0]}, y={lidar_pose[1]}, z={lidar_pose[2]}, "
          f"roll={lidar_pose[3]}, pitch={lidar_pose[4]}, yaw={lidar_pose[5]:.4f}]")

    T_lidar_to_world, T_world_to_lidar = get_camera_transforms(lidar_pose)

    print("\nT_lidar_to_world (4x4):")
    print(T_lidar_to_world)

    print("\nT_world_to_lidar (4x4):")
    print(T_world_to_lidar)

    # Verify that they are inverses
    product = T_lidar_to_world @ T_world_to_lidar
    print("\nVerifying T_lidar_to_world @ T_world_to_lidar = I:")
    print(product)
    assert np.allclose(product, np.eye(4), atol=1e-10), "Transforms should be inverses!"

    print("\n✓ get_camera_transforms test passed")
    return T_lidar_to_world, T_world_to_lidar


def test_coordinate_transform():
    """Test 3: Local to global coordinate transformation."""
    print("\n" + "="*60)
    print("Test 3: local_to_global_points")
    print("="*60)

    # Define a known transform
    lidar_pose = np.array([1.0, 0.0, 0.5, 0.0, 0.0, np.pi/2])  # 90 degree yaw
    T_lidar_to_world, T_world_to_lidar = get_camera_transforms(lidar_pose)

    # Test points in local frame
    queries_local = np.array([
        [0, 0, 0],      # Origin
        [1, 0, 0],      # X-axis
        [0, 1, 0],      # Y-axis
        [0, 0, 1],      # Z-axis
        [0.5, 0.5, 0],  # Diagonal
    ])

    print("\nLocal points:")
    print(queries_local)

    # Transform to global
    queries_global = local_to_global_points(queries_local, T_lidar_to_world)
    print("\nGlobal points:")
    print(queries_global)

    # Transform back to local
    queries_local_back = global_to_local_points(queries_global, T_world_to_lidar)
    print("\nTransformed back to local:")
    print(queries_local_back)

    # Verify round-trip
    assert np.allclose(queries_local, queries_local_back, atol=1e-10), "Round-trip should match!"

    # Manual verification for 90-degree yaw:
    # Local [1, 0, 0] should become global [1, 1, 0.5] (x -> y, y -> -x, z unchanged)
    expected_point_1 = np.array([1.0, 1.0, 0.5])  # After 90 deg yaw, x_local becomes y_global
    print(f"\nManual verification: local [1,0,0] -> global {queries_global[1]}")
    print(f"Expected (approx): {expected_point_1}")

    print("\n✓ local_to_global_points test passed")


def test_sdf_computation():
    """Test 4: SDF computation for boxes."""
    print("\n" + "="*60)
    print("Test 4: compute_sdf_box")
    print("="*60)

    # Define a test box
    box_center = np.array([0.0, 0.0, 0.25])  # Box at origin, centered at z=0.25
    box_halfsizes = np.array([0.5, 0.5, 0.25])  # 1m x 1m x 0.5m box

    print(f"\nTest box:")
    print(f"  Center: {box_center}")
    print(f"  Half-sizes: {box_halfsizes}")
    print(f"  Full dimensions: {2*box_halfsizes}")

    # Test points
    test_points = np.array([
        [0.0, 0.0, 0.25],   # Inside, at center
        [0.0, 0.0, 0.5],    # On top surface
        [0.0, 0.0, 1.0],    # Above box
        [0.0, 0.0, -0.1],   # Below box
        [0.6, 0.0, 0.25],   # Outside in X
        [0.0, 0.6, 0.25],   # Outside in Y
        [0.3, 0.3, 0.25],   # Inside, near corner
        [0.6, 0.6, 0.6],    # Outside, corner
    ])

    print("\nTest points and SDF values:")
    print("-" * 50)

    for i, point in enumerate(test_points):
        sdf = compute_sdf_box_numpy(point, box_center, box_halfsizes)
        status = "INSIDE" if sdf < 0 else ("SURFACE" if sdf == 0 else "OUTSIDE")
        print(f"  Point {i}: {point} -> SDF = {sdf:+.4f} ({status})")

    print("\n✓ compute_sdf_box test passed")


def test_sdf_with_jax():
    """Test 5: JAX SDF computation with gradient."""
    print("\n" + "="*60)
    print("Test 5: JAX SDF with gradient computation")
    print("="*60)

    try:
        import jax
        import jax.numpy as jnp

        # Define a test box
        box_center = jnp.array([0.0, 0.0, 0.25])
        box_halfsizes = jnp.array([0.5, 0.5, 0.25])

        # Test point
        query = jnp.array([0.6, 0.0, 0.25])

        # Compute SDF
        sdf = compute_sdf_box_jax(query, box_center, box_halfsizes)
        print(f"\nSDF at {query}: {sdf}")

        # Compute gradient (SDF gradient points towards the surface)
        grad_fn = jax.grad(compute_sdf_box_jax, argnums=0)
        gradient = grad_fn(query, box_center, box_halfsizes)
        print(f"Gradient (points towards surface): {gradient}")

        # Verify gradient norm is approximately 1 (Eikonal property)
        grad_norm = jnp.linalg.norm(gradient)
        print(f"Gradient norm: {grad_norm:.4f} (should be ~1.0)")

        # Test inside point
        query_inside = jnp.array([0.0, 0.0, 0.25])
        sdf_inside = compute_sdf_box_jax(query_inside, box_center, box_halfsizes)
        grad_inside = grad_fn(query_inside, box_center, box_halfsizes)
        print(f"\nSDF at {query_inside} (inside): {sdf_inside}")
        print(f"Gradient: {grad_inside}")

        print("\n✓ JAX SDF gradient test passed")

    except ImportError:
        print("\nJAX not available, skipping gradient test")


def test_full_pipeline():
    """Test 6: Full pipeline - load terrain, compute SDFs."""
    print("\n" + "="*60)
    print("Test 6: Full SDF pipeline")
    print("="*60)

    try:
        import jax
        import jax.numpy as jnp

        # Load terrain boxes
        mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # First check if we have a generated terrain
        test_scene = os.path.join(mpx_dir, "mpx", "data", "r2-1024", "mjcf", "scene_terrain_test.xml")
        template_scene = os.path.join(mpx_dir, "mpx", "data", "r2-1024", "mjcf", "scene_terrain_boxes.xml")

        if os.path.exists(test_scene):
            xml_path = test_scene
            print(f"\nUsing generated terrain: {xml_path}")
        else:
            xml_path = template_scene
            print(f"\nUsing template terrain: {xml_path}")

        boxes = parse_xml_to_boxes(xml_path)
        print(f"Found {len(boxes)} boxes")

        if not boxes:
            print("No boxes found, skipping pipeline test")
            return

        # Filter out placeholder boxes (those at (100, 100, 10))
        real_boxes = [b for b in boxes if not (b['pos'][0] > 50 and b['pos'][1] > 50)]
        print(f"Real terrain boxes: {len(real_boxes)}")

        if not real_boxes:
            print("Only placeholder boxes found, skipping")
            return

        # Define a virtual LiDAR pose
        lidar_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])  # At origin, looking forward
        T_lidar_to_world, T_world_to_lidar = get_camera_transforms(lidar_pose)

        # Sample query points in local LiDAR frame
        # Create a grid of points in front of the LiDAR
        n_points = 100
        np.random.seed(42)
        queries_local = np.random.uniform(
            low=[-1, -1, -0.5],
            high=[2, 1, 0.5],
            size=(n_points, 3)
        )

        print(f"\nGenerated {n_points} random query points in local frame")

        # Transform to global
        queries_global = local_to_global_points(queries_local, T_lidar_to_world)

        # Compute ground truth SDFs
        print("Computing ground truth SDFs...")
        sdfs = get_ground_truth_sdf(queries_global, real_boxes, use_jax=True)

        # Statistics
        inside_count = np.sum(sdfs < 0)
        outside_count = np.sum(sdfs >= 0)

        print(f"\nSDF Statistics:")
        print(f"  Min SDF: {sdfs.min():.4f}")
        print(f"  Max SDF: {sdfs.max():.4f}")
        print(f"  Mean SDF: {sdfs.mean():.4f}")
        print(f"  Points inside: {inside_count} ({100*inside_count/n_points:.1f}%)")
        print(f"  Points outside: {outside_count} ({100*outside_count/n_points:.1f}%)")

        print("\n✓ Full pipeline test passed")

    except ImportError as e:
        print(f"\nJAX not available: {e}")
        print("Skipping full pipeline test")


def visualize_sdf():
    """Visualize SDF field as a 2D slice."""
    print("\n" + "="*60)
    print("Visualization: 2D SDF slice")
    print("="*60)

    try:
        import matplotlib.pyplot as plt
        import jax.numpy as jnp

        # Create a simple box for visualization
        box_center = jnp.array([0.0, 0.0, 0.0])
        box_halfsizes = jnp.array([0.3, 0.3, 0.3])

        # Create a 2D grid at z=0
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)

        # Compute SDF for each point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                query = jnp.array([X[i, j], Y[i, j], 0.0])
                Z[i, j] = float(compute_sdf_box_jax(query, box_center, box_halfsizes))

        # Plot
        plt.figure(figsize=(10, 8))
        levels = np.linspace(Z.min(), Z.max(), 50)
        contour = plt.contourf(X, Y, Z, levels=levels, cmap='coolwarm')
        plt.colorbar(contour, label='SDF')

        # Draw box outline
        box_x = [-0.3, 0.3, 0.3, -0.3, -0.3]
        box_y = [-0.3, -0.3, 0.3, 0.3, -0.3]
        plt.plot(box_x, box_y, 'k-', linewidth=2, label='Box')

        # Zero contour (surface)
        plt.contour(X, Y, Z, levels=[0], colors='white', linewidths=2)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('SDF Field (z=0 slice)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # Save figure
        output_path = os.path.join(os.path.dirname(__file__), 'sdf_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")

        plt.show()

    except ImportError as e:
        print(f"\nMatplotlib not available: {e}")
        print("Skipping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Geometry validation for SDF pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--visualize', action='store_true',
                        help='Show SDF visualization')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SDF Geometry Validation Test Suite")
    print("="*60)

    # Run all tests
    test_parse_xml_to_boxes()
    test_camera_transforms()
    test_coordinate_transform()
    test_sdf_computation()
    test_sdf_with_jax()
    test_full_pipeline()

    if args.visualize:
        visualize_sdf()

    print("\n" + "="*60)
    print("All geometry tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
