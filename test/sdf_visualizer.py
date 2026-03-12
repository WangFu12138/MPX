#!/usr/bin/env python3
"""Interactive SDF Visualizer for MuJoCo terrain.

This tool allows you to visualize terrain SDF values interactively:
- A floating sphere represents the "radar" query point
- Use keyboard or command input to move the probe
- Real-time SDF value display

Usage:
    python test/sdf_visualizer.py
    python test/sdf_visualizer.py --xml mpx/data/r2-1024/mjcf/scene_terrain_test.xml
    python test/sdf_visualizer.py --cli           # CLI mode for coordinate input
    python test/sdf_visualizer.py --point "0.5 0.3 0.2"  # Single query

Controls (in visualizer mode):
    W/S: Move probe forward/backward (Y axis)
    A/D: Move probe left/right (X axis)
    Q/E: Move probe up/down (Z axis)
    R: Reset probe to origin
    P: Print current position and SDF
    Space: Toggle probe visibility
"""
import os
import sys
import argparse
import tempfile
import numpy as np
import xml.etree.ElementTree as ET

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mujoco
import mujoco.viewer


# ============================================================================
# SDF Computation Functions (from geometry_test.py)
# ============================================================================

def parse_xml_to_boxes(xml_path: str) -> list:
    """
    Parse MuJoCo XML file and extract all box geometries.

    Args:
        xml_path: Path to the MuJoCo XML file

    Returns:
        List of dicts containing box info
    """
    boxes = []

    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found: {xml_path}")
        return boxes

    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find('worldbody')
    if worldbody is None:
        return boxes

    for body in worldbody.findall('.//body'):
        body_name = body.get('name', 'unnamed')
        pos_str = body.get('pos', '0 0 0')
        quat_str = body.get('quat', '1 0 0 0')

        pos = np.array([float(x) for x in pos_str.split()])
        quat = np.array([float(x) for x in quat_str.split()])

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


def compute_sdf_box_oriented_numpy(query: np.ndarray, box_center: np.ndarray,
                                    box_quat: np.ndarray, box_halfsizes: np.ndarray) -> float:
    """Compute signed distance from a point to an oriented box."""
    w, x, y, z = box_quat
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    local = R.T @ (query - box_center)
    q = np.abs(local) - box_halfsizes
    outside_dist = np.linalg.norm(np.maximum(q, 0))
    inside_dist = np.minimum(np.max(q), 0)
    return outside_dist + inside_dist


def compute_sdf_at_point(query: np.ndarray, boxes_info: list) -> tuple:
    """
    Compute SDF for a query point against all boxes.

    Returns:
        (sdf_value, closest_box_name)
    """
    min_sdf = np.inf
    closest_box = None

    for box in boxes_info:
        sdf = compute_sdf_box_oriented_numpy(
            query,
            box['pos'],
            box['quat'],
            box['size']
        )
        if sdf < min_sdf:
            min_sdf = sdf
            closest_box = box['name']

    return min_sdf, closest_box


def filter_terrain_boxes(boxes: list) -> list:
    """Filter out placeholder boxes (those at far positions)."""
    return [b for b in boxes if not (b['pos'][0] > 50 and b['pos'][1] > 50)]


# ============================================================================
# MuJoCo Model Creation (without robot, with probe)
# ============================================================================

def create_visualization_xml(original_xml_path: str, probe_pos: np.ndarray = None) -> str:
    """
    Create a modified XML that excludes the robot but includes a probe sphere.

    Returns:
        Path to the modified XML file
    """
    if probe_pos is None:
        probe_pos = np.array([0.0, 0.0, 0.5])

    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    # Remove include statements (robot.xml)
    for include in root.findall('include'):
        root.remove(include)

    # Remove keyframe that references robot joints
    for keyframe in root.findall('keyframe'):
        root.remove(keyframe)

    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')

    # Add probe body (basketball style)
    probe_body = ET.SubElement(worldbody, 'body')
    probe_body.set('name', 'sdf_probe')
    probe_body.set('pos', f"{probe_pos[0]} {probe_pos[1]} {probe_pos[2]}")

    # Add free joint so the probe can move
    free_joint = ET.SubElement(probe_body, 'joint')
    free_joint.set('name', 'probe_joint')
    free_joint.set('type', 'free')

    # Main basketball sphere (orange)
    probe_geom = ET.SubElement(probe_body, 'geom')
    probe_geom.set('name', 'probe_geom')
    probe_geom.set('type', 'sphere')
    probe_geom.set('size', '0.08')
    probe_geom.set('rgba', '1 0.45 0.0 1.0')  # Basketball orange
    probe_geom.set('contype', '0')
    probe_geom.set('conaffinity', '0')

    # Basketball seam lines using ellipsoids (black)
    # Horizontal seam (equator)
    seam_h = ET.SubElement(probe_body, 'geom')
    seam_h.set('name', 'seam_horizontal')
    seam_h.set('type', 'ellipsoid')
    seam_h.set('size', '0.081 0.081 0.003')  # Slightly larger radius, very thin
    seam_h.set('rgba', '0.1 0.1 0.1 1.0')  # Black
    seam_h.set('contype', '0')
    seam_h.set('conaffinity', '0')

    # Vertical seam 1 (Y-axis)
    seam_v1 = ET.SubElement(probe_body, 'geom')
    seam_v1.set('name', 'seam_vertical_1')
    seam_v1.set('type', 'ellipsoid')
    seam_v1.set('size', '0.081 0.003 0.081')
    seam_v1.set('rgba', '0.1 0.1 0.1 1.0')
    seam_v1.set('contype', '0')
    seam_v1.set('conaffinity', '0')

    # Vertical seam 2 (X-axis)
    seam_v2 = ET.SubElement(probe_body, 'geom')
    seam_v2.set('name', 'seam_vertical_2')
    seam_v2.set('type', 'ellipsoid')
    seam_v2.set('size', '0.003 0.081 0.081')
    seam_v2.set('rgba', '0.1 0.1 0.1 1.0')
    seam_v2.set('contype', '0')
    seam_v2.set('conaffinity', '0')

    # Add a glowing site at center for visibility
    probe_site = ET.SubElement(probe_body, 'site')
    probe_site.set('name', 'probe_site')
    probe_site.set('type', 'sphere')
    probe_site.set('size', '0.02')
    probe_site.set('rgba', '1 1 0 0.8')  # Yellow glow

    # Save to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.xml', prefix='sdf_viz_')
    os.close(temp_fd)

    tree.write(temp_path)
    return temp_path


# ============================================================================
# Interactive Visualizer
# ============================================================================

class SDFVisualizer:
    """Interactive SDF visualizer with MuJoCo."""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.temp_xml_path = None

        # Parse boxes for SDF computation
        print(f"Loading terrain from: {xml_path}")
        all_boxes = parse_xml_to_boxes(xml_path)
        self.boxes = filter_terrain_boxes(all_boxes)
        print(f"Found {len(self.boxes)} terrain boxes")

        # Probe state
        self.probe_pos = np.array([0.0, 0.0, 0.5])
        self.step_size = 0.05

        # Create model with probe
        self.temp_xml_path = create_visualization_xml(xml_path, self.probe_pos)
        self.model = mujoco.MjModel.from_xml_path(self.temp_xml_path)
        self.data = mujoco.MjData(self.model)

        # Find probe body and joint ids
        self.probe_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sdf_probe")
        self.probe_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
        self.probe_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "probe_joint")

        print(f"Probe body ID: {self.probe_body_id}, joint ID: {self.probe_joint_id}")

        # Get qpos address for the free joint
        if self.probe_joint_id >= 0:
            self.probe_qpos_adr = self.model.jnt_qposadr[self.probe_joint_id]
        else:
            self.probe_qpos_adr = 0

        # Print instructions
        self._print_instructions()

    def _print_instructions(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("SDF Visualizer Controls (Keyboard)")
        print("="*60)
        print("  W/S: Move probe forward/backward (Y axis)")
        print("  A/D: Move probe left/right (X axis)")
        print("  Q/E: Move probe up/down (Z axis)")
        print("  R:   Reset probe to origin")
        print("  P:   Print current position and SDF")
        print("  Space: Toggle probe visibility")
        print("  +/-: Increase/decrease step size")
        print("  Esc: Exit viewer")
        print("="*60 + "\n")

    def update_probe_position(self):
        """Update probe position in MuJoCo data via qpos."""
        if self.probe_joint_id >= 0:
            # Free joint qpos: [x, y, z, qw, qx, qy, qz]
            self.data.qpos[self.probe_qpos_adr + 0] = self.probe_pos[0]  # x
            self.data.qpos[self.probe_qpos_adr + 1] = self.probe_pos[1]  # y
            self.data.qpos[self.probe_qpos_adr + 2] = self.probe_pos[2]  # z
            self.data.qpos[self.probe_qpos_adr + 3] = 1.0  # qw (identity quaternion)
            self.data.qpos[self.probe_qpos_adr + 4] = 0.0  # qx
            self.data.qpos[self.probe_qpos_adr + 5] = 0.0  # qy
            self.data.qpos[self.probe_qpos_adr + 6] = 0.0  # qz
            # Forward kinematics to update xpos
            mujoco.mj_forward(self.model, self.data)

    def get_current_sdf(self) -> tuple:
        """Get SDF value at current probe position."""
        return compute_sdf_at_point(self.probe_pos, self.boxes)

    def query_point(self, x: float, y: float, z: float) -> tuple:
        """
        Query SDF at a specific point.

        Args:
            x, y, z: World coordinates

        Returns:
            (sdf_value, status, closest_box_name)
        """
        query = np.array([x, y, z])
        sdf_val, closest_box = compute_sdf_at_point(query, self.boxes)
        status = "INSIDE" if sdf_val < 0 else ("SURFACE" if abs(sdf_val) < 0.001 else "OUTSIDE")
        return sdf_val, status, closest_box

    def run_interactive(self):
        """Run the interactive visualizer."""
        print("\nStarting interactive viewer...")
        print("Click on the viewer window and use keyboard controls.\n")

        # Key callback state
        keys_pressed = set()

        def key_callback(keycode):
            keys_pressed.add(keycode)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            # Set initial camera
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 45
            viewer.cam.lookat[:] = [0, 0, 0.5]

            step = 0
            last_print_step = 0

            while viewer.is_running():
                # Handle keyboard input
                # Note: MuJoCo passive viewer doesn't directly support key queries
                # We'll use the sync callback approach

                # Update probe position in simulation
                mujoco.mj_forward(self.model, self.data)
                if self.probe_body_id >= 0:
                    self.data.xpos[self.probe_body_id] = self.probe_pos

                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Print SDF info periodically
                if step - last_print_step > 30:
                    sdf_val, closest_box = self.get_current_sdf()
                    status = "INSIDE" if sdf_val < 0 else "OUTSIDE"
                    print(f"\rProbe: [{self.probe_pos[0]:6.2f}, {self.probe_pos[1]:6.2f}, {self.probe_pos[2]:6.2f}] "
                          f"SDF: {sdf_val:+8.4f} ({status}) closest: {closest_box[:15]:15s}", end="")
                    last_print_step = step

                viewer.sync()
                step += 1

        print()  # New line after the status line

    def run_with_keyboard_thread(self):
        """Run visualizer with a separate keyboard input thread."""
        import threading
        import time

        print("\nStarting visualizer with keyboard control...")
        print("Use the terminal window for keyboard input (W/A/S/D/Q/E to move)\n")

        stop_event = threading.Event()

        def keyboard_input_thread():
            """Thread for keyboard input."""
            while not stop_event.is_set():
                try:
                    # Non-blocking input using select on Unix
                    import select

                    # Check if input is available
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()

                        if key == 'w':
                            self.probe_pos[1] += self.step_size
                        elif key == 's':
                            self.probe_pos[1] -= self.step_size
                        elif key == 'a':
                            self.probe_pos[0] -= self.step_size
                        elif key == 'd':
                            self.probe_pos[0] += self.step_size
                        elif key == 'q':
                            self.probe_pos[2] += self.step_size
                        elif key == 'e':
                            self.probe_pos[2] -= self.step_size
                        elif key == 'r':
                            self.probe_pos = np.array([0.0, 0.0, 0.5])
                        elif key == 'p':
                            sdf_val, closest_box = self.get_current_sdf()
                            status = "INSIDE" if sdf_val < 0 else "OUTSIDE"
                            print(f"\n>>> Position: {self.probe_pos}")
                            print(f">>> SDF: {sdf_val:.6f} ({status})")
                            print(f">>> Closest box: {closest_box}\n")
                        elif key in ['+', '=']:
                            self.step_size = min(0.5, self.step_size + 0.01)
                            print(f"\nStep size: {self.step_size:.2f}")
                        elif key == '-':
                            self.step_size = max(0.01, self.step_size - 0.01)
                            print(f"\nStep size: {self.step_size:.2f}")
                        elif key == '\x1b':  # ESC
                            stop_event.set()

                except Exception as e:
                    pass

        # Start keyboard thread
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            keyboard_thread = threading.Thread(target=keyboard_input_thread)
            keyboard_thread.daemon = True
            keyboard_thread.start()

            # Run viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = 8.0
                viewer.cam.elevation = -30
                viewer.cam.azimuth = 45
                viewer.cam.lookat[:] = [0, 0, 0.5]

                step = 0
                last_print_step = 0

                while viewer.is_running() and not stop_event.is_set():
                    # Update probe position via qpos
                    self.update_probe_position()

                    mujoco.mj_step(self.model, self.data)

                    if step - last_print_step > 30:
                        sdf_val, closest_box = self.get_current_sdf()
                        status = "INSIDE" if sdf_val < 0 else "OUTSIDE"
                        sys.stdout.write(f"\rProbe: [{self.probe_pos[0]:6.2f}, {self.probe_pos[1]:6.2f}, {self.probe_pos[2]:6.2f}] "
                                         f"SDF: {sdf_val:+8.4f} ({status})  ")
                        sys.stdout.flush()
                        last_print_step = step

                    viewer.sync()
                    step += 1

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            stop_event.set()

        print("\nViewer closed.")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_xml_path and os.path.exists(self.temp_xml_path):
            os.remove(self.temp_xml_path)


def run_cli_mode(visualizer: SDFVisualizer):
    """Run in command-line input mode."""
    print("\n" + "="*60)
    print("CLI Mode - Enter coordinates to query SDF")
    print("Type 'x y z' to query a point (e.g., '0.5 0.3 0.2')")
    print("Type 'q' to quit")
    print("Type 'v' to launch visualizer")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Enter xyz coordinates: ").strip()

            if user_input.lower() == 'q':
                print("Exiting...")
                break
            elif user_input.lower() == 'v':
                visualizer.run_with_keyboard_thread()
                continue

            # Parse coordinates
            coords = user_input.split()
            if len(coords) != 3:
                print("Please enter exactly 3 numbers (x y z)")
                continue

            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])

            # Query SDF
            sdf_val, status, closest_box = visualizer.query_point(x, y, z)

            print(f"\n  Query: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  SDF:   {sdf_val:+.6f}")
            print(f"  Status: {status}")
            print(f"  Closest box: {closest_box}\n")

        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Interactive SDF Visualizer for MuJoCo terrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/sdf_visualizer.py                    # Launch interactive viewer
  python test/sdf_visualizer.py --cli              # CLI mode for coordinate input
  python test/sdf_visualizer.py --point "0.5 0.3 0.2"  # Single query
        """
    )
    parser.add_argument('--xml', type=str, default=None,
                        help='Path to MuJoCo XML file')
    parser.add_argument('--cli', action='store_true',
                        help='Run in CLI mode (input coordinates)')
    parser.add_argument('--point', type=str, default=None,
                        help='Query single point (format: "x y z")')

    args = parser.parse_args()

    # Default XML path
    if args.xml is None:
        mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.xml = os.path.join(mpx_dir, "mpx", "data", "r2-1024", "mjcf", "scene_terrain_test.xml")

    # Check if XML exists
    if not os.path.exists(args.xml):
        print(f"Error: XML file not found: {args.xml}")
        return

    # Create visualizer
    visualizer = SDFVisualizer(args.xml)

    try:
        # Single point query mode
        if args.point:
            coords = [float(x) for x in args.point.split()]
            if len(coords) != 3:
                print("Error: --point requires 3 values (x y z)")
                return
            sdf_val, status, closest_box = visualizer.query_point(*coords)
            print(f"\nQuery: ({coords[0]}, {coords[1]}, {coords[2]})")
            print(f"SDF: {sdf_val:+.6f}")
            print(f"Status: {status}")
            print(f"Closest box: {closest_box}")
            return

        # CLI mode
        if args.cli:
            run_cli_mode(visualizer)
        else:
            # Interactive visualizer mode
            visualizer.run_with_keyboard_thread()

    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    main()
