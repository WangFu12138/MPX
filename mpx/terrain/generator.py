"""Terrain generator for MPX project.

This module generates procedural terrains using Wave Function Collapse (WFC) algorithm.
Supports stairs, turning stairs, rough ground, and other complex terrain types.
"""
import xml.etree.ElementTree as xml_et
import numpy as np
import os
import sys
import argparse
import random
import string

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpx.terrain.wfc.wfc import WFCCore
from mpx.terrain.get_indexes import *


def _get_robot_paths(robot_name="r2"):
    """Get robot-specific file paths."""
    # Get the mpx package directory
    mpx_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(mpx_dir, "data", "r2-1024", "mjcf")

    return {
        "input": os.path.join(data_dir, "scene.xml"),
        "output": os.path.join(data_dir, "scene_terrain.xml"),
        "test": os.path.join(data_dir, "scene_terrain_test.xml"),
        "terrain_boxes": os.path.join(data_dir, "scene_terrain_boxes.xml"),
    }


# Default robot
ROBOT = "r2"
_paths = _get_robot_paths(ROBOT)
INPUT_SCENE_PATH = _paths["input"]
OUTPUT_SCENE_PATH = _paths["output"]
TEST_SCENE_PATH = _paths["test"]
TERRAIN_BOXES_PATH = _paths["terrain_boxes"]


def euler_to_quat(roll, pitch, yaw):
    """Convert zyx euler angles to quaternion."""
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def euler_to_rot(roll, pitch, yaw):
    """Convert zyx euler angles to rotation matrix."""
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def rot2d(x, y, yaw):
    """2D rotation."""
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


def rot3d(pos, euler):
    """3D rotation."""
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    """Convert list to space-separated string."""
    return " ".join(str(s) for s in vec)


def random_box_name():
    """Generate random box name."""
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"box_{suffix}"


class TerrainGenerator:
    """Generate procedural terrains for MuJoCo simulation."""

    def __init__(self, width=None, step_height=None, num_stairs=None, length=None,
                 render=False, max_bodies=None, input_scene_path=None) -> None:
        """
        Initialize terrain generator.

        Args:
            width: Width of each stair step
            step_height: Height of each stair step
            num_stairs: Number of stairs
            length: Total length of terrain
            render: If True, render to XML; if False, store box data
            max_bodies: Maximum number of box bodies to create
            input_scene_path: Path to input scene XML (defaults to R2 scene)
        """
        # Use provided path or default
        scene_path = input_scene_path if input_scene_path else INPUT_SCENE_PATH

        if os.path.exists(scene_path):
            self.scene = xml_et.parse(scene_path)
            self.root = self.scene.getroot()
            self.worldbody = self.root.find("worldbody")
            self.asset = self.root.find("asset")
        else:
            # Create a minimal scene if input doesn't exist
            self.root = xml_et.Element("mujoco")
            self.root.set("model", "terrain_scene")
            self.worldbody = xml_et.SubElement(self.root, "worldbody")
            self.asset = xml_et.SubElement(self.root, "asset")
            self.scene = xml_et.ElementTree(self.root)

        self.count_boxes = 0
        self.render = render
        self.max_bodies = max_bodies

        args = [width, step_height, num_stairs, length]
        if args.count(None) != 1:
            raise ValueError("Exactly three arguments must be provided.")
        if num_stairs is None:
            self.num_stairs = int(length / width)
            self.width, self.step_height, self.length = width, step_height, length
        elif length is None:
            self.length = num_stairs * width
            self.width, self.step_height, self.num_stairs = width, step_height, num_stairs

        self.block_height = self.num_stairs * self.step_height
        self.box_data = []

    def AddBox(self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[2, 2, 2]):
        """Add a box to the terrain."""
        if self.render:
            if self.max_bodies is not None and self.count_boxes >= self.max_bodies - 1:
                return
            # Create a new body inside worldbody
            box_body = xml_et.SubElement(self.worldbody, "body")
            box_body.attrib["pos"] = list_to_str(position)
            box_body.attrib["quat"] = list_to_str(euler_to_quat(euler[0], euler[1], euler[2]))
            box_body.attrib["name"] = random_box_name()

            geo = xml_et.SubElement(box_body, "geom")
            geo.attrib["type"] = "box"
            geo.attrib["size"] = list_to_str(0.5 * np.array(size))  # Half size for MuJoCo
            geo.attrib["contype"] = "2"
            geo.attrib["conaffinity"] = "1"
            self.count_boxes += 1
        else:
            quat = euler_to_quat(euler[0], euler[1], euler[2])
            self.box_data.append({
                "pos": np.array(position),
                "size": 0.5 * np.array(size),
                "quat": np.array(quat)
            })
            self.count_boxes += 1

    def AddGeometry(self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0],
                    size=[0.1, 0.1], geo_type="box"):
        """Add a geometry to the terrain."""
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self, init_pos=[0.0, 0.0, 0.0], yaw=0.0):
        """Add stairs to the terrain."""
        length = self.length
        height = self.step_height
        stair_nums = self.num_stairs
        width = self.width

        local_pos = [-width / 2, length / 2, 0.]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height

            x, y = rot2d(local_pos[0] - stair_nums * width / 2,
                         local_pos[1] - stair_nums * width / 2, yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw],
                [width, length, local_pos[2]]
            )

    def AddFlat(self, init_pos=[0.0, 0.0, 0.0], height=1., width=0.1):
        """Add a flat platform to the terrain."""
        length = self.length
        if height > 0.:
            self.AddBox([init_pos[0], init_pos[1], height / 2],
                       [0.0, 0.0, 0.], [length, length, height])
        else:
            self.AddBox([init_pos[0], init_pos[1], height - width / 2],
                       [0.0, 0.0, 0.], [length, length, width])

    def AddTurningStairsUp(self, init_pos=[0.0, 0.0, 0.0], yaw=np.pi / 2):
        """Add turning stairs going up."""
        width = self.width
        height = self.step_height
        stair_nums = self.num_stairs
        local_pos = [-stair_nums * width / 2 - width / 2, +stair_nums * width / 2, 0.]

        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[1] -= width / 2
            local_pos[2] += height

            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw],
                [width, width + width * i, local_pos[2]]
            )

        local_pos = [stair_nums * width / 2 - width / 2, stair_nums * width / 2, height]
        for i in range(stair_nums - 1):
            local_pos[0] -= width
            local_pos[1] -= width / 2
            local_pos[2] += height

            x, y = rot2d(local_pos[0], local_pos[1], yaw + np.pi / 2)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw + np.pi / 2],
                [width, width + width * i, local_pos[2]]
            )

    def AddTurningStairsDown(self, init_pos=[0.0, 0.0, 0.0], yaw=np.pi / 2):
        """Add turning stairs going down."""
        width = self.width
        height = self.step_height
        stair_nums = self.num_stairs
        local_pos = [-stair_nums * width / 2 - width / 2, +stair_nums * width / 2,
                    stair_nums * height + height]

        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[1] -= width / 2
            local_pos[2] -= height

            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw],
                [width, width + width * i, local_pos[2]]
            )

        local_pos = [stair_nums * width / 2 - width / 2, stair_nums * width / 2,
                    (stair_nums - 1) * height + height]
        for i in range(stair_nums - 1):
            local_pos[0] -= width
            local_pos[1] -= width / 2
            local_pos[2] -= height

            x, y = rot2d(local_pos[0], local_pos[1], yaw + np.pi / 2)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2] / 2 + init_pos[2]],
                [0.0, 0.0, yaw + np.pi / 2],
                [width, width + width * i, local_pos[2]]
            )

    def AddRoughGround(self, init_pos=[1.0, 0.0, 0.0], euler=[0.0, -0.0, 0.0],
                       nums=[10, 10], box_size=[0.5, 0.5, 0.5], box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2], box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2], separation_rand=[0.05, 0.05]):
        """Add rough ground with randomized boxes."""
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1.0, 1.0, 2)

        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def Save(self, filename=None):
        """Save the terrain scene to XML file."""
        if filename is None:
            self.scene.write(OUTPUT_SCENE_PATH)
        else:
            self.scene.write(filename)


def generate_14(size, test=False):
    """Generate a WFC map with 14 tile types."""
    left = (-1, 0)
    right = (1, 0)
    up = (0, 1)
    down = (0, -1)

    directions = [left, down, right, up]
    connections = {
        0: {left: (0, 4, 10, 11), down: (0, 5, 11, 12), right: (0, 2, 12, 13), up: (0, 3, 13, 10)},
        1: {left: (1, 2, 6, 7), down: (1, 3, 7, 8), right: (1, 4, 8, 9), up: (1, 5, 9, 6)}
    }

    connections.update(Stairs(directions))
    connections.update(StairsTurningUp(directions))
    connections.update(StairsTurningDown(directions))

    wfc = WFCCore(14, connections, (size, size))

    # Initialize outer boundary as tile 0 (outer/empty)
    outer = 0
    for x in range(size):
        wfc.init((x, 0), outer)
        wfc.init((x, size - 1), outer)

    for y in range(1, size - 1):
        wfc.init((0, y), outer)
        wfc.init((size - 1, y), outer)

    # Initialize center - prefer tile 1 (flat platform) to ensure terrain is generated
    # Tile 0 = outer/empty (no boxes), Tile 1 = flat platform (generates boxes)
    # When test=True, we want reliable terrain, so use tile 1
    # Otherwise, randomly choose with 70% chance for tile 1 (flat) or stairs
    if test:
        # For testing, always use flat platform to ensure boxes are generated
        wfc.init((size // 2, size // 2), 1)
    else:
        # For training, add variety: flat (1) or random stairs (2-5)
        if np.random.random() < 0.7:
            wfc.init((size // 2, size // 2), 1)  # Flat platform
        else:
            # Random stair direction
            wfc.init((size // 2, size // 2), np.random.randint(2, 6))

    wfc.solve()
    wave = wfc.wave.wave

    return wave


def addElement(map: TerrainGenerator, index, pos):
    """Add a terrain element based on WFC tile index."""
    height = map.block_height
    if index == 0:
        # tile 0 is outer boundary (empty) - don't add anything
        return
    elif index == 1:
        # tile 1 is flat platform
        map.AddFlat(init_pos=[pos[0], pos[1], 0.], height=height)
    elif index == 2:
        # tile 2: stairs going up (yaw=0)
        map.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=0)
    elif index == 3:
        # tile 3: stairs going up (yaw=pi/2)
        map.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=np.pi/2)
    elif index == 4:
        # tile 4: stairs going up (yaw=pi)
        map.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=np.pi)
    elif index == 5:
        # tile 5: stairs going up (yaw=-pi/2)
        map.AddStairs(init_pos=[pos[0], pos[1], 0], yaw=-np.pi/2)
    elif index == 6:
        # tile 6: turning stairs up (yaw=0)
        map.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=0.)
    elif index == 7:
        # tile 7: turning stairs up (yaw=pi/2)
        map.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=np.pi/2)
    elif index == 8:
        # tile 8: turning stairs up (yaw=pi)
        map.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=np.pi)
    elif index == 9:
        # tile 9: turning stairs up (yaw=3*pi/2)
        map.AddTurningStairsUp(init_pos=[pos[0], pos[1], 0.], yaw=3*np.pi/2)
    elif index == 10:
        # tile 10: turning stairs down (yaw=0)
        map.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0.], yaw=0.)
    elif index == 11:
        # tile 11: turning stairs down (yaw=pi/2)
        map.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0.], yaw=np.pi/2)
    elif index == 12:
        # tile 12: turning stairs down (yaw=pi)
        map.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0.], yaw=np.pi)
    elif index == 13:
        # tile 13: turning stairs down (yaw=-pi/2)
        map.AddTurningStairsDown(init_pos=[pos[0], pos[1], 0.], yaw=-np.pi/2)


def create_centered_grid(N, d):
    """Create a centered grid of positions."""
    half_size = (N - 1) / 2
    x = (np.arange(N) - half_size) * d
    y = (np.arange(N) - half_size) * d
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = np.stack((X, Y), axis=-1)
    return grid


def create_random_matrix(num_envs, num_bodies, size, height_min, height_max):
    """Create random terrain matrices for multiple environments.

    Note: This function uses numpy instead of jax.numpy for compatibility.
    """
    result_matrix = np.ones((num_envs, num_bodies, 10))
    incremented_values = np.arange(100, 100 + num_envs * num_bodies).reshape(num_envs, num_bodies, 1)
    result_matrix[..., :3] = incremented_values
    result_matrix[..., 3:7] = np.array([1, 0, 0, 0])

    for env_id in range(num_envs):
        height = np.random.uniform(height_min, height_max)
        width = np.random.uniform(0.25, 0.4)
        num_steps = np.random.choice([1, 2, 3, 4])

        tg = TerrainGenerator(width=width, step_height=height, num_stairs=num_steps, render=False)
        wave = generate_14(size=size)
        grid = create_centered_grid(size, tg.length)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                addElement(tg, wave[i, j], grid[i, j])

        print(f"Environment {env_id}: {tg.count_boxes} boxes generated")

        for i, box in enumerate(tg.box_data):
            result_matrix[env_id, i, :3] = box["pos"]
            result_matrix[env_id, i, 3:7] = box["quat"]
            result_matrix[env_id, i, 7:] = box["size"]

    return result_matrix


def random_test_env(num_bodies, size, step_height=0.08, width=0.4, num_steps=4,
                    output_path=None):
    """Generate a random test environment."""
    tg = TerrainGenerator(width=width, step_height=step_height, num_stairs=num_steps, render=True)
    wave = generate_14(size=size, test=True)
    grid = create_centered_grid(size, tg.length)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            addElement(tg, wave[i, j], grid[i, j])

    print(f"Total boxes: {tg.count_boxes}")

    # Fill remaining bodies with placeholder boxes
    if tg.count_boxes < num_bodies - 1:
        for i in range(num_bodies - tg.count_boxes - 1):
            tg.AddBox([100 + i, 100 + i, 10])

    output = output_path if output_path else TEST_SCENE_PATH
    tg.Save(output)
    print(f"Saved terrain to: {output}")
    return wave


def fill_terrain_template(num_objects, num_steps=3, width=0.1, step_height=0.15,
                         output_path=None):
    """Fill terrain template with placeholder boxes."""
    tg = TerrainGenerator(width=width, step_height=step_height, num_stairs=num_steps, render=True)
    for i in range(num_objects):
        tg.AddBox([100 + i, 100 + i, 10])

    output = output_path if output_path else OUTPUT_SCENE_PATH
    tg.Save(output)
    print(f"Wrote {tg.count_boxes} placeholder boxes -> {output}")


def _setup_paths(robot):
    """Setup paths for a specific robot."""
    global INPUT_SCENE_PATH, OUTPUT_SCENE_PATH, TEST_SCENE_PATH, TERRAIN_BOXES_PATH
    paths = _get_robot_paths(robot)
    INPUT_SCENE_PATH = paths["input"]
    OUTPUT_SCENE_PATH = paths["output"]
    TEST_SCENE_PATH = paths["test"]
    TERRAIN_BOXES_PATH = paths["terrain_boxes"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terrain Generator for MPX")
    parser.add_argument('--robot', type=str, default='r2', help='Robot name: r2')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # test mode — create a random test terrain
    p_test = subparsers.add_parser('test', help='Generate a random test terrain')
    p_test.add_argument('--num_objects', type=int, default=100)
    p_test.add_argument('--size', type=int, default=9)
    p_test.add_argument('--step_height', type=float, default=0.08, help='Height of each stair step')
    p_test.add_argument('--width', type=float, default=0.35, help='Width of each stair step')
    p_test.add_argument('--num_steps', type=int, default=3, help='Number of stair steps')

    # fill mode — fill the template scene XML with placeholder boxes
    p_fill = subparsers.add_parser('fill', help='Fill terrain template with placeholder boxes')
    p_fill.add_argument('--num_objects', type=int, default=100)
    p_fill.add_argument('--num_steps', type=int, default=3)
    p_fill.add_argument('--width', type=float, default=0.1)
    p_fill.add_argument('--step_height', type=float, default=0.15)

    # level mode — produce one terrain file from 0 to a given height
    p_level = subparsers.add_parser('level', help='Generate a single terrain .npy file')
    p_level.add_argument('--num_envs', type=int, default=100)
    p_level.add_argument('--num_objects', type=int, default=100)
    p_level.add_argument('--size', type=int, default=5)
    p_level.add_argument('--height', type=float, default=0.1, help='Max step height')
    p_level.add_argument('--height_min', type=float, default=None, help='Min step height')

    args = parser.parse_args()
    _setup_paths(args.robot)
    np.set_printoptions(precision=2, suppress=True)

    if args.mode == 'test':
        random_test_env(args.num_objects, args.size, step_height=args.step_height,
                       width=args.width, num_steps=args.num_steps)

    elif args.mode == 'fill':
        fill_terrain_template(args.num_objects, args.num_steps, args.width, args.step_height)

    elif args.mode == 'level':
        height_min = args.height_min if args.height_min is not None else args.height / 2
        res = create_random_matrix(args.num_envs, args.num_objects, args.size,
                                  height_min, args.height)
        level_name = f"level{round(args.height * 100):02d}"
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'terrains')
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, level_name), res)
        print(f"  -> terrains/{level_name}.npy  shape={res.shape}")
