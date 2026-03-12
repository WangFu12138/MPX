#!/usr/bin/env python3
"""可视化地形场景

使用 MuJoCo 加载并可视化生成的地形场景。

使用方法:
    # 基本可视化
    python mpx/examples/visualize_terrain.py

    # 指定场景文件
    python mpx/examples/visualize_terrain.py --scene path/to/scene.xml

    # 运行仿真（机器人会下落）
    python mpx/examples/visualize_terrain.py --simulate
"""
import argparse
import os
import sys

# 确保可以导入 mpx 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mujoco
import mujoco.viewer
import numpy as np

try:
    from mpx.terrain.heightmap import get_heightmap
    HAS_HEIGHTMAP = True
except ImportError:
    HAS_HEIGHTMAP = False
    print("Warning: Could not import get_heightmap from mpx.terrain.heightmap")


def visualize_scene(scene_path, simulate=False, duration=30.0):
    """
    可视化 MuJoCo 场景

    Args:
        scene_path: 场景 XML 文件路径
        simulate: 是否运行物理仿真
        duration: 仿真时长（秒）
    """
    # 检查文件是否存在
    if not os.path.exists(scene_path):
        print(f"错误: 场景文件不存在: {scene_path}")
        print("\n请先运行地形生成测试:")
        print("  python -m mpx.terrain.test_terrain_gen --size 9 --num-objects 150")
        return

    print(f"\n{'='*60}")
    print(f"加载场景: {scene_path}")
    print(f"{'='*60}\n")

    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"错误: 无法加载模型: {e}")
        return

    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  - 主体数量: {model.nbody}")
    print(f"  - 关节数量: {model.njnt}")
    print(f"  - 几何体数量: {model.ngeom}")
    print(f"  - 自由度: {model.nv}")

    # 统计地形方块
    terrain_boxes = sum(1 for i in range(model.nbody) if model.body(i).name.startswith('box_'))
    print(f"  - 地形方块数量: {terrain_boxes}")

    # 重置到初始姿态
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print(f"  - 初始高度: {data.qpos[2]:.3f}m")

    print(f"\n{'='*60}")
    print("启动可视化...")
    print("{'='*60}")
    print("\n控制说明:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 滚轮: 缩放")
    print("  - 双击: 选择物体")
    print("  - Ctrl + 右键: 施加力")
    print("  - 空格键: 暂停/继续仿真")
    print("  - Backspace: 重置仿真")
    print("  - q: 切换高程图显示 (开启/关闭)")
    print(f"\n{'='*60}\n")

    # 高程图可视化状态（默认关闭以提高性能）
    show_heightmap = True
    heightmap_points = None
    
    # 键盘回调
    def key_callback(keycode):
        nonlocal show_heightmap
        if chr(keycode).lower() == 'q':
            show_heightmap = not show_heightmap
            print(f"高程图显示: {'[开启]' if show_heightmap else '[关闭]'}")

    # 启动被动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角（俯视）
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 45
        viewer.cam.lookat[:] = [0, 0, 0.5]

        step = 0
        while viewer.is_running():
            # 运行仿真
            if simulate:
                mujoco.mj_step(model, data)
                step += 1

                # 每1秒打印一次状态
                if step % int(1.0 / model.opt.timestep) == 0:
                    time = data.time
                    height = data.qpos[2]
                    print(f"时间: {time:.1f}s | 高度: {height:.3f}m", end='\r')

                    # 检查是否仿真超时
                    if time > duration:
                        print(f"\n仿真完成 ({duration}s)")
                        break
            else:
                # 不仿真，仅查看静态场景
                mujoco.mj_forward(model, data)
                step += 1

            viewer.sync()

            # 渲染高程图 (优化：每 60 步提取一次，降低采样密度)
            if HAS_HEIGHTMAP and show_heightmap and (step % 60 == 0):
                try:
                    center_pos = data.qpos[:3]
                    quat = data.qpos[3:7]

                    w, x, y, z = quat
                    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

                    # 获取高程图
                    heightmap_points = get_heightmap(model, data, center_pos, yaw)

                    # 绘制到 viewer.user_scn 中
                    viewer.user_scn.ngeom = 0
                    if heightmap_points is not None:
                        points_flat = heightmap_points.reshape(-1, 3)
                        # 只绘制部分点（进一步降低渲染量）
                        for i, pt in enumerate(points_flat):
                            if i % 2 == 0:  # 隔一个点绘制一个
                                mujoco.mjv_initGeom(
                                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=[0.02, 0, 0],
                                    pos=pt,
                                    mat=np.eye(3).flatten(),
                                    rgba=[1, 0, 0, 0.8]
                                )
                                viewer.user_scn.ngeom += 1

                except Exception as e:
                    print(f"渲染高程图失败: {e}")
                    show_heightmap = False
                    
    print("\n\n可视化结束")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 MuJoCo 地形场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 查看静态场景（默认）
    python mpx/examples/visualize_terrain.py

    # 运行物理仿真
    python mpx/examples/visualize_terrain.py --simulate

    # 指定场景文件
    python mpx/examples/visualize_terrain.py --scene mpx/data/r2-1024/mjcf/scene_terrain_test.xml

    # 仿真10秒
    python mpx/examples/visualize_terrain.py --simulate --duration 10
        """
    )

    parser.add_argument(
        '--scene',
        type=str,
        default='mpx/data/r2-1024/mjcf/scene_terrain_test.xml',
        help='场景 XML 文件路径 (默认: mpx/data/r2-1024/mjcf/scene_terrain_test.xml)'
    )

    parser.add_argument(
        '--simulate',
        action='store_true',
        help='运行物理仿真（机器人会下落）'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='仿真时长（秒，默认: 30.0）'
    )

    args = parser.parse_args()

    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    scene_path = os.path.join(project_root, args.scene)

    # 可视化
    visualize_scene(scene_path, args.simulate, args.duration)


if __name__ == "__main__":
    main()
