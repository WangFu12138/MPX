"""
MuJoCo 地形预览工具 - 查看崎岖地形效果

用法:
    python mpx/utils/preview_terrain.py --type perlin
    python mpx/utils/preview_terrain.py --type steps
    python mpx/utils/preview_terrain.py --type rubble
"""

import mujoco
import mujoco.viewer
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='预览 MuJoCo 地形')
    parser.add_argument('--type', type=str, default='perlin',
                        choices=['perlin', 'steps', 'rubble', 'flat'],
                        help='地形类型')
    args = parser.parse_args()

    # 根据类型选择场景文件
    if args.type == 'flat':
        scene_path = 'mpx/data/r2-1024/mjcf/scene.xml'
        desc = "平坦地面"
    else:
        scene_path = f'mpx/data/r2-1024/mjcf/scene_{args.type}.xml'
        desc = f"{args.type} 地形"

    print(f"\n{'='*50}")
    print(f"加载地形: {desc}")
    print(f"场景文件: {scene_path}")
    print(f"{'='*50}\n")

    # 加载模型
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 打印地形信息
    print(f"模型信息:")
    print(f"  - geom 数量: {model.ngeom}")
    print(f"  - body 数量: {model.nbody}")

    # 查找高度场地形
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'terrain' in geom_name.lower():
            print(f"\n找到地形 geom: {geom_name}")
            print(f"  - 类型: {model.geom_type[i]}")
            print(f"  - 类型名称: {mujoco.mjtGeom(model.geom_type[i]).name}")

    print("\n控制:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 双击: 查看物体")
    print("  - Ctrl+C: 退出\n")

    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("查看器已启动...")

        # 设置相机初始位置
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20
        viewer.cam.distance = 3
        viewer.cam.lookat[:] = [0, 0, 0.3]

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    main()
