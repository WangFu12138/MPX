#!/usr/bin/env python3
"""在地形场景中运行 R2 机器人的 MPC 控制

这个示例展示如何:
1. 加载地形场景
2. 运行 MPC 控制器
3. 机器人在复杂地形上行走

注意: 这是一个基础示例，需要进一步完善地形感知功能。
"""
import argparse
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpx.config import config_r2
from mpx.utils.mpc_wrapper import MPCControllerWrapper


def run_mpc_on_terrain(
    scene_path: str,
    duration: float = 10.0,
    mpc_frequency: int = 50,
    use_mjx: bool = True
):
    """
    在地形场景中运行 MPC 控制

    Args:
        scene_path: 地形场景文件路径
        duration: 仿真时长（秒）
        mpc_frequency: MPC 更新频率（Hz）
        use_mjx: 是否使用 MJX 加速
    """
    # 加载模型
    print(f"\n加载场景: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 重置到初始姿态
    mujoco.mj_resetDataKeyframe(model, data, 0)
    initial_height = data.qpos[2]
    print(f"初始高度: {initial_height:.3f}m")

    # 创建 MPC 控制器
    print("\n初始化 MPC 控制器...")
    config = config_r2
    mpc = MPCControllerWrapper(config)
    mpc_data = mpc.make_data()

    # 转换为 MJX 模型（如果需要）
    if use_mjx:
        print("使用 MJX 加速...")
        mx = mjx.put_model(model)
        dx = mjx.put_data(model, data)

    # MPC 控制参数
    dt = 1.0 / mpc_frequency
    steps_per_mpc = int(dt / model.opt.timestep)

    print(f"\n开始 MPC 控制...")
    print(f"  - 仿真时长: {duration}s")
    print(f"  - MPC 频率: {mpc_frequency}Hz")
    print(f"  - 每次MPC的仿真步数: {steps_per_mpc}")
    print(f"\n{'='*60}\n")

    # 仿真循环
    time = 0.0
    step_count = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机
            viewer.cam.distance = 6.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = [0, 0, 0.8]

            while viewer.is_running() and time < duration:
                # 获取当前状态
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()

                # 运行 MPC（简化版本，实际需要更复杂的状态估计）
                # 这里仅作为示例，实际使用时需要:
                # 1. 从高程图获取地形信息
                # 2. 更新参考轨迹
                # 3. 调整代价函数

                # TODO: 实现 MPC 调用
                # u = mpc.run(mpc_data, qpos, qvel)

                # 应用控制（示例：保持初始姿态）
                data.ctrl[:] = 0

                # 仿真一步
                for _ in range(steps_per_mpc):
                    mujoco.mj_step(model, data)
                    time += model.opt.timestep
                    step_count += 1

                # 更新查看器
                viewer.sync()

                # 打印状态
                if step_count % int(1.0 / model.opt.timestep) == 0:
                    x, y, z = qpos[0], qpos[1], qpos[2]
                    print(f"时间: {time:.1f}s | 位置: ({x:.2f}, {y:.2f}, {z:.3f})m", end='\r')

    except KeyboardInterrupt:
        print(f"\n\n仿真被用户中断")

    print(f"\n\n{'='*60}")
    print("MPC 控制结束")
    print(f"  - 总仿真时间: {time:.1f}s")
    print(f"  - 总仿真步数: {step_count}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="在地形场景中运行 R2 机器人的 MPC 控制"
    )

    parser.add_argument(
        '--scene',
        type=str,
        default='mpx/data/r2-1024/mjcf/scene_terrain_test.xml',
        help='地形场景文件路径'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='仿真时长（秒）'
    )

    parser.add_argument(
        '--mpc-frequency',
        type=int,
        default=50,
        help='MPC 更新频率（Hz）'
    )

    parser.add_argument(
        '--no-mjx',
        action='store_true',
        help='不使用 MJX 加速'
    )

    args = parser.parse_args()

    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    scene_path = os.path.join(project_root, args.scene)

    # 运行 MPC
    run_mpc_on_terrain(
        scene_path,
        args.duration,
        args.mpc_frequency,
        use_mjx=not args.no_mjx
    )


if __name__ == "__main__":
    main()
