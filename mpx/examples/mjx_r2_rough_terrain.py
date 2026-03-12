"""
MPX 崎岖地形示例 - R2-1024 机器人在 Perlin 噪声地形上行走

运行方式:
    python mpx/examples/mjx_r2_rough_terrain.py

控制:
    ↑/↓ - 调整前进速度
    ←/→ - 调整转向速度
"""

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))

import jax
import jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
import mujoco
import mujoco.viewer
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_r2 as config
from timeit import default_timer as timer

# 尝试导入键盘控制库
try:
    from pynput import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Warning: 'pynput' library not found. Install with: pip install pynput")


def main():
    # ==================== 地形选择 ====================
    # 可选: "scene_perlin.xml" - 自然起伏地形
    #       "scene_steps.xml"  - 阶梯地形
    #       "scene_rubble.xml" - 碎石地形
    #       "scene.xml"        - 平坦地面（原始）
    terrain_type = "scene_perlin.xml"

    # 加载模型和场景
    scene_path = os.path.join(dir_path, '../data/r2-1024/mjcf', terrain_type)
    print(f"\n{'='*50}")
    print(f"地形类型: {terrain_type}")
    print(f"{'='*50}\n")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    sim_frequency = 500.0
    model.opt.timestep = 1/sim_frequency

    contact_id = []
    for name in config.contact_frame:
        contact_id.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    # 初始化 MPC 控制器
    mpc = mpc_wrapper.MPCControllerWrapper(config)

    # 设置初始状态（稍微抬高以适应崎岖地形）
    p0_rough = config.p0.at[2].set(0.7)  # 将 z 从 0.64 提高到 0.7
    data.qpos = jnp.concatenate([p0_rough, config.quat0, config.q0])

    # 键盘控制状态
    current_speed = 0.3  # 降低默认速度以适应崎岖地形
    current_turn_rate = 0.0
    key_states = {'up': False, 'down': False, 'left': False, 'right': False}

    def on_press(key):
        try:
            if key == keyboard.Key.up:
                key_states['up'] = True
            elif key == keyboard.Key.down:
                key_states['down'] = True
            elif key == keyboard.Key.left:
                key_states['left'] = True
            elif key == keyboard.Key.right:
                key_states['right'] = True
        except:
            pass

    def on_release(key):
        try:
            if key == keyboard.Key.up:
                key_states['up'] = False
            elif key == keyboard.Key.down:
                key_states['down'] = False
            elif key == keyboard.Key.left:
                key_states['left'] = False
            elif key == keyboard.Key.right:
                key_states['right'] = False
        except:
            pass

    # 启动键盘监听
    if HAS_KEYBOARD:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    # 启动仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_step(model, data)
        viewer.sync()

        delay = 0
        print(f'MPC 延迟: {delay} 仿真步')

        if HAS_KEYBOARD:
            print('键盘控制: ↑/↓ 调整速度, ←/→ 调整转向')
        else:
            print(f'无键盘库，速度固定为: {current_speed} m/s')

        # 调整机器人高度以适应地形
        mpc.robot_height = 0.7
        mpc.reset(data.qpos.copy(), data.qvel.copy())

        counter = 0
        last_speed = current_speed
        last_turn = current_turn_rate
        total_time = 0
        mpc_time = 0

        print("\n开始仿真... 按 Ctrl+C 退出\n")

        while viewer.is_running():
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()

            # 每 10 个仿真步执行一次 MPC (500Hz / 50Hz = 10)
            if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:

                if counter != 0 and delay > 0:
                    # 模拟 MPC 计算延迟
                    for i in range(delay):
                        qpos = data.qpos.copy()
                        qvel = data.qvel.copy()
                        tau_fb = -3 * (qvel[6:6+config.n_joints])
                        data.ctrl = tau_fb
                        mujoco.mj_step(model, data)
                        counter += 1

                start = timer()

                # 设置参考输入
                ref_base_lin_vel = jnp.array([current_speed, 0, 0])
                ref_base_ang_vel = jnp.array([0, 0, current_turn_rate])

                input = np.array([
                    ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_lin_vel[2],
                    ref_base_ang_vel[0], ref_base_ang_vel[1], ref_base_ang_vel[2],
                    1.0  # 机器人高度
                ])

                # 接触状态（目前全为 0，即"盲"步态）
                contact = np.zeros(config.n_contact)

                # 运行 MPC
                tau, q, dq = mpc.run(qpos, qvel, input, contact)

                stop = timer()
                mpc_time += (stop - start)

            # 键盘控制更新
            if HAS_KEYBOARD:
                if key_states['up']:
                    current_speed = min(current_speed + 0.01, 1.0)
                elif key_states['down']:
                    current_speed = max(current_speed - 0.01, -0.5)
                if key_states['left']:
                    current_turn_rate = min(current_turn_rate + 0.01, 1.0)
                elif key_states['right']:
                    current_turn_rate = max(current_turn_rate - 0.01, -1.0)

                # 打印状态变化
                if abs(current_speed - last_speed) > 0.1:
                    print(f"速度: {current_speed:.1f} m/s")
                    last_speed = current_speed
                if abs(current_turn_rate - last_turn) > 0.1:
                    print(f"转向角速度: {current_turn_rate:.1f} rad/s")
                    last_turn = current_turn_rate

            # 应用控制（MPC 输出 + 阻尼反馈）
            data.ctrl = tau - 3 * qvel[6:6+config.n_joints]
            data.ctrl[3:19] = 0  # 禁用某些关节

            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()

            counter += 1
            total_time += model.opt.timestep

            # 每 5 秒打印一次统计
            if counter % int(5 * sim_frequency) == 0:
                avg_mpc_time = mpc_time / (counter / (sim_frequency / config.mpc_frequency))
                print(f"时间: {total_time:.1f}s | 平均 MPC 时间: {avg_mpc_time*1000:.1f}ms | "
                      f"速度: {current_speed:.2f} m/s | 位置: ({qpos[0]:.2f}, {qpos[1]:.2f})")


if __name__ == '__main__':
    main()
