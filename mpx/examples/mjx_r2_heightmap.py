#!/usr/bin/env python3
"""R2 机器人高程图可视化演示

基于 mjx_r2.py，添加高程图实时显示功能。
按 q 键切换高程图显示开/关。

使用方法:
    python mpx/examples/mjx_r2_heightmap.py
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

# 尝试导入高程图模块
try:
    from mpx.terrain.heightmap import get_heightmap
    from mpx.terrain.heightmap import create_sensor_matrix_with_mask
    from mujoco import mjx
    HAS_HEIGHTMAP = True
except ImportError as e:
    HAS_HEIGHTMAP = False
    print(f"Warning: 无法导入高程图模块: {e}")

# 高程图模式: "original" 使用 get_heightmap, "training" 使用训练代码的方式
HEIGHTMAP_MODE = "training"  # 改为 "original" 可切换回原来的方式

# 加载模型 - 固定使用 terrain_test 场景
model = mujoco.MjModel.from_xml_path(dir_path + '/../data/r2-1024/mjcf/scene_terrain_test.xml')
data = mujoco.MjData(model)
sim_frequency = 500.0
model.opt.timestep = 1 / sim_frequency

# 初始化接触点 ID
contact_id = []
for name in config.contact_frame:
    contact_id.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

# 初始化 MPC 控制器
mpc = mpc_wrapper.MPCControllerWrapper(config)
data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])

# 键盘控制状态
current_speed = 0.3
current_turn_rate = 0.0

try:
    from pynput import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

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

if HAS_KEYBOARD:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

tau = jnp.zeros(config.n_joints)

print(f"\n{'='*60}")
print("R2 机器人高程图可视化")
print(f"{'='*60}")
print(f"场景: scene_terrain_test.xml")
print(f"高程图模式: {HEIGHTMAP_MODE}")
print(f"\n控制说明:")
print("  - ↑/↓: 增加/减少前进速度")
print("  - ←/→: 左/右转向")
print("  - q: 切换高程图显示")
print("  - 空格: 暂停/继续")
print("  - Backspace: 重置")
print(f"{'='*60}\n")

# 如果使用训练模式，初始化 MJX model/data
if HEIGHTMAP_MODE == "training" and HAS_HEIGHTMAP:
    # 必须先调用 mj_forward 初始化几何位置
    mujoco.mj_forward(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    print("[训练模式] MJX model/data 已初始化")

# 高程图显示状态
show_heightmap = False  # 默认关闭，按 q 开启
heightmap_update_counter = 0
HEIGHTMAP_UPDATE_INTERVAL = 60  # 每60步更新一次高程图

# 用于存储 viewer 引用的容器
viewer_ref = [None]

# 键盘回调函数（需要在 launch_passive 之前定义）
def key_callback(keycode):
    global show_heightmap
    if chr(keycode).lower() == 'q':
        show_heightmap = not show_heightmap
        status = "开启" if show_heightmap else "关闭"
        print(f"[高程图] {status}")
        # 关闭时清除用户场景
        if not show_heightmap and viewer_ref[0] is not None:
            viewer_ref[0].user_scn.ngeom = 0

try:
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 存储 viewer 引用
        viewer_ref[0] = viewer

        # 设置初始相机视角
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0, 0, 0.5]

        mujoco.mj_step(model, data)
        viewer.sync()

        mpc.robot_height = config.robot_height
        mpc.reset(data.qpos.copy(), data.qvel.copy())

        counter = 0
        last_speed = current_speed
        last_turn = current_turn_rate

        while viewer.is_running():
            # 键盘控制
            if HAS_KEYBOARD:
                if key_states['up']:
                    current_speed = min(current_speed + 0.01, 1.5)
                elif key_states['down']:
                    current_speed = max(current_speed - 0.01, -0.5)
                if key_states['left']:
                    current_turn_rate = min(current_turn_rate + 0.01, 1.0)
                elif key_states['right']:
                    current_turn_rate = max(current_turn_rate - 0.01, -1.0)

                if abs(current_speed - last_speed) > 0.1:
                    print(f"速度: {current_speed:.2f} m/s")
                    last_speed = current_speed
                if abs(current_turn_rate - last_turn) > 0.1:
                    print(f"转向: {current_turn_rate:.2f} rad/s")
                    last_turn = current_turn_rate

            qpos = data.qpos.copy()
            qvel = data.qvel.copy()

            # MPC 控制
            if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:
                ref_base_lin_vel = jnp.array([current_speed, 0, 0])
                ref_base_ang_vel = jnp.array([0, 0, current_turn_rate])

                input = np.array([
                    ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_lin_vel[2],
                    ref_base_ang_vel[0], ref_base_ang_vel[1], ref_base_ang_vel[2],
                    1.0
                ])

                contact = np.zeros(config.n_contact)
                tau, q, dq = mpc.run(qpos, qvel, input, contact)

            counter += 1
            data.ctrl = tau - 3 * qvel[6:6 + config.n_joints]
            # 手臂控制器设为初始角度 (position controller)
            data.ctrl[3:11] = config.q0[3:11]   # 左臂
            data.ctrl[11:19] = config.q0[11:19]  # 右臂
            mujoco.mj_step(model, data)

            # 高程图渲染（降低频率以提高性能）
            if HAS_HEIGHTMAP and show_heightmap:
                heightmap_update_counter += 1
                if heightmap_update_counter >= HEIGHTMAP_UPDATE_INTERVAL:
                    heightmap_update_counter = 0
                    try:
                        center_pos = data.qpos[:3]
                        quat = data.qpos[3:7]

                        # 计算偏航角
                        w, x, y, z = quat
                        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

                        # 根据模式获取高程图
                        if HEIGHTMAP_MODE == "training":
                            # 使用训练代码的方式：create_sensor_matrix_with_mask
                            # 参数与 dataset_generator.py 一致
                            H, W = 21, 21
                            dist = 0.05  # heightmap_resolution

                            hm, hit_mask = create_sensor_matrix_with_mask(
                                mjx_model, mjx_data, center_pos,
                                yaw=yaw,
                                key=None,
                                dist_x=dist,
                                dist_y=dist,
                                num_heightscans=H,
                                num_widthscans=W
                            )

                            # 处理未命中的 ray（与训练代码一致）
                            fallback_z = center_pos[2]
                            miss_mask = ~hit_mask
                            hm_z = jnp.where(miss_mask, fallback_z, hm[..., 2])
                            heightmap_points = hm.at[..., 2].set(hm_z)

                            # 转换为 numpy
                            heightmap_points = np.array(heightmap_points)
                        else:
                            # 使用原来的方式
                            heightmap_points = get_heightmap(model, data, center_pos, yaw)

                        # 渲染到用户场景
                        viewer.user_scn.ngeom = 0
                        if heightmap_points is not None:
                            points_flat = heightmap_points.reshape(-1, 3)
                            max_geoms = len(viewer.user_scn.geoms)  # 获取最大几何体数量
                            # 隔点渲染以提高性能
                            for i, pt in enumerate(points_flat):
                                if i % 3 == 0:  # 每3个点渲染1个
                                    if viewer.user_scn.ngeom >= max_geoms:
                                        break
                                    # 根据高度设置颜色（蓝色低，红色高）
                                    height = pt[2]
                                    h_normalized = np.clip((height - 0.0) / 0.3, 0, 1)
                                    rgba = [h_normalized, 0.2, 1 - h_normalized, 0.7]

                                    mujoco.mjv_initGeom(
                                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                        size=[0.015, 0, 0],
                                        pos=pt,
                                        mat=np.eye(3).flatten(),
                                        rgba=rgba
                                    )
                                    viewer.user_scn.ngeom += 1

                    except Exception as e:
                        print(f"[高程图] 渲染失败: {e}")

            viewer.sync()

except KeyboardInterrupt:
    print("\n[用户中断]")
finally:
    if HAS_KEYBOARD:
        listener.stop()

print("\n程序结束")
