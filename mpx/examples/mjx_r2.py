import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import numpy as np
import mujoco
import mujoco.viewer
import numpy as np
from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_r2 as config
import threading
import pickle
from datetime import datetime

# ==================== 数据记录器 ====================
class MPCDataRecorder:
    """用于记录 MPC 控制过程中的轨迹数据"""
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有记录数据"""
        self.time_stamps = []
        self.base_positions = []
        self.base_orientations = []
        self.base_velocities = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []
        self.planned_trajectories = []
        self.foot_positions = []
        self.contact_states = []
        self.reference_velocities = []

    def record_step(self, t, qpos=None, qvel=None, tau=None,
                    planned_traj=None, foot_pos=None, contact_state=None,
                    ref_vel=None):
        """记录单个时间步的数据"""
        self.time_stamps.append(t)

        if qpos is not None:
            # qpos = [px, py, pz, qw, qx, qy, qz, j1, j2, ..., jn]
            self.base_positions.append(qpos[:3].copy())
            self.base_orientations.append(qpos[3:7].copy())
            self.joint_positions.append(qpos[7:].copy())

        if qvel is not None:
            # qvel = [vx, vy, vz, wx, wy, wz, jv1, jv2, ..., jvn]
            self.base_velocities.append(qvel[:6].copy())
            self.joint_velocities.append(qvel[6:].copy())

        if tau is not None:
            self.joint_torques.append(tau.copy())

        if planned_traj is not None:
            self.planned_trajectories.append(planned_traj.copy() if isinstance(planned_traj, np.ndarray) else planned_traj)

        if foot_pos is not None:
            self.foot_positions.append(foot_pos)

        if contact_state is not None:
            self.contact_states.append(contact_state.copy() if isinstance(contact_state, np.ndarray) else contact_state)

        if ref_vel is not None:
            self.reference_velocities.append(ref_vel.copy() if isinstance(ref_vel, np.ndarray) else ref_vel)

    def save(self, filename=None):
        """保存数据到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mpc_trajectory_data_{timestamp}.pkl"

        def safe_array(data):
            """安全转换为 numpy 数组"""
            if data is None or (hasattr(data, '__len__') and len(data) == 0):
                return None
            return np.array(data)

        data = {
            'time': safe_array(self.time_stamps),
            'base_position': safe_array(self.base_positions),
            'base_orientation': safe_array(self.base_orientations),
            'base_velocity': safe_array(self.base_velocities),
            'joint_position': safe_array(self.joint_positions),
            'joint_velocity': safe_array(self.joint_velocities),
            'joint_torque': safe_array(self.joint_torques),
            'planned_trajectory': self.planned_trajectories,
            'foot_position': self.foot_positions,
            'contact_state': safe_array(self.contact_states),
            'reference_velocity': safe_array(self.reference_velocities),
        }

        project_root = os.path.abspath(os.path.join(dir_path, "..", ".."))
        save_path = os.path.join(project_root, "data", "trajectories", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n[数据记录器] 数据已保存到: {save_path}")
        return save_path

# 全局数据记录器实例
data_recorder = MPCDataRecorder()

# 尝试导入 pynput 库（不需要 root）
try:
    from pynput import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Warning: 'pynput' library not found. Install with: pip install pynput")

model = mujoco.MjModel.from_xml_path(dir_path+'/../data/r2-1024/mjcf/scene.xml')
data = mujoco.MjData(model)
sim_frequency = 500.0
model.opt.timestep = 1/sim_frequency

contact_id = []
for name in config.contact_frame:
    contact_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name))
# 初始化MPC控制器
mpc = mpc_wrapper.MPCControllerWrapper(config)
# 设置初始状态
data.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])

from timeit import default_timer as timer

ids = []
tau = jnp.zeros(config.n_joints)

# Keyboard control state
current_speed = 0.5
current_turn_rate = 0.0
key_states = {'up': False, 'down': False, 'left': False, 'right': False}

def on_press(key):
    """Key press handler"""
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
    """Key release handler"""
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

# Start keyboard listener in background
if HAS_KEYBOARD:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# 重置数据记录器
data_recorder.reset()
record_data = True

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_step(model, data)
        viewer.sync()
        delay = int(0*sim_frequency)
        print('Delay: ',delay)
        if HAS_KEYBOARD:
            print('Keyboard control: ↑/↓ speed, ←/→ turn')
        else:
            print('No keyboard library. Speed fixed at:', current_speed)
        print('[提示] 按 Ctrl+C 或关闭窗口可保存数据并退出')
        mpc.robot_height = config.robot_height
        mpc.reset(data.qpos.copy(),data.qvel.copy())
        counter = 0
        last_speed = current_speed
        last_turn = current_turn_rate

        while viewer.is_running():
            # Check keyboard input (using pynput key states)
            if HAS_KEYBOARD:
                if key_states['up']:
                    current_speed += 0.01
                elif key_states['down']:
                    current_speed -= 0.01
                if key_states['left']:
                    current_turn_rate += 0.01
                elif key_states['right']:
                    current_turn_rate -= 0.01
                # Print if values changed significantly
                if abs(current_speed - last_speed) > 0.1:
                    print(f"Speed: {current_speed:.1f} m/s")
                    last_speed = current_speed
                if abs(current_turn_rate - last_turn) > 0.1:
                    print(f"Turn rate: {current_turn_rate:.1f} rad/s")
                    last_turn = current_turn_rate

            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            # 每500/50 10个仿真步执行一次MPC
            if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:

                if counter != 0:
                    # 模拟MPC计算延迟,期间只使用力反馈控制阻尼关节
                    for i in range(delay):
                        qpos_delay = data.qpos.copy()
                        qvel_delay = data.qvel.copy()
                        tau_fb = -3*(qvel_delay[6:6+config.n_joints])
                        data.ctrl = tau + tau_fb
                        mujoco.mj_step(model, data)
                        counter += 1

                # 设置参考输入（使用键盘控制的动态速度）
                ref_base_lin_vel = jnp.array([current_speed, 0, 0])
                ref_base_ang_vel = jnp.array([0, 0, current_turn_rate])

                input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                               ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                               1.0])

                #set this to the current contact state to use the blind step adaptation
                contact = np.zeros(config.n_contact)

                tau, q, dq  = mpc.run(qpos,qvel,input,contact)
            counter += 1
            data.ctrl = tau - 3*qvel[6:6+config.n_joints]
            data.ctrl[3:11] = config.q0[3:11]   # 左臂
            data.ctrl[11:19] = config.q0[11:19]  # 右臂
            mujoco.mj_step(model, data)
            viewer.sync()

            # 记录数据（每10个仿真步记录一次，减少数据量）
            if record_data and counter % 10 == 0:
                sim_time = counter * model.opt.timestep

                # 获取足端位置
                foot_positions = {}
                for i, frame_name in enumerate(config.contact_frame):
                    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
                    if geom_id >= 0:
                        foot_positions[frame_name] = data.geom_xpos[geom_id].copy()

                # 记录当前步的数据
                data_recorder.record_step(
                    t=sim_time,
                    qpos=qpos,
                    qvel=qvel,
                    tau=np.array(tau),
                    foot_pos=foot_positions,
                    contact_state=contact if 'contact' in dir() else None,
                    ref_vel=np.array([current_speed, 0, 0, 0, 0, current_turn_rate])
                )

except KeyboardInterrupt:
    print("\n[用户中断] 正在保存数据...")
finally:
    # 保存数据
    if record_data and len(data_recorder.time_stamps) > 0:
        save_path = data_recorder.save("mpc_r2_trajectory.pkl")
        print(f"\n[完成] MPC轨迹数据已保存。共记录 {len(data_recorder.time_stamps)} 个时间步。")

    if HAS_KEYBOARD:
        listener.stop()

