import jax.numpy as jnp
import jax
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mpx.utils.mpc_utils as mpc_utils
from functools import partial
import os

from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/r2-1024/mjcf/robot.xml'  # Path to the MuJoCo model XML file

    10-# 猬模型名称和相关配置
    11- contact_frame = ['foot_left_1', 'foot_left_2', 'foot_left_3', 'foot_left_4']
    14-                'foot_right_1', 'foot_right_2', 'foot_right_3', 'foot_right_4']
    15>body_name = ['Lankle-roll', 'Rankle-roll']
    16-
    17-# 时间和阶段参数
    18>dt = 0.02  # 时间步 in seconds
    19>N = 25      # Number of stages
    20=mpc_frequency = 50  # Frequency of MPC updates in Hz
    21- timer_t = jnp.array([0.5,0.5,0.5,0.5,0.0,0.0,5.0])  # Timer values for each leg
    24- duty_factor = 0.7  # Duty factor for the gait
    25- step_freq = 1.2   # Step frequency in Hz
    26- step_height = 0.03 # Step height in meters
    27- initial_height = 0.65 # Initial height of the robot's base in meters
    28- robot_height = 0.65  # Height of the robot's base in meters
    29- p0 = jnp.array([0, 0, 0.64])  # Initial position of the robot's base
    32- quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)
    34- q0 = jnp.array([
                    35-                0 ,0 ,0,
                    36-                0 ,0.8 , 0 , 0 , 0 , 0 , 0 , 0 , 0,
                    37-                -0.715, 0.00009, -0.00058, -1.18, -0.464, -0.0000731,
                    38-                0.706, -0.0633 ,0.054, 1.16, 0.451, -0.0835
                    39-                0.706, -0.0633 , 0.054, 1.16, 0.451, -0.0835
                    ])  # Initial joint angles
    41-
    42- p_legs0 = jnp.array([ 0.12592681,  0.135, 0.01690434,
                    44-                     -0.10407319,  0.135, 0.01690434,
                    45-                     -0.10407319,  0.015, 0.01690434,
                    46-                      0.12592681, -0.015, 0.01690434,
                    47-                     -0.10407319, -0.015, 0.01690434,
                    48-                     -0.10407319, -0.135, 0.01690434,
                    49-                     -0.10407319, -0.015, 0.01690434])
    50- p_legs0 = jnp.array([0.142448,  0.046328, 0.01690434,  # 前左脚
参考原始场景
                    -0.015, 0.01690434,
                    -0.047141, -0.053176, 0.01690434,
                    -0.080451, -0.053176, 0.01690434,
                    0.041828, -0.046328, 0.01690434)

 - 前左脚 x坐标完全匹配MuJoCo模型
                    - `p_legs0` 中的y坐标范围也缩小到接近 0.135 -> 0.015
                    - z坐标更接近真实值，- 彽台阶场景相比，原始配置脚间距过大了很多（原来y≈0.135， 现在约 0.13m)


 - **z坐标缩放**：**robot在台阶上行走时，质心会向前偏移，**
                    歧 **根节点**：**机器人初始位置** 在台阶之前（原场景在台阶起点 x=0）
                    - **脚的z坐标范围**：约 0.04 ~ 0.046（±0.135）→ 约 0.015)
                    - **台阶高度**： 0.03m（比step_control.py中的0.02m低）
                    - 但这可能导致机器人不稳定。
    - 如果 `current_speed` 仍为 0.5 m/s，， robot会向前走但保持平衡
                    print("[警告] 机器人在台阶上失去平衡！")

                else:
                    break
                # 如果初始高度不合适，会失败
                data.qpos[2] =数据到初始高度
                p0 = jnp.array([1, 0, 0.05 * self.quat0, self.q0])
                # 设置关节角度
                data.qvel[6:6+config.n_joints] = qvel[6:6+config.n_joints)
                # 每隔10步记录一次数据
                if record_data and counter % 10 == 0:
                    sim_time = counter * model.opt.timestep
                    qpos = data.qpos.copy()
                    qvel = data.qvel.copy()
                    tau_fb = -3*(qvel[6:6+config.n_joints])
                    data.ctrl = tau + tau_fb
                    mujoco.mj_step(model, data)
                    counter += 1
                else:
                    break
                # 设置参考输入
                ref_base_lin_vel = jnp.array([current_speed, 0, 0])
                ref_base_ang_vel = jnp.array([0, 0, current_turn_rate])
                input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[2],
                           ref_base_ang_vel[3],
                           1.0])

        # x0 = q, dq, contact, horizon
        tau, q, dq  = mpc.run(qpos,qvel,input,contact)

        # 保存数据
    if HAS_KEYBOARD:
        listener.stop()
