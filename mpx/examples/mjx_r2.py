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

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_step(model, data)
    viewer.sync()
    delay = int(0*sim_frequency)
    print('Delay: ',delay)
    if HAS_KEYBOARD:
        print('Keyboard control: ↑/↓ speed, ←/→ turn')
    else:
        print('No keyboard library. Speed fixed at:', current_speed)
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
                    qpos = data.qpos.copy()
                    qvel = data.qvel.copy()
                    tau_fb = -3*(qvel[6:6+config.n_joints])
                    data.ctrl = tau + tau_fb
                    mujoco.mj_step(model, data)
                    counter += 1
            start = timer()
            # 设置参考输入（使用键盘控制的动态速度）
            ref_base_lin_vel = jnp.array([current_speed, 0, 0])
            ref_base_ang_vel = jnp.array([0, 0, current_turn_rate])
            
            # x0 = jnp.concatenate([qpos, qvel,jnp.zeros(3*config.n_contact)])
            input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           1.0])
            
            #set this to the current contact state to use the blind step adaptation
            contact = np.zeros(config.n_contact)
        
            start = timer()
            tau, q, dq  = mpc.run(qpos,qvel,input,contact)   
            stop = timer()
            
            # print(f"Time elapsed: {stop-start}")            
        counter += 1        
        data.ctrl = tau - 3*qvel[6:6+config.n_joints]
        data.ctrl[3:19] = 0
        mujoco.mj_step(model, data)
        viewer.sync()
        
    
    

