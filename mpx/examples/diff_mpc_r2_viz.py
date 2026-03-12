"""
R2 机器人 + 可微 MPC (IFT) 可视化仿真

本脚本将可微 MPC 求解器（带 IFT 梯度回传）集成到 MuJoCo 仿真环境中，
通过 mujoco.viewer 进行实时可视化，并支持键盘控制机器人行走。

功能:
    - MuJoCo 实时仿真 + 可微 MPC 控制
    - 键盘控制（↑/↓ 调速，←/→ 转向）
    - 在线梯度计算（可选，按 G 键触发）

使用方式:
    conda run -n mpx_env python mpx/examples/diff_mpc_r2_viz.py

作者: 王甫12138
日期: 2026-03-04
"""

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjx
from functools import partial
from timeit import default_timer as timer

# 项目模块
import mpx.config.config_r2 as config
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mpx.utils.mpc_utils as mpc_utils
from mpx.utils.differentiable_mpc import make_differentiable_mpc

# JAX 配置
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# 尝试导入 pynput 键盘库
try:
    from pynput import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Warning: 'pynput' library not found. Install with: pip install pynput")


def main():
    print("=" * 60)
    print("R2 机器人 + 可微 MPC (IFT) 可视化仿真")
    print("=" * 60)

    # ================================================================
    # 1. 加载 MuJoCo 模型
    # ================================================================
    print("\n[1/4] 加载 MuJoCo 模型...")
    # 仿真用模型（带场景）
    sim_model = mujoco.MjModel.from_xml_path(
        os.path.join(dir_path, '..', 'data', 'r2-1024', 'mjcf', 'scene.xml')
    )
    sim_data = mujoco.MjData(sim_model)
    sim_frequency = 500.0
    sim_model.opt.timestep = 1 / sim_frequency

    # MPC 内部用模型（纯机器人，用于 MJX 加速）
    mpc_model = mujoco.MjModel.from_xml_path(config.model_path)
    mpc_data = mujoco.MjData(mpc_model)
    mujoco.mj_fwdPosition(mpc_model, mpc_data)
    robot_mass = mpc_data.qM[0]
    mjx_model = mjx.put_model(mpc_model)

    print(f"  仿真模型: nq={sim_model.nq}, nv={sim_model.nv}")
    print(f"  MPC模型: nq={mpc_model.nq}, nv={mpc_model.nv}, mass={robot_mass:.1f}kg")

    # 获取接触点和刚体 ID
    # MJX 用（动力学函数内部）
    contact_id_mjx = []
    for name in config.contact_frame:
        contact_id_mjx.append(
            mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        )
    body_id = []
    for name in config.body_name:
        body_id.append(
            mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
        )
    # FK 用（正运动学取足端位置，必须用 mpc_model 的 ID，不能用 sim_model 的！）
    contact_id_fk = []
    for name in config.contact_frame:
        contact_id_fk.append(
            mujoco.mj_name2id(mpc_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        )

    # ================================================================
    # 2. 构建可微 MPC 求解器
    # ================================================================
    print("\n[2/4] 构建可微 MPC 求解器...")

    _cost = partial(config.cost, config.n_joints, config.n_contact, config.N)
    _hessian_approx = partial(
        config.hessian_approx, config.n_joints, config.n_contact
    )
    _dynamics = partial(
        config.dynamics, mpc_model, mjx_model, contact_id_mjx, body_id,
        config.n_joints, config.dt
    )

    diff_mpc = make_differentiable_mpc(
        cost=_cost,
        dynamics=_dynamics,
        hessian_approx=_hessian_approx,
        limited_memory=False,
        grad_clip_norm=10.0,
        tikhonov_eps=1e-6,
    )

    # 参考轨迹生成器和定时器
    _ref_gen = jax.jit(partial(
        mpc_utils.reference_generator,
        config.use_terrain_estimation, config.N, config.dt,
        config.n_joints, config.n_contact, robot_mass,
        config.p_legs0, config.q0,
    ))
    _timer_run = jax.jit(mpc_utils.timer_run)

    # 轨迹更新辅助函数
    shift = int(1 / (config.dt * config.mpc_frequency))

    @partial(jax.jit, static_argnums=(0, 1))
    def update_and_extract(n_joints, shift_val, U, X, V, x0, X0, U0):
        def safe_update():
            new_U0 = jnp.concatenate([U[shift_val:], jnp.tile(U[-1:], (shift_val, 1))])
            new_X0 = jnp.concatenate([X[shift_val:], jnp.tile(X[-1:], (shift_val, 1))])
            new_V0 = jnp.concatenate([V[shift_val:], jnp.tile(V[-1:], (shift_val, 1))])
            tau = U[0, :n_joints]
            q = X[0, 7:n_joints + 7]
            dq = X[1, 13 + n_joints:2 * n_joints + 13]
            return new_U0, new_X0, new_V0, tau, q, dq

        def unsafe_update():
            new_U0 = jnp.tile(config.u_ref, (config.N, 1))
            new_X0 = jnp.tile(x0, (config.N + 1, 1))
            new_V0 = jnp.zeros((config.N + 1, config.n))
            tau = U0[1, :n_joints]
            q = X0[1, 7:n_joints + 7]
            dq = X0[1, 13 + n_joints:2 * n_joints + 13]
            return new_U0, new_X0, new_V0, tau, q, dq

        return jax.lax.cond(jnp.isnan(U[0, 0]), unsafe_update, safe_update)

    _update_and_extract = jax.jit(
        partial(update_and_extract, config.n_joints, shift)
    )

    print(f"  状态维度 n={config.n}, 控制维度 m={config.m}")
    print(f"  预测步数 N={config.N}, 时间步 dt={config.dt}")
    print(f"  仿真频率 {sim_frequency}Hz, MPC频率 {config.mpc_frequency}Hz")
    print("  可微 MPC 创建成功 ✅")

    # ================================================================
    # 3. 初始化状态
    # ================================================================
    print("\n[3/4] 初始化状态...")

    # 设置仿真初始状态
    sim_data.qpos = np.concatenate([
        np.array(config.p0), np.array(config.quat0), np.array(config.q0)
    ])
    mujoco.mj_step(sim_model, sim_data)

    # MPC 内部状态
    W = config.W
    contact_time = config.timer_t.copy()
    liftoff = config.p_legs0.copy()
    clearence_speed = 0.4

    # 初始 foot positions（通过正运动学获取）
    mpc_data.qpos = sim_data.qpos.copy()
    mujoco.mj_kinematics(mpc_model, mpc_data)
    foot_op = np.array([
        mpc_data.geom_xpos[contact_id_fk[i]]
        for i in range(config.n_contact)
    ]).flatten()

    # 初始状态向量
    x0 = jnp.concatenate([
        jnp.array(sim_data.qpos), jnp.array(sim_data.qvel),
        jnp.array(foot_op)
    ])

    # 轨迹暖启动
    U0 = jnp.tile(config.u_ref, (config.N, 1))
    X0 = jnp.tile(x0, (config.N + 1, 1))
    V0 = jnp.zeros((config.N + 1, config.n))

    tau = jnp.zeros(config.n_joints)

    print(f"  初始状态 x0 形状: {x0.shape}")
    print(f"  权重矩阵 W 形状: {W.shape}")
    print("  初始化完成 ✅")

    # ================================================================
    # 4. 键盘控制
    # ================================================================
    current_speed = 0.5
    current_turn_rate = 0.0
    compute_grad_flag = False
    key_states = {'up': False, 'down': False, 'left': False, 'right': False}

    def on_press(key):
        nonlocal compute_grad_flag
        try:
            if key == keyboard.Key.up:
                key_states['up'] = True
            elif key == keyboard.Key.down:
                key_states['down'] = True
            elif key == keyboard.Key.left:
                key_states['left'] = True
            elif key == keyboard.Key.right:
                key_states['right'] = True
            elif hasattr(key, 'char') and key.char == 'g':
                compute_grad_flag = True
                print("  [G] 触发一次梯度计算...")
        except Exception:
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
        except Exception:
            pass

    if HAS_KEYBOARD:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    # ================================================================
    # 5. 主仿真循环
    # ================================================================
    print("\n[4/4] 启动可视化仿真...")
    print("=" * 60)
    if HAS_KEYBOARD:
        print("  键盘控制: ↑/↓ 调速, ←/→ 转向, G 触发梯度计算")
    else:
        print(f"  无键盘库，速度固定: {current_speed} m/s")
    print("=" * 60)

    delay = int(0 * sim_frequency)
    counter = 0
    last_speed = current_speed
    last_turn = current_turn_rate
    mpc_solve_count = 0

    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        mujoco.mj_step(sim_model, sim_data)
        viewer.sync()

        while viewer.is_running():
            # --- 键盘输入处理 ---
            if HAS_KEYBOARD:
                if key_states['up']:
                    current_speed += 0.01
                elif key_states['down']:
                    current_speed -= 0.01
                if key_states['left']:
                    current_turn_rate += 0.01
                elif key_states['right']:
                    current_turn_rate -= 0.01
                if abs(current_speed - last_speed) > 0.1:
                    print(f"  Speed: {current_speed:.2f} m/s")
                    last_speed = current_speed
                if abs(current_turn_rate - last_turn) > 0.1:
                    print(f"  Turn: {current_turn_rate:.2f} rad/s")
                    last_turn = current_turn_rate

            qpos = sim_data.qpos.copy()
            qvel = sim_data.qvel.copy()

            # --- 每隔 (sim_freq / mpc_freq) 步执行一次 MPC ---
            if counter % int(sim_frequency / config.mpc_frequency) == 0 or counter == 0:

                # 模拟 MPC 计算延迟
                if counter != 0:
                    for i in range(delay):
                        qpos = sim_data.qpos.copy()
                        qvel = sim_data.qvel.copy()
                        tau_fb = -3 * qvel[6:6 + config.n_joints]
                        sim_data.ctrl = np.array(tau) + tau_fb
                        mujoco.mj_step(sim_model, sim_data)
                        counter += 1

                # 获取足端位置
                mpc_data.qpos = qpos
                mujoco.mj_kinematics(mpc_model, mpc_data)
                foot_op = np.array([
                    mpc_data.geom_xpos[contact_id_fk[i]]
                    for i in range(config.n_contact)
                ]).flatten()

                # 构建状态向量
                x0 = jnp.concatenate([
                    jnp.array(qpos), jnp.array(qvel), jnp.array(foot_op)
                ])

                # 构建参考输入
                ref_input = jnp.array([
                    current_speed, 0, 0,
                    0, 0, current_turn_rate,
                    config.robot_height
                ])
                contact = jnp.zeros(config.n_contact)

                # 更新步态计时器
                des_contact, contact_time = _timer_run(
                    config.duty_factor, config.step_freq,
                    contact_time, 1 / config.mpc_frequency
                )

                # 生成参考轨迹
                reference, parameter, liftoff = _ref_gen(
                    contact_time, x0, jnp.array(foot_op), ref_input,
                    config.duty_factor, config.step_freq, config.step_height,
                    liftoff, contact, clearence_speed,
                )

                # ---- 可微 MPC 前向求解 ----
                t_start = timer()
                X, U, V = diff_mpc(reference, parameter, W, x0, X0, U0, V0)
                t_solve = timer() - t_start

                # 轨迹暖启动更新
                U0, X0, V0, tau_temp, q_temp, dq_temp = _update_and_extract(
                    U, X, V, x0, X0, U0
                )
                tau = np.clip(
                    np.array(tau_temp),
                    config.min_torque, config.max_torque
                )

                mpc_solve_count += 1
                if mpc_solve_count <= 3 or mpc_solve_count % 50 == 0:
                    print(f"  MPC #{mpc_solve_count}: 耗时 {t_solve:.3f}s, "
                          f"|u[0]|={np.linalg.norm(tau):.2f}")

                # ---- 可选：梯度计算 ----
                if compute_grad_flag:
                    compute_grad_flag = False
                    print("  计算 IFT 梯度中...")
                    t_start = timer()

                    def loss_fn(W_var):
                        X_g, U_g, V_g = diff_mpc(
                            reference, parameter, W_var, x0, X0, U0, V0
                        )
                        return (jnp.sum(X_g[-1, :3] ** 2)
                                + 1e-4 * jnp.sum(U_g[:, :config.n_joints] ** 2))

                    grad_W = jax.grad(loss_fn)(W)
                    t_grad = timer() - t_start

                    has_nan = jnp.any(jnp.isnan(grad_W))
                    max_grad = jnp.max(jnp.abs(grad_W))
                    print(f"  ✅ 梯度完成: {t_grad:.2f}s, "
                          f"NaN={has_nan}, max|grad|={max_grad:.4e}")

            # --- 施加控制力矩 ---
            counter += 1
            sim_data.ctrl = np.array(tau) - 3 * qvel[6:6 + config.n_joints]
            # 手臂关节置零（与 mjx_r2.py 一致）
            sim_data.ctrl[3:19] = 0
            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()

    print("\n仿真结束。")


if __name__ == "__main__":
    main()
