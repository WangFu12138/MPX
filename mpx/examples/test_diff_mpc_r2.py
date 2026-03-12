"""
R2 机器人 + 可微 MPC (IFT) 集成测试

本脚本验证 differentiable_mpc.py 在真实 R2 机器人配置（99维状态空间）下
是否能够正常完成前向求解和梯度回传。

无需 GUI，无需键盘，纯命令行运行。

使用方式:
    conda run -n mpx_env python mpx/examples/test_diff_mpc_r2.py

作者: 王甫12138
日期: 2026-03-04
"""

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))

import jax
import jax.numpy as jnp
import mujoco
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


def main():
    print("=" * 60)
    print("R2 机器人 + 可微 MPC (IFT) 集成测试")
    print("=" * 60)

    # ---- Step 1: 加载 MuJoCo 模型 ----
    print("\n[1/5] 加载 MuJoCo 模型...")
    model = mujoco.MjModel.from_xml_path(config.model_path)
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    robot_mass = data.qM[0]
    mjx_model = mjx.put_model(model)
    print(f"  模型加载成功: nq={model.nq}, nv={model.nv}, mass={robot_mass:.1f}kg")

    # ---- Step 2: 获取接触点和刚体 ID ----
    contact_id = []
    for name in config.contact_frame:
        contact_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name))
    body_id = []
    for name in config.body_name:
        body_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name))

    # ---- Step 3: 构建函数（与 MPCControllerWrapper 一致的绑定方式）----
    print("\n[2/5] 构建代价/动力学函数...")
    _cost = partial(config.cost, config.n_joints, config.n_contact, config.N)
    _hessian_approx = partial(config.hessian_approx, config.n_joints, config.n_contact)
    _dynamics = partial(
        config.dynamics, model, mjx_model, contact_id, body_id,
        config.n_joints, config.dt
    )

    print(f"  状态维度 n={config.n}, 控制维度 m={config.m}")
    print(f"  预测步数 N={config.N}, 时间步 dt={config.dt}")

    # ---- Step 4: 创建可微 MPC 求解器 ----
    print("\n[3/5] 创建可微 MPC 求解器 (IFT)...")
    diff_mpc = make_differentiable_mpc(
        cost=_cost,
        dynamics=_dynamics,
        hessian_approx=_hessian_approx,
        limited_memory=False,  # 使用 GPU 并行版本
        grad_clip_norm=10.0,
        tikhonov_eps=1e-6,
    )
    print("  可微 MPC 创建成功")

    # ---- Step 5: 构建初始状态与参考轨迹 ----
    print("\n[4/5] 构建初始状态与参考轨迹...")

    # 初始状态
    x0 = jnp.concatenate([
        config.p0, config.quat0, config.q0,
        jnp.zeros(6 + config.n_joints),
        config.p_legs0,
    ])

    # 初始轨迹猜测
    X_in = jnp.tile(x0, (config.N + 1, 1))
    U_in = jnp.tile(config.u_ref, (config.N, 1))
    V_in = jnp.zeros((config.N + 1, config.n))

    # 参考轨迹生成
    _ref_gen = jax.jit(partial(
        mpc_utils.reference_generator,
        config.use_terrain_estimation, config.N, config.dt,
        config.n_joints, config.n_contact, robot_mass,
        config.p_legs0, config.q0,
    ))
    _timer_run = jax.jit(mpc_utils.timer_run)

    # 参考输入（前进 0.5 m/s）
    input_cmd = jnp.array([0.5, 0, 0, 0, 0, 0, 0.65])
    contact = jnp.zeros(config.n_contact)

    # 生成参考轨迹
    contact_time = config.timer_t
    liftoff = config.p_legs0.copy()
    contact, contact_time = _timer_run(
        config.duty_factor, config.step_freq, contact_time, config.dt,
    )
    reference, parameter, liftoff_new = _ref_gen(
        contact_time, x0, liftoff, input_cmd,
        config.duty_factor, config.step_freq, config.step_height,
        liftoff, contact, 0.4,
    )
    W = config.W

    print(f"  x0 形状: {x0.shape}")
    print(f"  reference 形状: {reference.shape}")
    print(f"  parameter 形状: {parameter.shape}")
    print(f"  W 形状: {W.shape}")

    # ---- Step 6: 前向求解测试 ----
    print("\n[5/5] 执行前向求解 + 梯度回传...")

    # 前向求解（第一次会触发 JIT 编译）
    print("  前向求解中 (首次 JIT 编译可能需要 1-2 分钟)...")
    t_start = timer()
    X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
    t_fwd = timer() - t_start
    print(f"  ✅ 前向求解完成 (耗时: {t_fwd:.2f}s)")
    print(f"     X 形状: {X.shape}, 有限: {jnp.all(jnp.isfinite(X))}")
    print(f"     U 形状: {U.shape}, 有限: {jnp.all(jnp.isfinite(U))}")
    print(f"     首个控制 u[0][:5] = {U[0, :5]}")

    # 梯度回传测试
    print("\n  计算梯度 (IFT 反向传播)...")

    def loss_fn(W):
        X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
        # 简单的任务损失：最终位置偏差 + 力矩消耗
        return jnp.sum(X[-1, :3] ** 2) + 1e-4 * jnp.sum(U[:, :config.n_joints] ** 2)

    t_start = timer()
    grad_W = jax.grad(loss_fn)(W)
    t_bwd = timer() - t_start

    has_nan = jnp.any(jnp.isnan(grad_W))
    has_inf = jnp.any(jnp.isinf(grad_W))
    max_grad = jnp.max(jnp.abs(grad_W))

    print(f"  ✅ 梯度计算完成 (耗时: {t_bwd:.2f}s)")
    print(f"     NaN: {has_nan}, Inf: {has_inf}, Max|grad|: {max_grad:.4e}")
    print(f"     grad_W 形状: {grad_W.shape}")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("集成测试结果汇总")
    print("=" * 60)

    fwd_ok = jnp.all(jnp.isfinite(X)) and jnp.all(jnp.isfinite(U))
    bwd_ok = not has_nan and not has_inf

    print(f"  前向求解: {'✅ PASS' if fwd_ok else '❌ FAIL'}")
    print(f"  梯度回传: {'✅ PASS' if bwd_ok else '❌ FAIL'}")
    print(f"  梯度裁剪: {'✅ 已生效 (max≤10)' if max_grad <= 10.0 else '⚠️ 超过阈值'}")

    if fwd_ok and bwd_ok:
        print("\n🎉 可微 MPC (IFT) 在 R2 机器人上完全正常工作！")
    else:
        print("\n⚠️ 存在问题，请检查上述输出。")

    return fwd_ok and bwd_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
