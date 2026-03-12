"""
可微 MPC (IFT 版) 单元测试

验证 differentiable_mpc.py 中的各组件：
1. weight_transform: 对数空间权重转换
2. clip_gradient_elementwise: 梯度裁剪
3. make_differentiable_mpc: 端到端可微 MPC (IFT custom_vjp)
4. 大权重下的梯度稳定性

使用简化的 2D 质点追踪场景测试。

作者: 王甫12138
日期: 2026-03-04
"""

import sys
import os
# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from functools import partial

# 导入待测模块
from mpx.utils.differentiable_mpc import (
    weight_transform,
    safe_exp,
    clip_gradient_elementwise,
    clip_gradients,
    make_differentiable_mpc,
    DifferentiableMPCWrapper,
)


# ============== 测试辅助：简化的 2D 质点系统 ==============

def simple_dynamics(x, u, t, parameter=None):
    """简化的 2D 质点动力学: x_{t+1} = A x_t + B u_t"""
    dt = 0.1
    n = x.shape[0]
    # x = [px, py, vx, vy]
    A = jnp.eye(n)
    A = A.at[0, 2].set(dt)
    A = A.at[1, 3].set(dt)
    B = jnp.zeros((n, 2))
    B = B.at[2, 0].set(dt)
    B = B.at[3, 1].set(dt)
    return A @ x + B @ u


def simple_cost(W, reference, x, u, t):
    """简化的二次代价函数:
    J = 0.5 * (x - ref)^T @ W @ (x - ref) + 0.001 * u^T u
    """
    n = x.shape[0]
    W_mat = jnp.diag(W[:n])
    ref = reference[t]
    err = x - ref
    state_cost = 0.5 * err @ W_mat @ err
    ctrl_cost = 0.001 * jnp.sum(u[:2] ** 2)  # 只取前 2 维
    return state_cost + ctrl_cost


def simple_hessian_approx(W, reference, x, u, t):
    """简化的 Hessian 近似（解析）"""
    n = x.shape[0]
    m = u.shape[0]
    Q = jnp.diag(W[:n])
    R = 0.002 * jnp.eye(m)
    M = jnp.zeros((n, m))
    return Q, R, M


def generate_static_reference(T, n):
    """生成静态参考轨迹（目标点 [1, 1, 0, 0]）"""
    target = jnp.array([1.0, 1.0, 0.0, 0.0])
    return jnp.tile(target, (T + 1, 1))


# ============== 测试函数 ==============

def test_weight_transform():
    """测试对数空间权重转换"""
    print("\n" + "=" * 60)
    print("测试 1: weight_transform (对数空间权重)")
    print("=" * 60)

    # 正常值
    w_logits = jnp.array([0.0, 1.0, 2.0, -1.0])
    w = weight_transform(w_logits)
    expected = jnp.exp(w_logits)
    assert jnp.allclose(w, expected, atol=1e-6), f"不匹配: {w} vs {expected}"
    assert jnp.all(w > 0), "权重必须为正"
    print(f"  w_logits = {w_logits}")
    print(f"  W = {w}")
    print("  [PASS] 正常值转换正确")

    # 极端值（应被裁剪）
    w_extreme = jnp.array([100.0, -100.0])
    w_safe = weight_transform(w_extreme)
    assert jnp.all(jnp.isfinite(w_safe)), f"极端值产生非有限数: {w_safe}"
    print(f"  极端 w_logits = {w_extreme}")
    print(f"  安全 W = {w_safe}")
    print("  [PASS] 极端值安全处理")

    return True


def test_gradient_clipping():
    """测试梯度裁剪"""
    print("\n" + "=" * 60)
    print("测试 2: 梯度裁剪")
    print("=" * 60)

    # 逐元素裁剪
    grad = jnp.array([100.0, -200.0, 0.5, -0.3])
    clipped = clip_gradient_elementwise(grad, max_val=1.0)
    assert jnp.all(jnp.abs(clipped) <= 1.0), f"裁剪失败: {clipped}"
    print(f"  原始梯度: {grad}")
    print(f"  裁剪后:   {clipped}")
    print("  [PASS] 逐元素裁剪正确")

    # 全局范数裁剪
    grads = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
    clipped_tree = clip_gradients(grads, max_norm=5.0)
    total_norm_after = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(clipped_tree))
    )
    assert total_norm_after <= 5.0 + 1e-6, f"全局范数裁剪失败: {total_norm_after}"
    print(f"  原始范数: {jnp.sqrt(10**2 + 20**2 + 30**2):.4f}")
    print(f"  裁剪范数: {total_norm_after:.4f}")
    print("  [PASS] 全局范数裁剪正确")

    return True


def test_differentiable_mpc_basic():
    """测试可微 MPC 的基本前向传播和梯度计算"""
    print("\n" + "=" * 60)
    print("测试 3: 可微 MPC 基本功能 (IFT custom_vjp)")
    print("=" * 60)

    jax.config.update("jax_enable_x64", True)

    n = 4   # 状态维度
    m = 2   # 控制维度
    T = 10  # 时域步数

    # 初始化
    x0 = jnp.zeros(n)
    reference = generate_static_reference(T, n)
    parameter = jnp.zeros(1)  # 无额外参数
    W = jnp.array([10.0, 10.0, 1.0, 1.0])
    X_in = jnp.tile(x0, (T + 1, 1))
    U_in = jnp.zeros((T, m))
    V_in = jnp.zeros((T + 1, n))

    # 创建可微 MPC
    diff_mpc = make_differentiable_mpc(
        cost=simple_cost,
        dynamics=simple_dynamics,
        hessian_approx=simple_hessian_approx,
        limited_memory=True,
        grad_clip_norm=10.0,
        tikhonov_eps=1e-6,
    )

    # 前向传播
    print("  执行前向传播...")
    X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
    print(f"  X 形状: {X.shape}, U 形状: {U.shape}, V 形状: {V.shape}")
    assert X.shape == (T + 1, n), f"X 形状不对: {X.shape}"
    assert U.shape == (T, m), f"U 形状不对: {U.shape}"
    assert jnp.all(jnp.isfinite(X)), "X 包含非有限数"
    assert jnp.all(jnp.isfinite(U)), "U 包含非有限数"
    print("  [PASS] 前向传播形状和有限性正确")

    # 梯度计算
    print("  计算梯度 (IFT 反向传播)...")

    def loss_fn(W):
        X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
        return jnp.sum(X[-1, :2] ** 2) + 0.001 * jnp.sum(U ** 2)

    grad_W = jax.grad(loss_fn)(W)
    print(f"  W = {W}")
    print(f"  dL/dW = {grad_W}")
    assert grad_W.shape == W.shape, f"梯度形状不对: {grad_W.shape}"
    assert jnp.all(jnp.isfinite(grad_W)), f"梯度包含非有限数: {grad_W}"
    print("  [PASS] IFT 梯度计算成功，无 NaN/Inf")

    return True


def test_gradient_stability_large_weights():
    """测试大权重下的梯度稳定性"""
    print("\n" + "=" * 60)
    print("测试 4: 大权重梯度稳定性")
    print("=" * 60)

    jax.config.update("jax_enable_x64", True)

    n, m, T = 4, 2, 10

    x0 = jnp.zeros(n)
    reference = generate_static_reference(T, n)
    parameter = jnp.zeros(1)
    X_in = jnp.tile(x0, (T + 1, 1))
    U_in = jnp.zeros((T, m))
    V_in = jnp.zeros((T + 1, n))

    diff_mpc = make_differentiable_mpc(
        cost=simple_cost,
        dynamics=simple_dynamics,
        hessian_approx=simple_hessian_approx,
        limited_memory=True,
        grad_clip_norm=10.0,
        tikhonov_eps=1e-4,
    )

    test_cases = [
        ("正常权重", jnp.array([10.0, 10.0, 1.0, 1.0])),
        ("大权重", jnp.array([1000.0, 1000.0, 100.0, 100.0])),
        ("极大权重", jnp.array([1e5, 1e5, 1e4, 1e4])),
        ("不均衡权重", jnp.array([1e4, 0.1, 1e3, 0.01])),
    ]

    def loss_fn(W):
        X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
        return jnp.sum(X[-1, :2] ** 2)

    all_pass = True
    for name, W in test_cases:
        try:
            grad_W = jax.grad(loss_fn)(W)
            has_nan = jnp.any(jnp.isnan(grad_W))
            has_inf = jnp.any(jnp.isinf(grad_W))
            max_val = jnp.max(jnp.abs(grad_W))

            if has_nan or has_inf:
                print(f"  [FAIL] {name}: NaN={has_nan}, Inf={has_inf}")
                all_pass = False
            elif max_val > 10.0:
                print(f"  [WARN] {name}: 梯度已被裁剪, max={max_val:.4e}")
            else:
                print(f"  [OK]   {name}: max_grad={max_val:.4e}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_pass = False

    if all_pass:
        print("  [PASS] 所有权重配置下梯度稳定")
    return all_pass


def test_log_space_learning():
    """测试对数空间参数的端到端学习"""
    print("\n" + "=" * 60)
    print("测试 5: 对数空间端到端学习")
    print("=" * 60)

    jax.config.update("jax_enable_x64", True)

    n, m, T = 4, 2, 10

    x0 = jnp.zeros(n)
    reference = generate_static_reference(T, n)
    parameter = jnp.zeros(1)
    X_in = jnp.tile(x0, (T + 1, 1))
    U_in = jnp.zeros((T, m))
    V_in = jnp.zeros((T + 1, n))

    base_W = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))

    wrapper = DifferentiableMPCWrapper(
        cost=simple_cost,
        dynamics=simple_dynamics,
        hessian_approx=simple_hessian_approx,
        limited_memory=True,
        base_W=base_W,
        grad_clip_norm=10.0,
    )

    w_logits = wrapper.get_init_logits()
    print(f"  初始 w_logits = {w_logits}")

    def loss_fn(w_logits):
        # 直接用 weight_transform 得到 1D 权重向量（匹配 simple_cost 的接口）
        W = weight_transform(w_logits)
        X, U, V = wrapper.solve(reference, parameter, W, x0, X_in, U_in, V_in)
        return jnp.sum(X[-1, :2] ** 2)

    initial_loss = loss_fn(w_logits)
    print(f"  初始损失 = {initial_loss:.6f}")

    # 执行几步梯度下降
    lr = 0.01
    for i in range(3):
        g = jax.grad(loss_fn)(w_logits)
        w_logits = w_logits - lr * g
        loss = loss_fn(w_logits)
        print(f"  迭代 {i+1}: loss = {loss:.6f}, |grad| = {jnp.max(jnp.abs(g)):.4e}")

    final_loss = loss_fn(w_logits)
    print(f"  最终损失 = {final_loss:.6f}")
    print(f"  最终 w_logits = {w_logits}")

    if jnp.isfinite(final_loss):
        print("  [PASS] 对数空间学习正常运行")
        return True
    else:
        print("  [FAIL] 损失非有限")
        return False


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("可微 MPC (IFT 版) 验证测试")
    print("=" * 60)

    results = []
    results.append(("权重转换", test_weight_transform()))
    results.append(("梯度裁剪", test_gradient_clipping()))
    results.append(("IFT 基本功能", test_differentiable_mpc_basic()))
    results.append(("大权重稳定性", test_gradient_stability_large_weights()))
    results.append(("对数空间学习", test_log_space_learning()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "PASS ✅" if result else "FAIL ❌"
        print(f"  {name}: [{status}]")

    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n✅ 所有测试通过！可微 MPC (IFT) 底座已就绪。")
    else:
        print("\n⚠️ 部分测试未通过，请检查上述输出。")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
