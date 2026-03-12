import jax
import jax.numpy as jnp
from jax import custom_vjp

# 1. 定义 MPC 的代价函数
# 使用一个有耦合的代价函数，让最优解依赖于 Q
# f(u, Q) = 0.5 * (u * Q - 1)^2  -> 最优解在 u* = 1/Q
def mpc_cost(u, Q):
    return 0.5 * jnp.sum(jnp.square(u * Q - 1.0))

# 2. 定义前向传播：实际的迭代求解过程
def mpc_solve_fwd(Q):
    # 这里模拟一个不固定次数的迭代过程 (比如 iLQR)
    # 假设通过某种迭代找到了最优解 u_star
    # 对于 f(u, Q) = 0.5 * u^2 * Q，最优解是 u* = 0
    # 但我们需要让 u* 与 Q 有依赖关系来测试梯度
    # 让我们用一个更真实的例子：f(u, Q) = 0.5 * (u - 1/Q)^2 * Q
    # 这时最优解是 u* = 1/Q

    # 模拟迭代求解的结果：u* = 1 / Q
    # 使用 stop_gradient 阻止梯度直接回流，强制走自定义反向传播
    u_star = jax.lax.stop_gradient(1.0 / (Q + 0.1))  # +0.1 避免除零

    # 返回最优解，并保存用于反向传播的上下文 (u_star, Q)
    return u_star, (u_star, Q)

# 3. 定义反向传播：利用隐函数定理跳过迭代
def mpc_solve_bwd(res, g):
    """
    res: 前向传播保存的 (u_star, Q)
    g: 上层传回来的梯度 dLoss/du*
    """
    u_star, Q = res
    
    # 隐函数定理核心公式： dLoss/dQ = g * [du*/dQ]
    # 根据我们推导的公式： du*/dQ = - [H_uu]^-1 * [H_uQ]
    
    # 计算 Hessian 矩阵 (u方向的二阶导)
    hessian_uu = jax.hessian(mpc_cost, argnums=0)(u_star, Q)
    
    # 计算交叉偏导 (u 和 Q 的混合导数)
    hessian_uQ = jax.jacobian(jax.grad(mpc_cost, argnums=0), argnums=1)(u_star, Q)
    
    # 解线性方程组: H_uu * x = g  => x = [H_uu]^-1 * g
    # 这是 IFT 的标准实现方式，避免直接求逆矩阵
    inv_h_g = jnp.linalg.solve(hessian_uu, g)
    
    # 最终梯度梯度 dLoss/dQ = - inv_h_g * Hessian_uQ
    grad_Q = - jnp.dot(inv_h_g, hessian_uQ)
    
    return (grad_Q,)

# 4. 将两者绑定到自定义函数上
@custom_vjp
def differentiable_mpc(Q):
    u_star, _ = mpc_solve_fwd(Q)
    return u_star

# 绑定前向和反向逻辑
differentiable_mpc.defvjp(mpc_solve_fwd, mpc_solve_bwd)

# --- 使用示例 ---

# 简单的视觉网络：从输入数据生成权重 Q
def vision_net(params, input_data):
    """简单的线性网络：input -> Q（权重矩阵）"""
    W = params['W']
    b = params['b']
    # 确保输出为正（作为权重）
    Q = jnp.dot(input_data, W) + b
    return jnp.maximum(Q, 0.1)  # 避免负权重

def total_system_loss(vision_params, input_data):
    # 视觉网络输出 Q
    Q = vision_net(vision_params, input_data)
    # 通过"可微 MPC 层"
    u_star = differentiable_mpc(Q)
    # 计算损失
    return jnp.sum(jnp.square(u_star))

# 初始化参数和数据
key = jax.random.PRNGKey(0)
input_dim = 10
output_dim = 5

# 随机初始化网络参数
key, W_key, b_key = jax.random.split(key, 3)
params = {
    'W': jax.random.normal(W_key, (input_dim, output_dim)),
    'b': jax.random.normal(b_key, (output_dim,))
}

# 随机输入数据
data = jax.random.normal(key, (input_dim,))

# 现在你可以直接对 vision_params 求导了！
# JAX 会自动调用 mpc_solve_bwd，完全跳过前向的迭代细节。

# 方案：使用 value_and_grad 同时获取损失值和梯度
loss_and_grad = jax.value_and_grad(total_system_loss)

print("计算损失和梯度...")
loss_value, grads = loss_and_grad(params, data)
print("计算完成！")
print(f"\n损失值: {loss_value:.6f}")
print(f"\n梯度 dW:\n{grads['W']}")
print(f"\n梯度 db:\n{grads['b']}")

# 验证梯度是否正确
print("\n--- 梯度验证 ---")
# 数值梯度（有限差分法）

def manual_grad(params, data, eps=1e-5):
    grad_W = jnp.zeros_like(params['W'])
    for i in range(params['W'].shape[0]):
        for j in range(params['W'].shape[1]):
            params_plus = params.copy()
            params_plus['W'] = params['W'].at[i, j].add(eps)
            loss_plus = total_system_loss(params_plus, data)

            params_minus = params.copy()
            params_minus['W'] = params['W'].at[i, j].add(-eps)
            loss_minus = total_system_loss(params_minus, data)

            grad_W = grad_W.at[i, j].set((loss_plus - loss_minus) / (2 * eps))
    return {'W': grad_W}

# 只对第一个元素做数值验证（太慢了）
manual_g = manual_grad(params, data)
print(f"解析梯度 dW[0,0]: {grads['W'][0, 0]:.6f}")
print(f"数值梯度 dW[0,0]: {manual_g['W'][0, 0]:.6f}")
print(f"误差: {abs(grads['W'][0, 0] - manual_g['W'][0, 0]):.8f}")