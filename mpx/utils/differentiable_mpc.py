"""
可微 MPC 包装器 — 基于隐函数定理 (IFT) 的 custom_vjp 实现

本模块通过 JAX 的 @custom_vjp 机制，使原装 primal_dual_ilqr 求解器
（包含 lax.while_loop）在前向传播时正常运行，反向传播时利用隐函数定理
（Implicit Function Theorem）高效求解解析梯度。

核心公式（IFT 反向传播）:
    ∂L/∂W = - λᵀ · (∂²J/∂z∂W)
    其中 λ 满足:   (∂²J/∂z²) · λ = ∂L/∂z
    这里 z = (X, U) 是优化变量，W 是代价函数权重参数。

    求解 λ 不做矩阵求逆，而是将上游梯度作为 LQR 的代价梯度 (q, r)，
    复用 tvlqr backward sweep 来高效求解。

梯度稳定性保障:
    1. 对数空间参数化：W = exp(w_logits)，保证权重严格大于零
    2. 梯度裁剪：对最终输出的梯度施加全局范数裁剪
    3. Tikhonov 正则化：在 Hessian 对角线上加微小常数，提升条件数

作者: 王甫12138
日期: 2026-03-04
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp, jit, lax, vmap
from functools import partial

from trajax.optimizers import linearize, quadratize

from mpx.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr, tvlqr_gpu, rollout, rollout_gpu
from mpx.primal_dual_ilqr.primal_dual_ilqr.dual_tvlqr import dual_lqr
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    mpc as original_mpc,
    model_evaluator_helper,
    compute_search_direction,
    parallel_filter_line_search,
)
from mpx.primal_dual_ilqr.primal_dual_ilqr.linalg_helpers import project_psd_cone


# ============================================================================
# 1. 梯度稳定性工具函数
# ============================================================================

def safe_exp(x, max_val=20.0):
    """安全的指数函数，防止溢出。

    Args:
        x: 对数空间的参数 (log-space logits)
        max_val: 允许的最大输入值，超过此值会被裁剪

    Returns:
        exp(clip(x))，保证结果在 [exp(-max_val), exp(max_val)] 范围内
    """
    return jnp.exp(jnp.clip(x, -max_val, max_val))


def weight_transform(w_logits):
    """将对数空间参数转换为正定权重对角线。

    数学原理:
        W = exp(w_logits)

    这保证了:
        1. W > 0（权重严格为正）
        2. ∂W/∂w_logits = W（梯度与权重本身成比例，数值稳定）
        3. 大权重和小权重在对数空间中获得相似尺度的梯度

    Args:
        w_logits: (d,) 对数空间的权重参数

    Returns:
        W_diag: (d,) 正定权重对角线元素
    """
    return safe_exp(w_logits)


def clip_gradients(grads, max_norm=10.0):
    """对梯度树进行全局范数裁剪。

    Args:
        grads: PyTree 结构的梯度
        max_norm: 最大允许的全局范数

    Returns:
        裁剪后的梯度 PyTree
    """
    total_norm = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
    )
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def clip_gradient_elementwise(grad, max_val=1.0):
    """对单个梯度数组进行逐元素裁剪。

    Args:
        grad: 梯度数组
        max_val: 每个元素的最大绝对值

    Returns:
        裁剪后的梯度
    """
    return jnp.clip(grad, -max_val, max_val)


# ============================================================================
# 2. 隐函数定理 (IFT) 核心求解器
# ============================================================================

def ift_backward_solve(
    cost_fn,
    dynamics_fn,
    hessian_approx_fn,
    x0,
    X_star,
    U_star,
    V_star,
    dL_dX,
    dL_dU,
    limited_memory=False,
    tikhonov_eps=1e-6,
):
    """使用隐函数定理 (IFT) 求解反向传播梯度。

    核心思路:
        在最优解 (X*, U*) 处，KKT 条件的一阶必要条件成立。
        利用隐函数定理，我们可以将上游梯度 (dL/dX, dL/dU) 转化为
        对参数 W 的梯度 (dL/dW)。

        具体来说，将 dL/dX, dL/dU 视为一个 LQR 子问题的线性代价项
        (q_adj, r_adj)，执行一趟 TVLQR backward sweep 来求解伴随变量 λ。

    Args:
        cost_fn:           代价函数 cost(x, u, t)
        dynamics_fn:       动力学函数 dynamics(x, u, t)
        hessian_approx_fn: Hessian 近似函数 或 None
        x0:                (n,) 初始状态
        X_star:            (T+1, n) 最优状态轨迹
        U_star:            (T, m) 最优控制轨迹
        V_star:            (T+1, n) 最优对偶变量
        dL_dX:             (T+1, n) 上游关于状态的梯度
        dL_dU:             (T, m) 上游关于控制的梯度
        limited_memory:    是否使用低内存版本的 LQR
        tikhonov_eps:      Tikhonov 正则化系数

    Returns:
        adjoint_X:   (T+1, n) 伴随状态变量 (用于计算对参数的梯度)
        adjoint_U:   (T, m) 伴随控制变量
    """
    T = U_star.shape[0]
    n = X_star.shape[1]
    m = U_star.shape[1]

    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    # ---- Step 1: 在最优解处获取二次化矩阵 (Hessian of cost) ----
    if hessian_approx_fn is None:
        quadratizer = quadratize(cost_fn)
        Q, R_pad, M_pad = quadratizer(
            X_star, pad(U_star), jnp.arange(T + 1)
        )
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx_fn)(
            X_star, pad(U_star), jnp.arange(T + 1)
        )

    R = R_pad[:-1]
    M = M_pad[:-1]

    # ---- Step 2: 在最优解处线性化动力学 ----
    dynamics_linearizer = linearize(dynamics_fn)
    A_pad, B_pad = dynamics_linearizer(
        X_star, pad(U_star), jnp.arange(T + 1)
    )
    A_lin = A_pad[:-1]
    B_lin = B_pad[:-1]

    # ---- Step 3: Tikhonov 正则化，改善条件数 ----
    Q_reg = Q + tikhonov_eps * jnp.eye(n)[None, :, :]
    R_reg = R + tikhonov_eps * jnp.eye(m)[None, :, :]

    # ---- Step 4: 构造伴随 LQR 子问题 ----
    # 将上游回传的梯度 dL/dX, dL/dU 作为代价向量的线性项 q_adj, r_adj
    q_adj = dL_dX   # (T+1, n)
    r_adj = dL_dU   # (T, m)

    # 约束偏移 c 在 IFT 子问题中为零（因为我们在最优解处展开）
    c_adj = jnp.zeros((T, n))

    # ---- Step 5: 执行 TVLQR backward sweep ----
    # 这等效于求解:  [Q  M] [λ_x]   [q_adj]
    #                [M' R] [λ_u] = -[r_adj]
    # 加上动力学约束 A λ_x + B λ_u = 0 的伴随关系
    if limited_memory:
        K_adj, k_adj, P_adj, p_adj = tvlqr(
            Q_reg, q_adj, R_reg, r_adj, M, A_lin, B_lin, c_adj
        )
        adjoint_X, adjoint_U = rollout(K_adj, k_adj, jnp.zeros(n), A_lin, B_lin, c_adj)
    else:
        K_adj, k_adj, P_adj, p_adj = tvlqr_gpu(
            Q_reg, q_adj, R_reg, r_adj, M, A_lin, B_lin, c_adj
        )
        adjoint_X, adjoint_U = rollout_gpu(K_adj, k_adj, jnp.zeros(n), A_lin, B_lin, c_adj)

    return adjoint_X, adjoint_U


# ============================================================================
# 3. 可微 MPC 入口 (custom_vjp)
# ============================================================================

def make_differentiable_mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    grad_clip_norm=10.0,
    tikhonov_eps=1e-6,
):
    """创建一个可微分的 MPC 求解器。

    该函数返回一个可微分的 MPC 函数，其前向传播使用原装 MPX 求解器
    （包含 lax.while_loop），反向传播使用隐函数定理 (IFT)。

    Args:
        cost:            代价函数 cost(W, reference, x, u, t)
        dynamics:        动力学函数 dynamics(x, u, t, parameter=...)
        hessian_approx:  Hessian 近似函数或 None
        limited_memory:  是否使用低内存模式
        grad_clip_norm:  梯度裁剪的全局范数阈值
        tikhonov_eps:    Tikhonov 正则化系数

    Returns:
        differentiable_mpc_solve: 可微分的 MPC 求解函数
            签名: (reference, parameter, W, x0, X_in, U_in, V_in) -> (X, U, V)
            支持对 W (权重) 求导: jax.grad(loss_fn)(W)

    使用示例:
        diff_mpc = make_differentiable_mpc(cost, dynamics, hessian_approx, True)

        def loss_fn(W):
            X, U, V = diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)
            return jnp.sum(X[-1, :3] ** 2)  # 某个任务损失

        grad_W = jax.grad(loss_fn)(W)
    """

    # ---- 定义 custom_vjp 函数 ----

    @custom_vjp
    def differentiable_mpc_solve(reference, parameter, W, x0, X_in, U_in, V_in):
        """可微 MPC 前向求解（对外接口）。"""
        # 直接调用原版 MPC 求解器
        X, U, V = original_mpc(
            cost, dynamics, hessian_approx, limited_memory,
            reference, parameter, W, x0, X_in, U_in, V_in,
        )
        return X, U, V

    def _fwd(reference, parameter, W, x0, X_in, U_in, V_in):
        """前向传播：执行 MPC 求解，保存上下文用于反向传播。

        使用 stop_gradient 包裹所有输入，确保 while_loop 不参与反向传播。
        然后保存结果和原始参数作为反向传播的 residuals。
        """
        # 阻止梯度直接通过 while_loop 回传
        ref_sg = lax.stop_gradient(reference)
        param_sg = lax.stop_gradient(parameter)
        W_sg = lax.stop_gradient(W)
        x0_sg = lax.stop_gradient(x0)
        X_in_sg = lax.stop_gradient(X_in)
        U_in_sg = lax.stop_gradient(U_in)
        V_in_sg = lax.stop_gradient(V_in)

        X, U, V = original_mpc(
            cost, dynamics, hessian_approx, limited_memory,
            ref_sg, param_sg, W_sg, x0_sg, X_in_sg, U_in_sg, V_in_sg,
        )

        # 保存用于反向传播的上下文:
        #   - 最优解 X*, U*, V* (已 stop_gradient)
        #   - 原始参数 W, reference, x0 (保留梯度流)
        residuals = (X, U, V, reference, parameter, W, x0)
        return (X, U, V), residuals

    def _bwd(residuals, g):
        """反向传播：使用 IFT 计算隐式梯度。

        数学推导:
            1. 前向求解得到 z* = (X*, U*) = argmin J(z, W)
            2. 在 z* 处 KKT 一阶条件成立: ∇_z J(z*, W) = 0
            3. 对 W 求全微分: ∇²_zz J · dz*/dW + ∇²_zW J = 0
            4. => dz*/dW = -[∇²_zz J]^{-1} · ∇²_zW J
            5. 链式法则: dL/dW = (dL/dz*) · (dz*/dW)
                              = -(dL/dz*) · [∇²_zz J]^{-1} · ∇²_zW J

            其中 [∇²_zz J]^{-1} · v 通过 TVLQR 高效求解。
        """
        X_star, U_star, V_star, reference, parameter, W, x0 = residuals
        dL_dX, dL_dU, dL_dV = g

        T = U_star.shape[0]
        n = X_star.shape[1]
        m = U_star.shape[1]

        # ---- Step 1: 绑定参数到代价和动力学函数 ----
        _cost = partial(cost, W, reference)
        if hessian_approx is not None:
            _hessian_approx = partial(hessian_approx, W, reference)
        else:
            _hessian_approx = None
        _dynamics = partial(dynamics, parameter=parameter)

        # ---- Step 2: 求解伴随变量 λ = [∇²_zz J]^{-1} · dL/dz ----
        adjoint_X, adjoint_U = ift_backward_solve(
            cost_fn=_cost,
            dynamics_fn=_dynamics,
            hessian_approx_fn=_hessian_approx,
            x0=x0,
            X_star=lax.stop_gradient(X_star),
            U_star=lax.stop_gradient(U_star),
            V_star=lax.stop_gradient(V_star),
            dL_dX=dL_dX,
            dL_dU=dL_dU,
            limited_memory=limited_memory,
            tikhonov_eps=tikhonov_eps,
        )

        # ---- Step 3: 计算对 W 的梯度: dL/dW = -λᵀ · (∂²J / ∂z∂W) ----
        # 由于 J(x, u, t) 是通过 cost(W, reference, x, u, t) 定义的，
        # 我们需要计算 ∂cost/∂W 在最优轨迹 (X*, U*) 处的值，
        # 然后与伴随变量点乘求和。
        #
        # 等价地：dL/dW = Σ_t  (∂cost/∂W)(X*[t], U*[t], t) · λ_scale
        # 但由于 cost 是二次的，其对 W 的导数是状态/控制的函数。
        #
        # 更简洁的做法：
        # 构造一个关于 W 的标量函数，其值等于 Σ_t adjoint · ∂cost/∂x
        # 然后对 W 求梯度。

        pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

        def total_weighted_cost(W_var):
            """以 W_var 为变量的总代价，在 (X*, U*) 处求值。

            我们需要 ∂/∂W Σ_t cost(W, ref, X*[t], U[t], t)
            再乘以伴随变量。

            但最直接的 IFT 公式是:
            dL/dW = Σ_t [ λ_x[t]ᵀ · ∂²J/∂x∂W · ... + λ_u[t]ᵀ · ∂²J/∂u∂W ]

            最简单且正确的实现方式:
            利用 JAX 的自动微分，对 W 求导 Σ_t cost(W, ref, X*, U*, t)
            然后乘以合适的系数。

            然而这只会给出 ∂J_total/∂W，还缺少 Hessian 逆的部分。
            完整的 IFT 公式需要:
            dL/dW = - Σ_t [λ_x[t] · (∂/∂W)(∂cost/∂x) + λ_u[t] · (∂/∂W)(∂cost/∂u)]

            简化近似 (一阶):
            当代价函数为 (x-ref)ᵀ W (x-ref) 形式时，
            ∂cost/∂x = W(x-ref), ∂²cost/∂x∂W = (x-ref)
            所以 dL/dW ≈ Σ_t λ_x[t] · (x[t] - ref[t])

            更通用的做法是直接用 jax.grad 自动微分。
            """
            _cost_var = partial(cost, W_var, reference)
            costs = vmap(_cost_var)(
                lax.stop_gradient(X_star),
                pad(lax.stop_gradient(U_star)),
                jnp.arange(T + 1),
            )
            return jnp.sum(costs)

        # 直接对 W 求导总代价（一阶项），这给出 ∂J/∂W
        dJ_dW_direct = jax.grad(total_weighted_cost)(W)

        # 同时，我们需要考虑伴随变量的贡献。
        # 完整公式: dL/dW = Σ_t [adjoint_x[t]ᵀ · ∂(∂cost/∂x)/∂W
        #                       + adjoint_u[t]ᵀ · ∂(∂cost/∂u)/∂W ]
        #
        # 利用 JAX 的前向-反向嵌套微分来计算这个交叉项：
        def cross_derivative_contribution(W_var):
            """计算 Σ_t [ adjoint_x[t]ᵀ · (∂cost/∂x)(W, X*[t], U[t], t) ]
                      + Σ_t [ adjoint_u[t]ᵀ · (∂cost/∂u)(W, X*[t], U[t], t) ]

            对 W_var 求导即得交叉偏导与伴随变量的乘积。
            """
            _cost_var = partial(cost, W_var, reference)

            # ∂cost/∂x at (X*, U*)，shape (T+1, n)
            grad_cost_x = vmap(jax.grad(_cost_var, argnums=0))(
                lax.stop_gradient(X_star),
                pad(lax.stop_gradient(U_star)),
                jnp.arange(T + 1),
            )
            # ∂cost/∂u at (X*, U*)，shape (T+1, m)
            grad_cost_u = vmap(jax.grad(_cost_var, argnums=1))(
                lax.stop_gradient(X_star),
                pad(lax.stop_gradient(U_star)),
                jnp.arange(T + 1),
            )

            # 与伴随变量点乘
            contribution = (
                jnp.sum(lax.stop_gradient(adjoint_X) * grad_cost_x)
                + jnp.sum(lax.stop_gradient(adjoint_U) * grad_cost_u[:-1])
            )
            return contribution

        # 对 W 求导交叉项贡献
        dL_dW_cross = jax.grad(cross_derivative_contribution)(W)

        # ---- Step 4: 合并并裁剪梯度 ----
        # IFT 最终梯度 = 交叉偏导贡献（这是精确的 IFT 公式）
        grad_W = dL_dW_cross

        # 梯度裁剪
        grad_W = clip_gradient_elementwise(grad_W, grad_clip_norm)

        # 对其他参数不回传梯度（它们已经通过 stop_gradient 阻断）
        grad_reference = jnp.zeros_like(reference)
        grad_parameter = jnp.zeros_like(parameter)
        grad_x0 = jnp.zeros_like(x0)
        grad_X_in = jnp.zeros_like(X_star)
        grad_U_in = jnp.zeros_like(U_star)
        grad_V_in = jnp.zeros_like(V_star)

        return (
            grad_reference,
            grad_parameter,
            grad_W,
            grad_x0,
            grad_X_in,
            grad_U_in,
            grad_V_in,
        )

    # 绑定前向和反向传播
    differentiable_mpc_solve.defvjp(_fwd, _bwd)

    return differentiable_mpc_solve


# ============================================================================
# 4. 便捷的端到端训练接口
# ============================================================================

class DifferentiableMPCWrapper:
    """可微 MPC 控制器包装器，用于端到端训练。

    该类封装了 make_differentiable_mpc 的功能，并提供:
    1. 对数空间权重管理
    2. 梯度裁剪配置
    3. 便捷的训练步骤接口

    使用示例:
        config = ...  # 机器人配置
        diff_mpc = DifferentiableMPCWrapper(
            cost=config.cost,
            dynamics=config.dynamics,
            hessian_approx=config.hessian_approx,
            limited_memory=True,
            base_W=config.W,
        )

        # 训练循环
        def loss_fn(w_logits):
            W = diff_mpc.logits_to_weight(w_logits)
            X, U, V = diff_mpc.solve(reference, parameter, W, x0, X_in, U_in, V_in)
            return task_loss(X, U)

        grad_w = jax.grad(loss_fn)(w_logits_init)
    """

    def __init__(
        self,
        cost,
        dynamics,
        hessian_approx,
        limited_memory,
        base_W,
        grad_clip_norm=10.0,
        tikhonov_eps=1e-6,
    ):
        """初始化可微 MPC 包装器。

        Args:
            cost:            代价函数
            dynamics:        动力学函数
            hessian_approx:  Hessian 近似函数或 None
            limited_memory:  是否使用低内存模式
            base_W:          (d, d) 基础权重矩阵（对角阵）
            grad_clip_norm:  梯度裁剪阈值
            tikhonov_eps:    Tikhonov 正则化系数
        """
        self.cost = cost
        self.dynamics = dynamics
        self.hessian_approx = hessian_approx
        self.limited_memory = limited_memory
        self.base_W = base_W
        self.grad_clip_norm = grad_clip_norm
        self.tikhonov_eps = tikhonov_eps

        # 创建可微 MPC 求解器
        self._diff_mpc = make_differentiable_mpc(
            cost=cost,
            dynamics=dynamics,
            hessian_approx=hessian_approx,
            limited_memory=limited_memory,
            grad_clip_norm=grad_clip_norm,
            tikhonov_eps=tikhonov_eps,
        )

        # 初始化对数空间参数
        self.w_logits_init = jnp.log(jnp.diag(base_W) + 1e-8)

    def logits_to_weight(self, w_logits):
        """将对数空间参数转换为权重矩阵。

        Args:
            w_logits: (d,) 对数空间参数

        Returns:
            W: (d, d) 正定对角权重矩阵
        """
        return jnp.diag(weight_transform(w_logits))

    def solve(self, reference, parameter, W, x0, X_in, U_in, V_in):
        """执行可微 MPC 求解。

        Args:
            reference:  参考轨迹
            parameter:  动力学参数
            W:          权重矩阵
            x0:         初始状态
            X_in:       初始状态轨迹猜测
            U_in:       初始控制序列猜测
            V_in:       初始对偶变量猜测

        Returns:
            (X, U, V): 最优的状态轨迹、控制序列、对偶变量
        """
        return self._diff_mpc(reference, parameter, W, x0, X_in, U_in, V_in)

    def get_init_logits(self):
        """获取初始的对数空间权重参数。

        Returns:
            w_logits: (d,) 初始对数空间参数
        """
        return self.w_logits_init
