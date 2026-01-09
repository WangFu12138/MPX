# 视觉端到端可微 MPC 技术方案总结

**项目**: 基于视觉端到端可微MPC的双足机器人非结构化地形运动控制

**文档版本**: v1.0
**日期**: 2026-01-08

---

## 目录

1. [核心思想](#1-核心思想)
2. [MPX 代码架构解析](#2-mpx-代码架构解析)
3. [端到端训练架构](#3-端到端训练架构)
4. [关键技术细节](#4-关键技术细节)
5. [实现路线图](#5-实现路线图)
6. [核心代码示例](#6-核心代码示例)

---

## 1. 核心思想

### 1.1 问题动机

当前双足机器人控制的三大痛点：

| 范式 | 痛点 | 原因 |
|------|------|------|
| **分层MPC** | 感知与控制解耦 | 视觉误差对MPC不可见，只能盲目执行 |
| **端到端RL** | 样本效率极低 | 需要亿级样本，黑盒不可解释 |
| **视觉残差学习** | 核心参数固定 | MPC内部参数未利用视觉信息 |

### 1.2 我们的方案

```
视觉感知（深度图）
    ↓
视觉编码器（CNN/ViT）
    ↓
地形表示（代价图 + 可学习权重）
    ↓
可微 MPC（MPX）
    ↓
关节力矩
```

**核心创新**: 将 MPC 作为**可学习、可求导的层**嵌入神经网络，实现真正的端到端优化。

---

## 2. MPX 代码架构解析

### 2.1 状态空间（R2 机器人）

```python
# 状态维度: 99 (当 grf_as_state = False)
x = [
    p(3),           # 基座位置 [x, y, z]
    quat(4),        # 基座姿态 [qw, qx, qy, qz]
    q(31),          # 关节角度
    dp(3),          # 线速度
    omega(3),       # 角速度
    dq(31),         # 关节速度
    p_leg(24)       # 足端位置 (8脚 × 3坐标)
]
```

### 2.2 MPC 控制循环流程

```
┌─────────────────────────────────────────────┐
│  主循环 (500 Hz)                            │
│  每 2ms 执行一次 mujoco.mj_step               │
└─────────────────────────────────────────────┘
                    │
                    ▼ 每 20ms (50 Hz)
┌─────────────────────────────────────────────┐
│  1. 读取状态 (qpos, qvel)                    │
│  2. 正向运动学 (计算足端位置)                │
│  3. 构建初始状态 x0                          │
│  4. 生成参考轨迹 (reference_generator)       │
│  5. MPC 优化 (_solve)                        │
│  6. 提取控制 (tau, q, dq)                   │
│  7. 应用到仿真器                             │
└─────────────────────────────────────────────┘
```

### 2.3 参考轨迹生成

**关键函数**: `reference_generator` in `mpc_utils.py`

```python
# 参考 trajectory 结构: (N+1, 60) = (26, 60)
reference = [
    p_ref(3), quat_ref(4), q_ref(31),
    dp_ref(3), omega_ref(3),
    p_leg_ref(24), contact(8), grf_ref(24)
]

# 生成逻辑:
1. 更新步态计时器 (timer_run)
2. 检测抬脚时刻 (liftoff)
3. 计算落脚点 (calc_foothold)
4. 生成摆动轨迹 (cubic_spline)
5. 更新地面反力参考
```

### 2.4 MPC 优化求解器

**核心算法**: Primal-Dual iLQR (in `optimizers.py`)

```python
# 优化问题:
min Σ cost(x_t, u_t)
s.t. dynamics(x_t, u_t) = 0
     constraints(x_t, u_t) ≤ 0

# 求解步骤:
1. 二次化代价函数: cost ≈ 1/2 * x'Qx + q'x
2. 线性化动力学: dynamics ≈ A*x + B*u
3. 求解 KKT 系统 (原始+对偶变量)
4. TVLQR 反向递推 (Riccati 方程)
5. 线搜索更新轨迹
```

### 2.5 代价函数结构

**当前实现** (`objectives.py:r2_wb_obj`):

```python
stage_cost =
    # 1. 位置代价
    (p - p_ref)ᵀ @ Qp @ (p - p_ref)

    # 2. 姿态代价
    (quat - quat_ref)ᵀ @ Qrot @ (quat - quat_ref)

    # 3. 关节角度代价
    (q - q_ref)ᵀ @ Qq @ (q - q_ref)

    # 4. 速度代价
    (dp - dp_ref)ᵀ @ Qdp @ (dp - dp_ref)
    (omega - omega_ref)ᵀ @ Qomega @ (omega - omega_ref)
    dqᵀ @ Qdq @ dq

    # 5. 足端位置代价 (关键!)
    (p_leg - p_leg_ref)ᵀ @ Qleg @ (p_leg - p_leg_ref)

    # 6. 力矩代价
    tauᵀ @ Qtau @ tau

    # 7. 地面反力代价
    (grf - grf_ref)ᵀ @ Qgrf @ (grf - grf_ref)
```

---

## 3. 端到端训练架构

### 3.1 完整的梯度流

```
任务成功指标
    ↓
total_loss
    ↓
┌─────────────────────────────────────────┐
│  MPC 优化输出 (X, U, V)                  │
│  - 状态轨迹                              │
│  - 控制序列                              │
│  - 对偶变量                              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  反向传播通过 MPC                        │
│  - 通过 KKT 条件                         │
│  - 隐函数微分                            │
│  - 链式法则                              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  可微代价函数                            │
│  - 跟踪代价                              │
│  - 地形代价 (可微插值)                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  视觉网络输出                            │
│  - 代价图 (cost_map)                     │
│  - 权重调整 (weight_adjustments)         │
└─────────────────────────────────────────┘
    ↓
视觉网络参数更新
```

### 3.2 视觉网络设计

```python
class TerrainAwareVisionNet(flax.linen.Module):
    """地形感知视觉网络"""

    @flax.linen.compact
    def __call__(self, depth_image):
        # 输入: (batch, H, W, 1)

        # 共享编码器
        x = Conv(32, (5,5), strides=2)(depth_image)
        x = relu(x)
        x = Conv(64, (5,5), strides=2)(x)
        x = relu(x)
        x = Conv(128, (3,3), strides=2)(x)
        x = relu(x)

        # 全局特征
        global_feat = Dense(256)(x.reshape((-1,)))

        # 分支 1: 代价图
        x = Dense(8*8*128)(global_feat)
        x = x.reshape((-1, 8, 8, 128))
        x = ConvTranspose(64, (3,3), strides=2)(x)
        cost_map = ConvTranspose(1, (3,3), strides=2)(x)
        # 输出: (batch, H, W)

        # 分支 2: 权重调整
        weight_adjustments = Dense(154)(global_feat)
        # 输出: (batch, 154)
        # 表示对数空间的权重调整

        return cost_map, weight_adjustments
```

### 3.3 自适应代价函数

```python
def adaptive_terrain_aware_cost(
    n_joints, n_contact, N, W_base, reference,
    x, u, t, cost_map, weight_adjustments
):
    """
    自适应地形感知代价函数

    Args:
        W_base: (154, 154) 基础权重矩阵
        cost_map: (H, W) 地形代价图
        weight_adjustments: (154,) 权重调整系数 (对数空间)
    """

    # 1. 构建自适应权重
    log_W = jnp.log(jnp.diag(W_base)) + weight_adjustments
    W_adaptive = jnp.exp(jax.scipy.linalg.diag(log_W))

    # 2. 提取状态
    p = x[:3]
    quat = x[3:7]
    q = x[7:7+n_joints]
    p_leg = x[13+2*n_joints:13+2*n_joints+3*n_contact]

    # 3. 参考状态
    p_ref = reference[t, :3]
    quat_ref = reference[t, 3:7]
    q_ref = reference[t, 7:7+n_joints]
    p_leg_ref = reference[t, 13+n_joints:13+n_joints+3*n_contact]
    contact = reference[t, 13+n_joints+3*n_contact:13+n_joints+4*n_contact]

    # 4. 地形代价 (可微插值)
    terrain_cost = 0.0
    for i in range(n_contact):
        foot_x, foot_y = p_leg[3*i], p_leg[3*i+1]

        # 可微双线性插值
        cost_val = map_coordinates(
            cost_map,
            [foot_x, foot_y],
            order=1,  # 双线性插值
            mode='constant'
        )

        # 只在接触时施加
        terrain_cost += contact[i] * cost_val ** 2

    # 5. 总代价 (使用自适应权重)
    stage_cost = (
        (p - p_ref).T @ W_adaptive[:3,:3] @ (p - p_ref) +
        (quat - quat_ref).T @ W_adaptive[3:7,3:7] @ (quat - quat_ref) +
        (q - q_ref).T @ W_adaptive[7:7+n_joints,7:7+n_joints] @ (q - q_ref) +
        (p_leg - p_leg_ref).T @
            W_adaptive[12+2*n_joints:12+2*n_joints+3*n_contact,
                        12+2*n_joints:12+2*n_joints+3*n_contact]
            @ (p_leg - p_leg_ref) +
        1e4 * terrain_cost  # 地形代价权重
    )

    return stage_cost
```

---

## 4. 关键技术细节

### 4.1 为什么必须学习权重？

**问题**: 固定权重会导致梯度无法优化

```python
# 固定权重 (错误做法)
W_fixed = diag([1e3, 1e3, 1e4, ...])
cost = (x - x_ref)ᵀ @ W_fixed @ (x - x_ref)
∂cost/∂W_fixed = 0  ← 无法优化权重!

# 可学习权重 (正确做法)
log_W = learnable_parameter
W = exp(log_W)
cost = (x - x_ref)ᵀ @ W @ (x - x_ref)
∂cost/∂log_W ≠ 0  ← 可以优化权重!
```

### 4.2 梯度如何自动找到最优策略

**场景 1: 小石头**

```
代价图: [0, 0, 1, 0, 0]  (局部高代价)

MPC 尝试:
  - 踏上: tracking_cost=0, terrain_cost=1e4 → total=10000
  - 绕开: tracking_cost=100, terrain_cost=0 → total=100 ← 最优

梯度反馈:
  ∂total/∂W_xy < 0  (增加 XY 权重有帮助)
  ∂total/∂W_z ≈ 0   (Z 权重影响不大)

权重更新: W_xy ↑, W_z 不变
```

**场景 2: 大台阶**

```
代价图: [1, 1, 1, 1, 1]  (大范围高代价)

MPC 尝试:
  - 绕开: tracking_cost=10000 (偏离太大)
  - 踏上: tracking_cost=100 (Z偏离), terrain_cost=0 → total=100 ← 最优

梯度反馈:
  ∂total/∂W_xy ≈ 0  (XY 权重影响不大)
  ∂total/∂W_z < 0   (增加 Z 权重有帮助)

权重更新: W_xy 不变, W_z ↑
```

### 4.3 关键技术点

#### 1. 对数空间权重

```python
# 使用对数空间确保权重为正
log_W = learnable_parameter  # 范围约 [-10, 10]
W = exp(log_W)              # 范围约 [1e-4, 1e4]

# 优点:
# - 保证权重始终为正
# - 梯度更稳定
# - 避免数值下溢
```

#### 2. 可微地形插值

```python
from jax.scipy.ndimage import map_coordinates

# 可微双线性插值
cost_val = map_coordinates(
    cost_map,        # (H, W) 代价图
    [foot_x, foot_y], # 足端位置 (浮点坐标)
    order=1,          # 双线性插值
    mode='constant'
)

# 自动计算:
# ∂cost_val/∂cost_map ← 可用于更新视觉网络
# ∂cost_val/∂foot_x, ∂cost_val/∂foot_y ← MPC 使用
```

#### 3. 梯度裁剪

```python
# 防止梯度爆炸
clipped_grads = jax.tree_map(
    lambda g: jnp.clip(g, -1.0, 1.0),
    grads
)

# 或者按范数裁剪
clipped_grads = jax.tree_map(
    lambda g: g * min(1.0, 1.0 / (jnp.linalg.norm(g) + 1e-6)),
    grads
)
```

#### 4. 权重正则化

```python
def weight_regularization(log_W, log_W_init):
    # L2 正则: 鼓励接近初始值
    l2_reg = jnp.mean((log_W - log_W_init) ** 2)

    # 熵正则: 鼓励分布均匀
    entropy_reg = -jnp.mean(
        jax.scipy.special.entropy(jax.nn.softmax(log_W))
    )

    return l2_reg + 0.1 * entropy_reg
```

---

## 5. 实现路线图

### 阶段 1: 验证可行性 (1-2周)

**目标**: 在 MPX 中集成地形代价函数

```python
# 修改 objectives.py
def r2_wb_obj_terrain_aware(n_joints, n_contact, N, W, reference, x, u, t, cost_map):
    # 添加地形代价项
    terrain_cost = compute_terrain_cost(x, cost_map)

    # 原始代价
    original_cost = r2_wb_obj(n_joints, n_contact, N, W, reference, x, u, t)

    return original_cost + 1e4 * terrain_cost
```

**验证**:
- [ ] MPC 能否根据静态代价图调整轨迹
- [ ] 可微插值是否正常工作
- [ ] 梯度能否正常反向传播

### 阶段 2: 视觉编码器 (2-3周)

**目标**: 实现轻量级视觉编码器

```python
# 创建 mpx/utils/vision_encoder.py
class VisionEncoder(flax.linen.Module):
    def __call__(self, depth_image):
        # CNN 特征提取
        # 输出代价图和权重调整
        return cost_map, weight_adjustments
```

**验证**:
- [ ] 视觉网络能否收敛
- [ ] 输出的代价图是否合理
- [ ] 权重调整是否在合理范围

### 阶段 3: 端到端训练框架 (3-4周)

**目标**: 实现完整的训练循环

```python
# 创建 mpx/examples/train_visual_mpc.py
def train_step(vision_params, mpc_wrapper, env_batch):
    # 1. 视觉前向传播
    cost_map, weight_adj = vision_network.apply(vision_params, depth_images)

    # 2. MPC 求解
    X, U, V = mpc_wrapper.run_with_terrain(qpos, qvel, input, contact, cost_map, weight_adj)

    # 3. 计算损失
    loss = compute_task_loss(X, U, terrain_types)

    # 4. 反向传播
    grads = jax.grad(loss)(vision_params)

    return loss, grads
```

**验证**:
- [ ] 梯度能否从任务成功传回视觉网络
- [ ] 训练是否稳定收敛
- [ ] 样本效率是否优于纯 RL

### 阶段 4: 实验评估 (2-3周)

**测试场景**:
- [ ] 小石头: 测试 XY 方向调整
- [ ] 大台阶: 测试 Z 方向调整
- [ ] 斜坡: 测试姿态适应
- [ ] 湿滑地面: 测试摩擦系数适应

**评估指标**:
- [ ] 地形穿越成功率
- [ ] 样本效率 (与纯 RL 对比)
- [ ] 能量效率 (COT)
- [ ] 计算延迟

---

## 6. 核心代码示例

### 6.1 完整的训练循环

```python
import jax
import jax.numpy as jnp
import flax
import optax

class VisualMPCSystem:
    def __init__(self, config, mpc_wrapper):
        self.config = config
        self.mpc = mpc_wrapper

        # 初始化视觉网络
        self.vision_net = TerrainAwareVisionNet()
        self.vision_params = self.vision_net.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 128, 128, 1))  # dummy input
        )

        # 初始化可学习权重
        base_W = config.W
        init_log_W = jnp.log(jnp.diag(base_W))
        self.log_W = jnp.array(init_log_W)

        # 优化器
        self.optimizer = optax.adam(learning_rate=1e-4)
        self.opt_state = self.optimizer.init(self.vision_params)

    def train_step(self, batch_rng, env_batch):
        """单步训练"""

        # 1. 视觉前向传播
        cost_maps, weight_adjustments = jax.vmap(self.vision_net.apply)(
            self.vision_params, env_batch['depth_images']
        )

        # 2. 构建当前权重
        current_W = jnp.exp(jax.scipy.linalg.diag(self.log_W))

        # 3. 批量 MPC 求解
        def mpc_objective(cost_map):
            X, U, V = self.mpc.solve(
                env_batch['qpos'],
                env_batch['qvel'],
                env_batch['input'],
                env_batch['contact'],
                current_W,
                cost_map
            )

            # 计算任务指标
            success = evaluate_success(X, U, env_batch['terrain_type'])
            energy = compute_energy(U)

            # 目标: 最小化负成功 = 最大化成功
            objective = -success + 0.1 * energy

            return objective

        # 向量化计算所有环境的 MPC
        objectives = jax.vmap(mpc_objective)(cost_maps)
        total_loss = jnp.mean(objectives)

        # 4. 反向传播
        loss, grads = jax.value_and_grad(total_loss)(self.vision_params)

        # 5. 更新参数
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state
        )
        self.vision_params = optax.apply_updates(updates, self.vision_params)

        # 6. 同步更新权重 (如果需要)
        # 这里可以添加权重更新的逻辑

        return loss, self.vision_params
```

### 6.2 自适应 MPC 包装器

```python
class AdaptiveMPCWrapper:
    """支持地形感知的 MPC 包装器"""

    def __init__(self, base_config):
        self.config = base_config
        self.base_W = base_config.W

        # 初始化基础 MPC
        self.base_mpc = MPCControllerWrapper(base_config)

    def run_with_terrain(self, qpos, qvel, input, contact,
                         cost_map, weight_adjustments):
        """使用地形信息运行 MPC"""

        # 1. 动态构建权重
        log_W = jnp.log(jnp.diag(self.base_W)) + weight_adjustments
        W_adaptive = jnp.exp(jax.scipy.linalg.diag(log_W))

        # 2. 替换代价函数
        original_cost = self.config.cost
        terrain_aware_cost = partial(
            adaptive_terrain_aware_cost,
            W_base=W_adaptive,
            cost_map=cost_map,
            weight_adjustments=weight_adjustments
        )

        # 3. 临时替换配置
        self.config.cost = terrain_aware_cost

        # 4. 运行 MPC
        tau, q, dq = self.base_mpc.run(qpos, qvel, input, contact)

        # 5. 恢复原始配置
        self.config.cost = original_cost

        return tau, q, dq
```

### 6.3 环境生成器

```python
class TerrainEnv:
    """地形环境生成器"""

    def __init__(self):
        self.terrain_types = [
            'flat',      # 平地
            'small_rock', # 小石头
            'large_stair', # 大台阶
            'ramp',      # 斜坡
            'slippery'   # 湿滑
        ]

    def sample_batch(self, batch_size, rng_key):
        """采样批量环境"""
        keys = jax.random.split(rng_key, batch_size)

        batch = {
            'depth_images': [],
            'qpos': [],
            'qvel': [],
            'input': [],
            'contact': [],
            'terrain_type': []
        }

        for i in range(batch_size):
            # 随机选择地形类型
            terrain_type = jax.random.choice(
                keys[i],
                jnp.array(self.terrain_types),
                ()
            )

            # 生成对应的深度图
            depth_image = self.generate_depth_image(keys[i], terrain_type)

            # 初始化机器人状态
            qpos, qvel = self.init_robot_state(keys[i], terrain_type)

            batch['depth_images'].append(depth_image)
            batch['qpos'].append(qpos)
            batch['qvel'].append(qvel)
            batch['input'].append(jnp.array([0.5, 0, 0, 0, 0, 0, 0.65]))
            batch['contact'].append(jnp.zeros(8))
            batch['terrain_type'].append(terrain_type)

        return batch
```

---

## 7. 关键公式总结

### 7.1 优化问题

```
外层优化 (训练视觉网络和权重):
  min_vision_params  E_env[ -success(mpc(vision_params, env)) ]

内层优化 (MPC):
  min_x,u  Σ tracking_cost(x, u; W(vision_params))
          + terrain_cost(x; cost_map(vision_params))
  s.t.    dynamics(x, u) = 0
          constraints(x, u) ≤ 0
```

### 7.2 代价函数

```
J(x, u) = (x - x_ref)ᵀ @ W(θ) @ (x - x_ref)  # 跟踪代价
        + Σ_i contact[i] · cost_map(p_leg[i])²    # 地形代价

其中:
  W(θ) = exp(log_W_base + log_W_adjustment(θ))
  cost_map: 可微插值从视觉网络输出
```

### 7.3 梯度链式法则

```
∂success/∂vision_params =
    ∂success/∂(X, U) ·
    ∂(X, U)/∂W ·
    ∂W/∂log_W_adjustment ·
    ∂log_W_adjustment/∂vision_params
```

---

## 8. 技术优势总结

| 维度 | 传统方法 | 本方案 |
|------|---------|--------|
| **样本效率** | 纯 RL: 1e8 样本 | **物理梯度指导: 1e6 样本** |
| **适应性** | 固定权重 | **自适应权重学习** |
| **可解释性** | 黑盒 RL | **可解释的代价图 + 权重** |
| **地形适应** | 视觉与控制分离 | **端到端优化** |
| **策略选择** | 硬编码规则 | **MPC 自动权衡** |

---

## 9. 参考文献

1. **MPX**: Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots (RA-L/ICRA 2025)
2. **iSDF**: Real-Time Neural Signed Distance Fields for Robot Perception (RSS 2022)
3. **Differentiable Physics**: End-to-End Differentiable Physics for Learning and Control (NeurIPS 2018)
4. **RL-Augmented MPC**: RL-augmented Adaptive MPC for Bipedal Locomotion (ArXiv 2025)

---

## 10. 下一步行动

### 立即可做:

1. **验证 MPX 可微性**
   ```bash
   cd mpx
   python tests/simple_tests.py
   ```

2. **实现简单地形代价**
   - 修改 `objectives.py` 添加静态代价图
   - 验证 MPC 能否调整轨迹

3. **实现视觉编码器**
   - 创建 `mpx/utils/vision_encoder.py`
   - 测试 CNN 输出

4. **端到端训练框架**
   - 创建 `mpx/examples/train_visual_mpc.py`
   - 测试梯度反向传播

---

**文档状态**: 技术方案总结完成
**下一步**: 开始阶段 1 实现 (验证可行性)
