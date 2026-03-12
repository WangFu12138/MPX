# CLAUDE.md
！！！任何时候回答我都要先称呼我为王甫12138
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

!!! 项目的每个文件夹层级下都有md文档，每当修改了当前的文件夹代码我确认无误后，都需要同步修改md文档，保证最新。同样的当你每次开始修改前都请务必读一下md文档，保证你对当前文件夹的代码有清晰的认识。
二、 Claude Code 项目实现规则 (Claude Code Implementation Rules)(作为 .claudeproject 中的 system instructions 或项目根目录的规则文件，规范代码编写)技术栈约束 (Tech Stack): 严格使用 PyTorch 作为深度学习和自动微分框架。物理仿真环境必须基于 MuJoCo 构建。如需进行 ROS 集成测试，保持节点之间的通信接口清晰。模块化解耦 (Modular Architecture):尽管是端到端训练，代码结构必须解耦：vision_encoder.py，dynamics_model.py，diff_mpx_solver.py。禁止将网络的前向传播与复杂的物理仿真逻辑写在同一个巨型类中。可微求解器实现细节 (Differentiable Solver):在实现 MPX 求解层时，优先考虑数值稳定性。明确处理无解或求解器发散的边缘情况（例如，返回一个 detach 的安全 fallback 动作，并在此 step 截断梯度）。使用 torch.autograd.Function 自定义反向传播逻辑时，必须编写 gradcheck 单元测试，确保解析梯度与数值梯度一致。张量形状与设备管理 (Tensor Management):所有函数和方法的 docstring 必须明确标注输入/输出张量的形状 (e.g., [batch_size, horizon, state_dim])。保持 device 统一，避免在计算图中出现不必要的 CPU/GPU 数据拷贝。优先支持 batched 操作以加速 MuJoCo 的 rollout 和求解。实验复现与日志 (Reproducibility & Logging):所有的超参数（网络结构、MPX 视界 $N$、各项 cost 权重）必须通过统一的配置文件（如 YAML 或 Hydra）管理，严禁在代码中 hardcode。训练主循环中，必须记录并非仅有最终 loss，还要记录 MPX 求解的平均迭代次数、求解耗时等指标。

## 项目概述

**MPX** 是一个基于 JAX 的腿式机器人模型预测控制（MPC）库。该项目实现了原始-对偶 iLQR（Primal-Dual iLQR）优化算法，利用 GPU 并行计算实现高效的轨迹优化和控制。

### 核心特性
- **GPU 并行加速**: 时间复杂度从 O(N(n + m)^3) 降低到 O(log^2 n log N + log^2 m)
- **完全可微**: 求解器支持 JAX 自动微分，可集成到学习流水线中
- **批量环境支持**: 支持并行运行多个 MPC 控制器
- **MJX 集成**: 支持 MuJoCo/MJX 整体动力学模型

## 安装和设置

```bash
# 克隆仓库（包含子模块）
git clone git@github.com:iit-DLSLab/mpx.git
cd mpx && git submodule update --init --recursive

# 创建 conda 环境
conda create -n mpx_env python=3.13 -y
conda activate mpx_env

# 安装依赖
pip install -e .
```

## 运行示例

```bash
# 激活环境
conda activate mpx_env

# 运行 R2-1024 机器人示例（键盘控制）
python mpx/examples/mjx_r2.py

# 运行其他机器人示例
python mpx/examples/mjx_h1.py
python mpx/examples/mjx_talos.py
python mpx/examples/mjx_quad.py
```

**注意**: 首次运行可能需要超过一分钟来进行 JIT 编译。JAX 编译缓存会保存到 `./jax_cache` 目录。

## 测试

测试使用 `absl.testing` 框架，位于 `mpx/primal_dual_ilqr/tests/` 目录：

```bash
# 运行所有测试
python -m absltest mpx.primal_dual_ilqr.tests.simple_tests

# 运行特定测试
python mpx/primal_dual_ilqr/tests/simple_tests.py
python mpx/primal_dual_ilqr/tests/cartpole.py
python mpx/primal_dual_ilqr/tests/quadpendulum.py
```

## 代码架构

### 目录结构

```
mpx/
├── config/              # 机器人配置文件（每个机器人一个配置）
├── data/               # MuJoCo/URDF 模型文件
├── examples/           # 运行示例
├── primal_dual_ilqr/   # 核心优化算法（Git 子模块）
└── utils/              # MPC 工具和包装器
```

### 核心模块

1. **primal_dual_ilqr/** (Git 子模块)
   - `optimizers.py`: 主要的优化求解器实现
   - `primal_tvlqr.py`: 原始时变 LQR
   - `dual_tvlqr.py`: 对偶时变 LQR
   - `kkt_helpers.py`: KKT 条件辅助函数
   - `linalg_helpers.py`: 线性代数辅助函数

2. **utils/mpc_wrapper.py**
   - `MPCControllerWrapper`: 单个 MPC 控制器包装器
   - `BatchedMPCControllerWrapper`: 批量并行 MPC 控制器
   - `mpx_data`: 存储控制器数据的 PyTreeNode 类

3. **utils/mpc_wrapper_srbd.py**
   - SRBD（简化刚体动力学）模型的 MPC 包装器

4. **config/**: 机器人配置文件
   - `config_*.py`: 定义动力学模型、成本函数、初始状态等
   - 每个机器人有独立的配置文件

5. **utils/models.py**: 动力学模型实现
6. **utils/objectives.py`: 目标/成本函数定义
7. **utils/mpc_utils.py`: MPC 辅助工具（参考生成器等）

### MPC 工作流程

1. 配置文件定义：机器人的动力学模型、成本函数、初始状态
2. `MPCControllerWrapper` 初始化：加载 MuJoCo 模型，设置优化函数
3. 控制循环：
   - 读取当前状态 (qpos, qvel)
   - 调用 `mpc.run()` 计算最优控制输入
   - 应用控制到仿真器

## 添加新机器人

1. 在 `mpx/data/` 添加机器人的 MuJoCo/URDF 模型文件
2. 在 `mpx/config/` 创建新的 `config_*.py` 文件，定义：
   - `model_path`: 模型文件路径
   - `contact_frame`: 接触点名称
   - `body_name`: 相关刚体名称
   - `n_joints`, `n_contact`: 维度信息
   - `cost`, `dynamics`, `hessian_approx`: 使用的函数
   - 成本矩阵 (Qp, Qrot, Qq, 等)
   - 初始状态 (p0, quat0, q0, p_legs0)
3. 在 `mpx/examples/` 创建对应的运行脚本

## 两种模型类型

1. **Whole Body Dynamics** (整体动力学)
   - 使用 `MPCControllerWrapper` 或 `BatchedMPCControllerWrapper`
   - 使用 `mpc_wrapper.py`
   - 适用于完整动力学仿真

2. **SRBD** (简化刚体动力学)
   - 使用 `mpc_wrapper_srbd.py`
   - 适用于简化的动力学模型
   - 参考 `examples/srbd_quad.py`

## 并行环境

对于批量并行处理（如强化学习），使用 `BatchedMPCControllerWrapper`：

```python
from mpx.utils.mpc_wrapper import BatchedMPCControllerWrapper

mpc = BatchedMPCControllerWrapper(config, n_env=100)
mpc_data = mpc.make_data()
# 使用 vmap 或循环处理多个环境
```

参考 `examples/multi_env.py`。

## 关键配置参数

- `dt`: MPC 时间步长（秒）
- `N`: 预测 horizon 的阶段数
- `mpc_frequency`: MPC 更新频率 (Hz)
- `step_freq`: 步频 (Hz)
- `duty_factor`: 步态占空比
- `step_height`: 步高（米）

---

# 创新研究方向

## 方向一：视觉端到端的可微MPC（Pixel-to-Torque）

### 痛点分析

目前的双足机器人"看路"和"走路"是分离的：
- 视觉处理 → 高程图（Elevation Map）→ 提取平面 → MPC
- 这个过程慢，且丢失地形的几何梯度信息
- 当前代码中 `use_terrain_estimation = False`，完全没有地形感知

### MPX创新结合点

利用MPX的GPU原生特性，实现**零拷贝（Zero-Copy）**的视觉-控制一体化：

#### 核心思路

1. **不要把地形压缩成2D平面**：在JAX中构建SDF（符号距离场）碰撞模型
2. **Pixel-to-Torque端到端控制器**：
   ```
   深度图/NeRF → CNN特征提取 → 代价权重预测 → MPC → 关节力矩
   ```
3. **自适应步态调整**：根据地形特征动态调整MPC参数
   - 软沙地：自动调大阻尼
   - 台阶前：自动拉长预测时域

### 实现路线图

#### 1. 创建视觉特征提取模块 (`utils/vision_encoder.py`)

```python
# 功能：从深度图/RGB图提取地形特征
# 输入：深度图 (H, W) 或 RGB (H, W, 3)
# 输出：地形特征向量 (dim_feature,)
# 关键：使用JAX实现，可微分
```

#### 2. 实现SDF地形碰撞检测 (`utils/sdf_terrain.py`)

```python
# 功能：计算足端位置到地形的SDF距离
# 输入：足端3D坐标 (x, y, z)
# 输出：SDF值（负值=穿透，正值=悬空）
# 关键：GPU加速，可微分，支持任意复杂地形
```

#### 3. 创建自适应代价权重预测网络 (`utils/cost_weight_predictor.py`)

```python
# 功能：根据地形特征预测MPC代价权重
# 输入：地形特征向量
# 输出：自适应的W矩阵（替代config中固定的对角W）
# 架构：小型MLP或Transformer
```

#### 4. 修改代价函数支持地形约束 (`utils/objectives_terrain_aware.py`)

```python
# 在原有 r2_wb_obj 基础上增加：
# - SDF碰撞惩罚项（避免脚穿模）
# - 地形梯度感知项（脚掌贴合地形）
# - 自适应权重注入
```

#### 5. 创建端到端训练框架 (`examples/train_pixel_to_torque.py`)

```python
# 训练目标：端到端学习地形适应策略
# 损失函数：任务成功 + 能耗 + 稳定性
# 数据：各种地形下的仿真数据（台阶、斜坡、碎石等）
```

### 关键技术点

1. **可微SDF**：使用 JAX 自动微分，SDF对位置的梯度可直接用于优化
2. **零拷贝集成**：视觉编码器、SDF、MPC 全在 GPU 上，无 CPU-GPU 数据传输
3. **批量并行训练**：利用 `BatchedMPCControllerWrapper` 同时训练多个地形场景

### 预期创新点

- **首个真正的视觉-运动端到端可微控制器**：从像素直接到力矩
- **自适应步态**：MPC自动学会在不同地形调整策略
- **GPU加速的地形感知**：SDF查询比传统高程图快100x

### 相关文件

- 当前代价函数：`utils/objectives.py:r2_wb_obj` (第252行)
- 当前配置：`config/config_r2.py` (W矩阵定义在第64-86行)
- 示例入口：`examples/mjx_r2.py`
