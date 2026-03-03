## 项目概述

**MPX** 是一个基于 JAX 的腿式机器人模型预测控制（MPC）库。该项目实现了原始-对偶 iLQR（Primal-Dual iLQR）优化算法，利用 GPU 并行计算实现高效的轨迹优化和控制。

### 核心特性
- **GPU 并行加速**: 时间复杂度从 O(N(n + m)^3) 降低到 O(log^2 n log N + log^2 m)
- **完全可微**: 求解器支持 JAX 自动微分，可集成到学习流水线中
- **批量环境支持**: 支持并行运行多个 MPC 控制器
- **MJX 集成**: 支持 MuJoCo/MJX 整体动力学模型

## 安装和设置


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