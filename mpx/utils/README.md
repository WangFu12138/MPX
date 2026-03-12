# Utils 目录说明

`utils` 目录是项目的**核心工具与外壳层 (Wrapper Layer)**，直接连接底层的 MPC 求解器、MuJoCo/MJX 物理引擎以及上层应用，提供各种控制抽象、代价函数以及地形处理方法。

## 目录与文件功能：

- **`mpc_wrapper.py` & `mpc_wrapper_srbd.py`**:
  MPC 包装器核心文件。负责封装底层 Primal-Dual iLQR 优化器，处理状态同步、仿真器 (`MJX`) 与控制器的交互工作。`srbd` 对应的是单刚体动力学（Simplified Rigid Body Dynamics）特化包装。

- **`mpc_utils.py`**:
  主要包含轨迹参考生成（reference generator）等控制辅助功能，例如步态周期设定、落脚点计划等。

- **`objectives.py`**:
  端到端可微的核心代价函数（Cost Function）存储地。此处包括了用于不同机器人配置（如 R2, H1 等）的基础代价配置以及后续集成视觉输入形变的自适应地形代价项（Adaptive Terrain Cost）。

- **`models.py`**:
  封装机器人的动力学方程或正逆运动学解析结构，供求解器作为算子使用。

- **`terrain_generator.py`**:
  地形生成器，用于在 JAX 环境中生成平地、台阶、碎石等多种地形，便于训练或前向仿真。

- **`preview_terrain*.py`**:
  各种不同种类地形预测和轨迹分析功能的文件（包括动态地形处理和阶梯类地形等预览功能），帮助 MPC 实现感知预期。

- **`rotation.py` & `console.py`**:
  包含系统底层的四元数/旋转矩阵转换与终端打印相关的通用辅助函数。
