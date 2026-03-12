# Examples 目录说明

`examples` 目录负责封装所有的**使用用例与启动脚本**，是用户测试本项目各种不同机器人和不同地形策略的最直接入口。

## 目录文件说明：

- **`mjx_r2.py` & `mjx_r2_rough_terrain.py`**:
  这是 R2 机器人在基础平台和“粗糙地形 (rough_terrain)” 下的单例执行控制脚本，运行此脚本会启动物理引擎并且让机器人执行 MPC 处理（也是验证前向控制系统是否运转正常的常用入口）。

- **`mjx_h1.py` / `mjx_quad.py` / `mjx_talos.py`**:
  其它主流机器人（如 Unitree H1、传统四足Quadruped、乃至 Talos）的控制演示代码。

- **`multi_env.py`**:
  演示如何使用 GPU 并行环境 (`BatchedMPCControllerWrapper`)，在一张显卡上同时控制数量众多的机器人。该方案多用于强化学习的数据收集与并行推导测试。

- **`srbd_quad.py`**:
  展示如何使用单刚体动力学（而非全身 Whole-Body 动力学）求解四足机器人的运动规划与控制。

- **`acrobat.py` & `barel_roll.py`**:
  专为高动态机动（例如侧空翻或后空翻）设计的特技动作模拟脚本。

- **`test_diff_mpc_r2.py`**:
  R2 机器人可微 MPC (IFT) 的集成测试脚本。用于验证 `differentiable_mpc.py` 在真实 99 维状态空间下是否能正常完成前向求解和梯度回传（纯命令行无 GUI）。

- **`diff_mpc_r2_viz.py`**:
  R2 机器人可微 MPC (IFT) 的可视化仿真脚本。将可微 MPC 结合到 MuJoCo 仿真中，实现实时物理仿真、键盘交互控制行走（调速/转向），并支持按 `G` 键在线计算代价权重的隐式梯度。
