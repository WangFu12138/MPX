# Config 目录说明

`config` 目录用于存放系统内各类机器人实体或特定控制任务的**参数配置文件**。所有的系统权重、预测视野、控制周期参数均在这里被集中定义，供 `MPCControllerWrapper` 在初始化时调用。

## 文件功能详述：

- **`config_r2.py` / `config_h1.py` / `config_go2.py` / `config_aliengo.py`**:
  针对特定结构双足/四足机器人（如 R2、Unitree H1、Go2、Aliengo）的专用控制参数设定。它们包括各个关节角限位、状态维度、默认的目标行走频次以及对应的不同状态权重 $Q$ 矩阵和输入权重 $R$ 矩阵基底。

- **`config_srbd.py`**:
  专门用于“简化的单刚体动力学模型”的控制参数配置（常结合 `mpc_wrapper_srbd.py` 使用）。

- **`config_talos.py` & `config_barrel_roll.py`**:
  针对特殊机器人实体（如 Talos 的 MJX 移植）或复杂的高难度特技运动（如 Barrel Roll 翻滚），其姿态惩罚权重、容错条件或末端受力容限在此单独配置。
