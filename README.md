# MPX - 基于JAX的腿式机器人MPC库

基于JAX的双足机器人模型预测控制（MPC）库，实现原始-对偶iLQR优化算法。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/WangFu12138/MPX.git
cd MPX
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
# 创建conda环境
conda create -n mpx_env python=3.13 -y
conda activate mpx_env

# 安装项目
pip install -e .
```

### 3. 运行示例

```bash
# 激活环境
conda activate mpx_env

# 运行 R2-1024 机器人
python mpx/examples/mjx_r2.py
```

首次运行可能需要超过一分钟进行JIT编译。

## 其他机器人示例

```bash
python mpx/examples/mjx_h1.py      # H1机器人
python mpx/examples/mjx_talos.py   # Talos机器人
python mpx/examples/mjx_quad.py    # 四足机器人
```
