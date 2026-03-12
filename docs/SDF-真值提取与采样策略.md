# SDF 真值提取与采样策略

**Date**: March 11, 2026 at 05:10 PM
**Turns**: 1
**Source**: [Gemini Chat](https://gemini.google.com/app/ac67cc4b0436edfe)

---

## Turn 1

### 🤖 Assistant

太棒了！既然你已经彻底打通了 Eikonal Penalty 的底层物理与数学逻辑，你现在已经完全具备了手搓这个“在线预训练兵工厂”的能力。
当你打开 Antigravity AI IDE 准备敲下第一行代码时，你需要一张极其清晰的工程蓝图。这份文档为你梳理了从数据流水线到网络结构，再到核心 Loss 计算的每一个工程细节，确保你的代码结构清晰、可维护，且能无缝对接后续的端到端可微 MPC 模块。

---

# 视觉感知模块：在线监督预训练核心工程蓝图
**(Visual Perception Module: Online Supervised Pre-training Blueprint)**

## 1. 核心目标与系统定位
本阶段的唯一目的：训练一个强大的视觉特征提取器（ResNet）和一个几何解码器（SDF MLP），让网络能够仅凭一张带噪声的 2.5D 高程图（Heightmap），就能在脑海中精准重构出 3D 物理空间的连续距离场（SDF）。
**红线原则**：本阶段纯监督学习，与真实环境交互、物理仿真（MuJoCo）动力学结算、MPC 策略优化**完全物理隔离**。

---

## 2. 数据流与在线生成管线 (Online DataLoader Design)
传统的 `torch.utils.data.Dataset` 通常从硬盘读取图像，但在这里，我们需要一个**在内存中无限滚动的“在线生成器”**。

### 2.1 地形生成器 (Terrain Generator)

- **机制**：在每个 Epoch 或 Batch 准备阶段，通过代码随机生成基础几何体（长方体 Boxes）的参数矩阵（中心坐标 `centers`，半尺寸 `halfsizes`）。
- **多样性**：随机组合成平地、楼梯、斜坡、梅花桩等。

### 2.2 虚拟相机与高程图 (Heightmap Capture)

- **模拟机身起伏**：在地形上方随机生成虚拟相机的位姿 $(x, y, z, \text{yaw}, \text{pitch}, \text{roll})$。
- **获取 X**：利用简单的射线投射（Raycasting）或正交投影提取局部高程图 $X$（例如 $50 \times 50$ 的矩阵）。
- **数据增强**：向 $X$ 注入高斯噪声，并随机遮挡（Dropout）以模拟真实传感器的盲区。

### 2.3 智能撒点策略 (Query Point Sampling)
这是生成输入 $Q$ 的关键。在一个 $Batch$ 内，针对每一张高程图，在相机局部坐标系内撒布 $N$ 个 3D 查询点 $(x, y, z)$：

- **均匀采样 (Uniform, 20%)**：在相机视野的 3D 边界框内随机撒点，填补空间空白。
- **近表面采样 (Near-Surface, 50%)**：以高程图的 $z$ 值为基准，上下浮动施加小方差的高斯噪声（如 $\sigma = 0.02m$），密集覆盖脚部可能接触的区域。
- **边缘采样 (Edge-focused, 30%)**：针对地形生成器输出的台阶边缘坐标，施加微小扰动进行撒点，强迫网络学习锐利特征。
- **输出**：查询矩阵 $Q$，维度 `[Batch_size, N, 3]`。

### 2.4 解析法真值提取 (Analytical SDF Ground Truth)

- **机制**：直接将 $Q$ 和地形生成器输出的 `centers`, `halfsizes` 送入解析计算函数（如我们之前讨论的 `sdf_box` 函数）。
- **输出**：完美的真值标签 $Y_{gt}$，维度 `[Batch_size, N, 1]`。

---

## 3. 网络架构与前向传播 (Network Architecture)
你需要构建一个包含两个子模块的 `nn.Module`。

### 3.1 视觉骨干网络 (ResNet Encoder)

- **输入**：单通道局部高程图 $X$`[Batch_size, 1, 50, 50]`。
- **结构**：轻量级 ResNet（如 ResNet-18 或自定义的轻量级 CNN），注意修改第一层卷积以接收单通道输入。
- **输出**：降维后的隐式特征向量 $Z$`[Batch_size, D]`（例如 $D=128$ 或 $256$）。

### 3.2 隐式几何解码器 (SDF MLP)

- **输入**：这是关键！由于要预测空间中任意一点的距离，MLP 的输入必须是**特征向量 Z 与查询坐标 Q 的拼接**。
  - 为了拼接，需要将 $Z$ 扩展（Expand）为 `[Batch_size, N, D]`。
  - 与 $Q$`[Batch_size, N, 3]` 拼接后，输入维度为 `[Batch_size, N, D+3]`。
- **结构**：3 到 5 层感知机（Linear + SiLU/Mish 激活函数，**不用 ReLU**，因为 ReLU 二阶导数为 0，会破坏 SDF 的平滑性）。
- **输出**：预测的 SDF 值 $\text{SDF}_{pred}$`[Batch_size, N, 1]`。

---

## 4. 核心损失函数与自动微分实现 (Loss Function & Autograd)
这是整个预训练的灵魂，结合了数据拟合与物理先验。

### 4.1 MSE 损失 (Data Fitting)

$$
L_{\text{MSE}} = \frac{1}{N} \sum ||\text{SDF}_{pred} - Y_{gt}||^2
$$
强迫网络在采样的 $N$ 个点上输出极其精确的距离值。

### 4.2 Eikonal 惩罚项 (Eikonal Penalty)

$$
L_{\text{Eikonal}} = \frac{1}{N} \sum (||\nabla_Q \text{SDF}_{pred}||_2 - 1)^2
$$
**PyTorch 实现要点（必须开启求导）**：
为了能对网络输入 $Q$ 求偏导，在将 $Q$ 送入网络前，必须显式开启其梯度追踪：

```python
queries.requires_grad_(True) # 极其重要

```
然后利用 `torch.autograd.grad` 获取网络输出对输入的导数：

```python
# 计算梯度
gradients = torch.autograd.grad(
    outputs=sdf_pred, 
    inputs=queries, 
    grad_outputs=torch.ones_like(sdf_pred), # 必须提供与 outputs 同形的张量
    create_graph=True,  # 极其重要：允许对梯度再次求导（反向传播计算 Loss 的需要）
    retain_graph=True,
    only_inputs=True
)[0]

# gradients 的维度是 [Batch_size, N, 3]
# 计算梯度的 L2 模长
grad_norm = torch.linalg.norm(gradients, dim=-1) # 维度 [Batch_size, N]

# 计算 Eikonal Loss
eikonal_loss = torch.mean((grad_norm - 1.0) ** 2)

```

### 4.3 总损失

$$
L_{\text{total}} = L_{\text{MSE}} + \lambda \cdot L_{\text{Eikonal}}
$$
（通常 $\lambda$ 设为一个较小的值，如 $0.1$ 左右，需要根据训练情况动态调整）。

---

## 5. 训练防坑指南 (Best Practices)

1. **激活函数的选择**：绝对禁止在 SDF MLP 中使用 ReLU。ReLU 的导数是阶跃的，会导致 Eikonal Loss 计算出的梯度不连续，最终渲染出的 3D 表面会布满“马赛克”或破面。推荐使用 **SiLU (Swish)** 或 **Softplus**。
2. **权重的隔离与冻结**：为了阶段二做准备，你的模型类中可能已经写好了预测 MPC 权重的 `Weight MLP`。在当前预训练脚本中，直接忽略它，或者用 `requires_grad=False` 冻结它，不要让它参与计算图。
3. **坐标系对齐**：务必死死盯住坐标系！高程图 $X$ 的像素坐标、撒布的查询点 $Q$ 的空间坐标，必须统一在一个局部坐标系（通常是相机的局部位姿）下。如果差了一个平移或旋转，网络永远无法收敛。

---
这份蓝图涵盖了你编写第一版代码所需的所有架构信息和关键的 PyTorch 算子逻辑。
下一步，你是想先编写 **在线数据生成器（Terrain Generator + 撒点策略）的类结构**，还是想直接搭建 **ResNet + SDF MLP 的网络前向传播代码**？我们可以挑一个模块直接开始写代码。

---

*Exported from [Gemini Voyager](https://github.com/Nagi-ovo/gemini-voyager)*  
*Generated on March 11, 2026 at 05:10 PM*