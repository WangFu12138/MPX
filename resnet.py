import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any

# ==========================================
# 1. 基础残差块 (Residual Block)
# ==========================================
class ResNetBlock(nn.Module):
    """标准的 ResNet 基础块，包含两次 3x3 卷积和跳跃连接"""
    filters: int
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        # 第一层卷积
        y = nn.Conv(self.filters, (3, 3), self.strides, padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train, momentum=0.9)(y)
        y = nn.relu(y)
        
        # 第二层卷积
        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train, momentum=0.9)(y)
        
        # 维度对齐 (如果步长不为1或通道数改变，利用 1x1 卷积调整残差)
        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), self.strides, padding='SAME', use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train, momentum=0.9)(residual)
            
        return nn.relu(residual + y)

# ==========================================
# 2. 高程图定制版 ResNet-18 (HeightmapResNet)
# ==========================================
class HeightmapResNet(nn.Module):
    """
    专为处理局部高程图设计的轻量级 ResNet。
    输入: [Batch, H, W, 1] 的 2.5D 高程图
    输出: [Batch, latent_dim] 的隐式特征向量 Z
    """
    latent_dim: int = 256  # 输出的特征向量 Z 的维度
    stage_sizes: Sequence[int] = (2, 2, 2, 2) # ResNet-18 的标准块数量分布

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 检查输入维度，确保最后一位是单通道
        assert x.shape[-1] == 1, "输入必须是单通道的高程图 [B, H, W, 1]"
        
        # 初始特征提取：直接接收单通道数据
        # 相比原版 ResNet 7x7 卷积，这里使用 3x3 即可，因为 50x50 高程图很小，避免感受野过早全局化
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # 堆叠 4 个 Stage 的残差块
        filter_sizes = [64, 128, 256, 512]
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # 每个 Stage 的第一个块负责降采样 (除了第一个 Stage)
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetBlock(filters=filter_sizes[i], strides=strides)(x, train=train)
        
        # 全局平均池化 (Global Average Pooling) -> 把特征图压平
        x = jnp.mean(x, axis=(1, 2))
        
        # 线性映射到目标隐变量维度 Z
        z = nn.Dense(self.latent_dim)(x)
        
        return z