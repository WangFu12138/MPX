"""
JAX/Flax implementation of the Conditioned Neural SDF (Stage 2)
1:1 复刻原仓库的 Positional Encoding + ReLU MLP 架构
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    """
    位置编码模块 (Positional Encoding)
    将低维的 3D 坐标 (x,y,z) 映射到高维的高频空间，
    让普通的 MLP 能够敏锐地捕捉到台阶的锐利边缘。
    """
    num_freqs: int = 6 # 默认使用 6 个频率频段

    @nn.compact
    def __call__(self, x):
        # x 的输入形状通常是 (Batch, Num_Points, 3)
        # 生成频率序列: [2^0, 2^1, 2^2, ..., 2^(L-1)] * pi
        freqs = 2.0 ** jnp.arange(self.num_freqs) * jnp.pi
        
        # 增加一个维度以进行广播相乘
        # x[..., None] 变成 (Batch, Num_Points, 3, 1)
        # freqs 形状是 (6,)
        # scaled 相乘后变成 (Batch, Num_Points, 3, 6)
        scaled = x[..., None] * freqs
        
        # 将最后的 3x6 展平为 18 维
        scaled = scaled.reshape(list(x.shape[:-1]) + [-1])
        
        # 拼接原始坐标 x, 以及它的 sin 和 cos 映射
        # 最终维度 = 3 (原坐标) + 18 (sin) + 18 (cos) = 39 维
        return jnp.concatenate([x, jnp.sin(scaled), jnp.cos(scaled)], axis=-1)


class ConditionedSDFMLP(nn.Module):
    """
    条件隐式场多层感知机 (Conditioned SDF MLP)
    接收 90维的地形特征 + 3D空间点坐标，输出该点的距离场标量 (SDF)
    """
    hidden_dim: int = 256
    num_layers: int = 6  # 经典的 6 层 256 宽 MLP

    @nn.compact
    def __call__(self, z, points):
        """
        参数:
          z: 来自 VAE 的潜变量特征, 形状 (Batch, Latent_Dim)  [例如: B, 90]
          points: 3D 空间采样点, 形状 (Batch, Num_Points, 3)
        """
        # --- 1. 特征广播 (Latent Broadcasting) ---
        # 目标: 将 (B, 90) 扩展为 (B, Num_Points, 90) 才能和点拼接
        z_expanded = jnp.expand_dims(z, axis=1) # 变成 (B, 1, 90)
        # jnp.broadcast_to 极其高效，不会在显存里真正复制数据
        z_broadcast = jnp.broadcast_to(
            z_expanded, 
            (z.shape[0], points.shape[1], z.shape[-1])
        )
        
        # --- 2. 空间点位置编码 ---
        pe_points = PositionalEncoding(num_freqs=6)(points) # 变成 (B, Num_Points, 39)
        
        # --- 3. 维度拼接 (Conditioning) ---
        # 拼接后输入 MLP 的维度是: 90 + 39 = 129 维
        x = jnp.concatenate([z_broadcast, pe_points], axis=-1)
        
        # --- 4. MLP 前向传播 ---
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x) # 坚守最纯粹的 ReLU，放弃 SIREN
            
        # 最后一层，输出单维度 SDF 标量 (没有任何激活函数)
        sdf_out = nn.Dense(features=1)(x)
        
        # 形状是 (Batch, Num_Points, 1)，通常我们会把最后的 1 挤压掉
        return jnp.squeeze(sdf_out, axis=-1)