"""Neural network models for SDF pretraining.

This module provides:
- PositionalEncoding: NeRF-style Fourier positional encoding
- ResNetBlock: Basic residual block with Swish activation
- HeightmapResNet: ResNet-18 encoder for heightmap input
- SirenLayer: SIREN layer with special initialization for sin activation
- ImplicitSDFDecoder: MLP decoder with SIREN activation and Tanh output
- EndToEndSDFNetwork: Complete end-to-end network

Key improvements (based on iSDF and SIREN papers):
- SIREN activation (sin) for sharp edge representation
- Tanh output with truncation distance for bounded SDF
- 4-layer funnel architecture: 256 -> 256 -> 128 -> 64
"""

from typing import Sequence
import math
import jax.numpy as jnp
import flax.linen as nn
import jax
from jax.nn import initializers


class PositionalEncoding(nn.Module):
    """NeRF风格的傅里叶位置编码。

    将低维坐标映射到高维空间，使MLP能够学习高频特征（如锐利的边缘）。

    Input: p = (x, y, z), shape (..., 3)
    Output: γ(p), shape (..., 3 * 2 * L)

    Args:
        L: 频率层数，输出维度 = 3 * 2 * L (例如 L=6 → 36维)
    """
    L: int = 6

    @nn.compact
    def __call__(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: 输入坐标, shape (..., 3)

        Returns:
            编码后的特征, shape (..., 3 * 2 * L)
        """
        encoded = []
        for i in range(self.L):
            freq = 2 ** i
            encoded.append(jnp.sin(freq * jnp.pi * p))
            encoded.append(jnp.cos(freq * jnp.pi * p))
        return jnp.concatenate(encoded, axis=-1)


class ResNetBlock(nn.Module):
    """ResNet basic block with two 3x3 convolutions and skip connection.

    Uses Swish activation instead of ReLU for smoother gradients.
    """
    filters: int
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # First conv
        y = nn.Conv(self.filters, (3, 3), self.strides, padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train, momentum=0.9)(y)
        y = nn.swish(y)

        # Second conv
        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train, momentum=0.9)(y)

        # Dimension alignment (if stride != 1 or channels change)
        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), self.strides, padding='SAME', use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train, momentum=0.9)(residual)

        return nn.swish(residual + y)


class HeightmapResNet(nn.Module):
    """Lightweight ResNet-18 for heightmap encoding.

    Input: [Batch, H, W, C] heightmap (C=3 for x,y,z or C=1 for z only)
    Output: [Batch, latent_dim] latent feature vector Z
    """
    latent_dim: int = 256
    stage_sizes: Sequence[int] = (2, 2, 2, 2)  # ResNet-18 standard

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial feature extraction
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
        x = nn.swish(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # 4 stages of residual blocks
        filter_sizes = [64, 128, 256, 512]
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # First block of each stage (except first) does downsampling
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetBlock(filters=filter_sizes[i], strides=strides)(x, train=train)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Linear projection to latent dimension
        z = nn.Dense(self.latent_dim)(x)

        return z


class SirenLayer(nn.Module):
    """SIREN layer with special initialization for sin activation.

    Based on "Implicit Neural Representations with Periodic Activation Functions"
    (Sitzmann et al., 2020)

    Key: Uses sin(omega_0 * Wx + b) instead of ReLU/Wx + b

    Initialization (CRITICAL for training stability):
    - First layer: W ~ U(-1/in_dim, 1/in_dim)
    - Hidden layers: W ~ U(-sqrt(6/in_dim)/omega_0, sqrt(6/in_dim)/omega_0)

    This prevents gradient explosion through multiple sin layers.

    Args:
        features: Output dimension
        omega_0: Frequency multiplier (default 30.0, as recommended by SIREN paper)
        is_first: Whether this is the first layer (affects initialization)
    """
    features: int
    omega_0: float = 30.0
    is_first: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (..., in_dim)

        Returns:
            Output tensor, shape (..., features)
        """
        in_dim = x.shape[-1]

        # SIREN special initialization
        if self.is_first:
            # First layer: uniform distribution in [-1/in_dim, 1/in_dim]
            bound = 1.0 / in_dim
            w_init = initializers.uniform(scale=bound * 2, dtype=jnp.float32)
        else:
            # Hidden layers: uniform distribution in [-sqrt(6/in_dim)/omega_0, sqrt(6/in_dim)/omega_0]
            bound = math.sqrt(6.0 / in_dim) / self.omega_0
            w_init = initializers.uniform(scale=bound * 2, dtype=jnp.float32)

        # Linear transformation
        linear = nn.Dense(features=self.features, kernel_init=w_init, bias_init=initializers.zeros)
        h = linear(x)

        # Apply sin activation with omega_0 frequency
        return jnp.sin(self.omega_0 * h)


class ImplicitSDFDecoder(nn.Module):
    """MLP decoder for SDF prediction with SIREN activation and Tanh output.

    Architecture (based on iSDF + SIREN best practices):
    - 4-layer funnel: 256 -> 256 -> 128 -> 64
    - SIREN activation (sin) for sharp edge representation
    - Tanh output with truncation distance

    Key improvements over vanilla MLP:
    1. SIREN activation captures high-frequency geometry (sharp 90° edges)
    2. Tanh + truncation focuses network capacity on near-surface region
    3. Funnel architecture efficiently compresses information

    Args:
        hidden_dims: MLP hidden layer dimensions (default: 256, 256, 128, 64 funnel)
        pos_encoding_L: Positional encoding frequency layers (default 6 → 36-dim encoding)
        omega_0: SIREN frequency multiplier (default 30.0)
        truncation_distance: Physical truncation distance in meters (default 0.3)
    """
    hidden_dims: Sequence[int] = (256, 256, 128, 64)  # 4-layer funnel
    pos_encoding_L: int = 6  # L=6 → 36维编码
    omega_0: float = 30.0  # SIREN frequency
    truncation_distance: float = 0.3  # 物理截断距离（米）

    @nn.compact
    def __call__(self, z: jnp.ndarray, q_local: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: Latent features from encoder, shape (batch, latent_dim)
            q_local: Query points in local frame, shape (batch, num_queries, 3)

        Returns:
            SDF predictions (truncated to [-truncation_distance, +truncation_distance]),
            shape (batch, num_queries)
        """
        num_queries = q_local.shape[1]

        # Apply positional encoding to query points
        # (batch, N, 3) -> (batch, N, 3 * 2 * L) e.g., (batch, N, 36) for L=6
        q_encoded = PositionalEncoding(L=self.pos_encoding_L)(q_local)

        # Expand latent to match query count
        z_expanded = jnp.tile(z[:, None, :], (1, num_queries, 1))  # (batch, N, latent_dim)

        # Concatenate latent and encoded queries as input
        x = jnp.concatenate([z_expanded, q_encoded], axis=-1)  # (batch, N, latent_dim + 36)
        in_dim = x.shape[-1]

        # SIREN layers (4-layer funnel: 256 -> 256 -> 128 -> 64)
        for i, dim in enumerate(self.hidden_dims):
            x = SirenLayer(
                features=dim,
                omega_0=self.omega_0,
                is_first=(i == 0)
            )(x)

        # Output layer with conservative initialization (NOT SIREN!)
        # Use smaller init to prevent large initial outputs
        final_dim = self.hidden_dims[-1]
        bound = math.sqrt(6.0 / final_dim) / self.omega_0
        final_w_init = initializers.uniform(scale=bound * 2, dtype=jnp.float32)

        raw_out = nn.Dense(features=1, kernel_init=final_w_init, bias_init=initializers.zeros)(x)

        # Tanh normalization + physical scaling
        # Output is bounded to [-truncation_distance, +truncation_distance]
        sdf_out = nn.tanh(raw_out) * self.truncation_distance

        return sdf_out.squeeze(-1)  # (batch, num_queries)


class EndToEndSDFNetwork(nn.Module):
    """End-to-end SDF prediction network.

    Combines ResNet encoder and SIREN decoder with positional encoding.

    Architecture:
    - Encoder: ResNet-18 -> latent_dim
    - Decoder: SIREN MLP (4-layer funnel) with Tanh output

    Key features:
    - SIREN activation for sharp geometric features
    - Truncated SDF output (focuses on near-surface region)
    - Positional encoding for high-frequency detail
    """
    latent_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256, 128, 64)  # 4-layer funnel
    pos_encoding_L: int = 6  # L=6 → 36维编码
    omega_0: float = 30.0  # SIREN frequency
    truncation_distance: float = 0.3  # 物理截断距离（米）

    @nn.compact
    def __call__(self,
                 heightmap: jnp.ndarray,
                 queries_local: jnp.ndarray,
                 train: bool = True) -> jnp.ndarray:
        """
        Args:
            heightmap: Heightmap input, shape (batch, H, W, C)
            queries_local: Query points in local frame, shape (batch, N, 3)
            train: Training mode flag

        Returns:
            SDF predictions (truncated), shape (batch, N)
        """
        # Encode heightmap
        z = HeightmapResNet(latent_dim=self.latent_dim)(heightmap, train)

        # Decode with queries (SIREN decoder with Tanh output)
        sdf_pred = ImplicitSDFDecoder(
            hidden_dims=self.hidden_dims,
            pos_encoding_L=self.pos_encoding_L,
            omega_0=self.omega_0,
            truncation_distance=self.truncation_distance
        )(z, queries_local)

        return sdf_pred
