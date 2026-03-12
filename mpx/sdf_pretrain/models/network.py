"""Neural network models for SDF pretraining.

This module provides:
- ResNetBlock: Basic residual block with Swish activation
- HeightmapResNet: ResNet-18 encoder for heightmap input
- ImplicitSDFDecoder: MLP decoder for SDF prediction
- EndToEndSDFNetwork: Complete end-to-end network
"""

from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn


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

    Input: [Batch, H, W, 1] single-channel heightmap
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


class ImplicitSDFDecoder(nn.Module):
    """MLP decoder for SDF prediction.

    Takes latent features and 3D query point coordinates, outputs SDF value.
    """
    hidden_dims: Sequence[int] = (256, 256, 128)
    output_dim: int = 1

    @nn.compact
    def __call__(self, z: jnp.ndarray, q_local: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: Latent features from encoder, shape (batch, latent_dim)
            q_local: Query points in local frame, shape (batch, num_queries, 3)

        Returns:
            SDF predictions, shape (batch, num_queries)
        """
        batch_size = z.shape[0]
        num_queries = q_local.shape[1]

        # Expand latent to match query count
        z_expanded = jnp.tile(z[:, None, :], (1, num_queries, 1))  # (batch, N, latent_dim)

        # Concatenate latent and queries
        x = jnp.concatenate([z_expanded, q_local], axis=-1)  # (batch, N, latent_dim + 3)

        # MLP with Swish activation
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.swish(x)

        # Output layer (no activation for SDF regression)
        x = nn.Dense(features=self.output_dim)(x)

        return x.squeeze(-1)  # (batch, num_queries)


class EndToEndSDFNetwork(nn.Module):
    """End-to-end SDF prediction network.

    Combines ResNet encoder and MLP decoder.
    """
    latent_dim: int = 256
    hidden_dims: Sequence[int] = (256, 256, 128)

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
            SDF predictions, shape (batch, N)
        """
        # Encode heightmap
        z = HeightmapResNet(latent_dim=self.latent_dim)(heightmap, train)

        # Decode with queries
        sdf_pred = ImplicitSDFDecoder(hidden_dims=self.hidden_dims)(z, queries_local)

        return sdf_pred
