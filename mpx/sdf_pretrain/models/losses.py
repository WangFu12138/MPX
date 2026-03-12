"""Loss functions for SDF pretraining.

This module provides:
- sdf_mse_loss: MSE loss for SDF prediction accuracy
- eikonal_loss: Regularization loss for gradient (|∇SDF| = 1)
- sdf_total_loss: Combined loss function
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Callable, Optional


def sdf_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss for SDF prediction.

    Args:
        predictions: Predicted SDF values, shape (batch, N)
        targets: Ground truth SDF values, shape (batch, N)

    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((predictions - targets) ** 2)


def compute_sdf_gradient(
    apply_fn: Callable,
    params: dict,
    batch_stats: dict,
    heightmap: jnp.ndarray,
    queries: jnp.ndarray,
    epsilon: float = 1e-4
) -> jnp.ndarray:
    """Compute SDF gradient norm w.r.t. query points using finite differences.

    Uses central difference for numerical stability:
    grad_f(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    Args:
        apply_fn: Model apply function
        params: Model parameters
        batch_stats: Batch normalization statistics
        heightmap: Heightmap input
        queries: Query points, shape (batch, N, 3)
        epsilon: Finite difference step size

    Returns:
        Gradient norm at each query point, shape (batch, N)
    """
    batch_size, num_queries, _ = queries.shape

    # Compute gradient for each axis
    grad_norm_sq = jnp.zeros((batch_size, num_queries))

    for axis in range(3):
        # Create perturbation vectors
        offset = jnp.zeros(3).at[axis].set(epsilon)

        # f(x + ε)
        queries_plus = queries + offset
        sdf_plus = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries_plus, train=False, mutable=['batch_stats']
        )[0]

        # f(x - ε)
        queries_minus = queries - offset
        sdf_minus = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries_minus, train=False, mutable=['batch_stats']
        )[0]

        # Central difference gradient
        grad_axis = (sdf_plus - sdf_minus) / (2 * epsilon)

        # Accumulate squared gradient
        grad_norm_sq = grad_norm_sq + grad_axis ** 2

    return jnp.sqrt(grad_norm_sq)


def eikonal_loss(
    apply_fn: Callable,
    params: dict,
    batch_stats: dict,
    heightmap: jnp.ndarray,
    queries: jnp.ndarray,
    predictions: jnp.ndarray,
    epsilon: float = 1e-4,
    surface_threshold: float = 0.3
) -> jnp.ndarray:
    """Eikonal regularization loss.

    Enforces |∇SDF| = 1 (gradient norm should be 1).

    Only applies the loss near the surface (within surface_threshold) to focus
    learning on the region that matters for collision checking.

    Args:
        apply_fn: Model apply function
        params: Model parameters
        batch_stats: Batch normalization statistics
        heightmap: Heightmap input
        queries: Query points, shape (batch, N, 3)
        predictions: SDF predictions (to determine surface proximity)
        epsilon: Finite difference step size
        surface_threshold: Distance threshold for surface region

    Returns:
        Eikonal loss (scalar)
    """
    # Compute gradient norm using finite differences
    grad_norm = compute_sdf_gradient(
        apply_fn, params, batch_stats,
        heightmap, queries, epsilon
    )

    # Mask: only penalize points near the surface
    surface_mask = jnp.abs(predictions) < surface_threshold

    # Eikonal loss: (|grad| - 1)^2, weighted by surface mask
    eik_loss = ((grad_norm - 1.0) ** 2) * surface_mask

    # Normalize by number of surface points (avoid division by zero)
    num_surface = jnp.maximum(surface_mask.sum(), 1.0)

    return eik_loss.sum() / num_surface


def sdf_total_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mse_weight: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    MSE loss only (Eikonal computed separately).

    Args:
        predictions: Predicted SDF values, shape (batch, N)
        targets: Ground truth SDF values, shape (batch, N)
        mse_weight: Weight for MSE loss

    Returns:
        Dict with 'mse'
    """
    mse = sdf_mse_loss(predictions, targets)

    return {
        'mse': mse,
    }
