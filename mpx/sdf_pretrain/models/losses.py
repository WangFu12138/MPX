"""Loss functions for SDF pretraining.

This module provides:
- truncate_sdf: Truncate SDF values to a bounded range
- sdf_mse_loss: Basic MSE loss for SDF prediction
- sdf_mse_loss_weighted: Distance-weighted MSE loss (near points matter more)
- eikonal_loss: Regularization loss for gradient (|∇SDF| = 1)
- eikonal_loss_decoupled: iSDF-style decoupled Eikonal loss (uniform sampling)
- compute_sdf_gradient: Compute SDF gradient using finite differences

Key improvements:
- Distance-weighted loss: W = e^(-λD), where D is distance from origin
- GT truncation: Clamp ground truth to match network output range
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Callable, Optional


def truncate_sdf(sdf: jnp.ndarray, truncation_distance: float) -> jnp.ndarray:
    """Truncate SDF values to a bounded range.

    This is CRITICAL when using Tanh output layer. The network can only output
    values in [-truncation_distance, +truncation_distance]. If we don't truncate
    the ground truth, the network will never be able to match far-away GT values,
    causing training instability.

    Args:
        sdf: SDF values, shape (...)
        truncation_distance: Maximum absolute SDF value (e.g., 0.3 for ±30cm)

    Returns:
        Truncated SDF values, clamped to [-truncation_distance, +truncation_distance]
    """
    return jnp.clip(sdf, -truncation_distance, truncation_distance)


def sdf_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Basic Mean squared error loss for SDF prediction.

    Args:
        predictions: Predicted SDF values, shape (batch, N)
        targets: Ground truth SDF values, shape (batch, N)

    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((predictions - targets) ** 2)


def sdf_mse_loss_weighted(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    queries_local: jnp.ndarray,
    lambda_decay: float = 1.0,
    epsilon: float = 1e-8
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Distance-weighted MSE loss for SDF prediction.

    Near points (close to robot center) have higher weight, far points have lower weight.
    Weight formula: W = e^(-λ * D), where D = sqrt(x² + y² + z²)

    This forces the network to focus on near-surface geometry (within 50cm) which
    is critical for collision avoidance, rather than wasting capacity on far-away
    details that don't matter for control.

    Args:
        predictions: Predicted SDF values, shape (batch, N)
        targets: Ground truth SDF values (should be truncated!), shape (batch, N)
        queries_local: Query points in local frame, shape (batch, N, 3)
        lambda_decay: Distance decay coefficient (default 1.0)
            - λ=1.0: 0.5m → W≈0.61, 1.0m → W≈0.37, 2.0m → W≈0.14
            - λ=2.0: 0.5m → W≈0.37, 1.0m → W≈0.14, 2.0m → W≈0.02 (more aggressive)
        epsilon: Small constant to prevent NaN gradient when D=0 (default 1e-8)

    Returns:
        Tuple of (weighted_mse_loss, info_dict)
        - weighted_mse_loss: Scalar loss value
        - info_dict: Dictionary with additional metrics for monitoring
    """
    # Compute distance from origin (robot center)
    # D = sqrt(x² + y² + z² + epsilon) to prevent NaN gradient at D=0
    distances = jnp.sqrt(jnp.sum(queries_local ** 2, axis=-1) + epsilon)  # (batch, N)

    # Compute distance-based weights: W = e^(-λ * D)
    weights = jnp.exp(-lambda_decay * distances)  # (batch, N)

    # Weighted MSE: mean(W * (pred - target)²)
    squared_error = (predictions - targets) ** 2  # (batch, N)
    weighted_loss = jnp.mean(weights * squared_error)  # scalar

    # Compute additional metrics for monitoring
    # Near-surface error (within 0.5m)
    near_mask = distances < 0.5
    near_error = jnp.where(near_mask, squared_error, 0.0)
    near_mse = jnp.sum(near_error) / jnp.maximum(jnp.sum(near_mask), 1.0)

    # Far-surface error (beyond 1.0m)
    far_mask = distances > 1.0
    far_error = jnp.where(far_mask, squared_error, 0.0)
    far_mse = jnp.sum(far_error) / jnp.maximum(jnp.sum(far_mask), 1.0)

    info = {
        'near_surface_mse': near_mse,  # Core metric: must be < 0.001 (1mm error)
        'far_surface_mse': far_mse,    # Less important
        'mean_distance': jnp.mean(distances),
        'mean_weight': jnp.mean(weights),
    }

    return weighted_loss, info


def compute_sdf_gradient(
    apply_fn: Callable,
    params: dict,
    batch_stats: dict,
    heightmap: jnp.ndarray,
    queries: jnp.ndarray,
    epsilon: float = 1e-3,
    train: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute SDF gradient norm w.r.t. query points using finite differences.

    Uses central difference for numerical stability:
    grad_f(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    Args:
        apply_fn: Model apply function
        params: Model parameters
        batch_stats: Batch normalization statistics
        heightmap: Heightmap input
        queries: Query points, shape (batch, N, 3)
        epsilon: Finite difference step size (default 1e-3 for stability)
        train: Whether to use training mode (default True for consistency)

    Returns:
        Tuple of (gradient norm, gradient per axis), shape (batch, N) and (batch, N, 3)
    """
    batch_size, num_queries, _ = queries.shape

    # Compute gradient for each axis
    grad_components = []

    for axis in range(3):
        # Create perturbation vectors
        offset = jnp.zeros(3).at[axis].set(epsilon)

        # f(x + ε) - use same train mode as main training
        queries_plus = queries + offset
        sdf_plus = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries_plus, train=train, mutable=['batch_stats']
        )[0]

        # f(x - ε)
        queries_minus = queries - offset
        sdf_minus = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries_minus, train=train, mutable=['batch_stats']
        )[0]

        # Central difference gradient with clipping for stability
        grad_axis = (sdf_plus - sdf_minus) / (2 * epsilon)
        # Clip gradient to prevent explosion
        grad_axis = jnp.clip(grad_axis, -100.0, 100.0)
        grad_components.append(grad_axis)

    # Stack gradients: (batch, N, 3)
    grads = jnp.stack(grad_components, axis=-1)
    grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=-1))

    # Clip gradient norm for additional stability
    grad_norm = jnp.clip(grad_norm, 0.0, 100.0)

    return grad_norm, grads


def eikonal_loss_decoupled(
    apply_fn: Callable,
    params: dict,
    batch_stats: dict,
    heightmap: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_eikonal_points: int = 64,
    bounds: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -0.5, 0.5),
    epsilon: float = 1e-3
) -> jnp.ndarray:
    """iSDF-style decoupled Eikonal loss with uniform spatial sampling.

    Instead of computing Eikonal loss on surface points (which can cause
    gradient conflicts at sharp edges), this samples points uniformly in
    the entire bounding volume.

    This follows iSDF paper Section III-C: the Eikonal regularization
    points are decoupled from the SDF supervision points.

    Args:
        apply_fn: Model apply function
        params: Model parameters
        batch_stats: Batch normalization statistics
        heightmap: Heightmap input, shape (batch, H, W, C)
        key: JAX random key for sampling
        num_eikonal_points: Number of points to sample per batch
        bounds: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        epsilon: Finite difference step size

    Returns:
        Eikonal loss (scalar)
    """
    batch_size = heightmap.shape[0]
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Uniform random sampling in bounding box
    # Shape: (batch, num_eikonal_points, 3)
    keys = jax.random.split(key, 3)

    x_samples = jax.random.uniform(keys[0], (batch_size, num_eikonal_points),
                                   minval=x_min, maxval=x_max)
    y_samples = jax.random.uniform(keys[1], (batch_size, num_eikonal_points),
                                   minval=y_min, maxval=y_max)
    z_samples = jax.random.uniform(keys[2], (batch_size, num_eikonal_points),
                                   minval=z_min, maxval=z_max)

    eikonal_queries = jnp.stack([x_samples, y_samples, z_samples], axis=-1)

    # Compute gradient norm on uniformly sampled points
    grad_norm, _ = compute_sdf_gradient(
        apply_fn, params, batch_stats,
        heightmap, eikonal_queries, epsilon, train=True
    )

    # Eikonal loss: (|grad| - 1)^2 over all sampled points
    # No surface mask needed since we're sampling in free space
    eik_loss = jnp.mean((grad_norm - 1.0) ** 2)

    return eik_loss


def eikonal_loss(
    apply_fn: Callable,
    params: dict,
    batch_stats: dict,
    heightmap: jnp.ndarray,
    queries: jnp.ndarray,
    predictions: jnp.ndarray,
    epsilon: float = 1e-3,
    surface_threshold: float = 0.3
) -> jnp.ndarray:
    """Eikonal regularization loss (legacy, coupled with MSE points).

    Enforces |∇SDF| = 1 (gradient norm should be 1).

    Only applies the loss near the surface (within surface_threshold) to focus
    learning on the region that matters for collision checking.

    WARNING: This uses the same points as MSE loss, which can cause gradient
    conflicts at sharp edges. Consider using eikonal_loss_decoupled instead.

    Args:
        apply_fn: Model apply function
        params: Model parameters
        batch_stats: Batch normalization statistics
        heightmap: Heightmap input
        queries: Query points, shape (batch, N, 3)
        predictions: SDF predictions (to determine surface proximity)
        epsilon: Finite difference step size (default 1e-3 for stability)
        surface_threshold: Distance threshold for surface region

    Returns:
        Eikonal loss (scalar)
    """
    # Compute gradient norm using finite differences with train=True for consistency
    grad_norm, _ = compute_sdf_gradient(
        apply_fn, params, batch_stats,
        heightmap, queries, epsilon, train=True
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
