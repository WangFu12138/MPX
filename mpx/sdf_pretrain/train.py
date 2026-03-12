#!/usr/bin/env python3
"""Training script for SDF pretraining.

Features:
- ResNet-18 encoder with BatchNorm (properly handled mutable state)
- Eikonal regularization loss
- Online data generation

Usage:
    python -m mpx.sdf_pretrain.train --epochs 100 --batch-size 8
"""

import os
import sys
import argparse
from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import struct

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFOnlineGenerator
from mpx.sdf_pretrain.models.network import EndToEndSDFNetwork
from mpx.sdf_pretrain.models.losses import sdf_mse_loss, eikonal_loss


def parse_args():
    parser = argparse.ArgumentParser(description="SDF Pretraining")
    parser.add_argument('--xml', type=str,
                        default='mpx/data/r2-1024/mjcf/scene_terrain_test.xml',
                        help='Path to terrain XML')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-queries', type=int, default=128)
    parser.add_argument('--heightmap-size', type=int, default=21)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mse-weight', type=float, default=1.0)
    parser.add_argument('--eikonal-weight', type=float, default=0.1)
    parser.add_argument('--eikonal-every', type=int, default=5,
                        help='Compute eikonal loss every N epochs (expensive)')
    parser.add_argument('--randomize-every', type=int, default=20,
                        help='Randomize terrain layout every N epochs (0 to disable)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='checkpoints/sdf_pretrain')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    return parser.parse_args()

# 使用Flax的struct来定义训练指标类，包含MSE, Eikonal和Total Loss
@struct.dataclass
class Metrics:
    """Training metrics."""
    mse: float
    eikonal: float
    total: float

# 继承Flax的TrainState，添加batch_stats和metrics
class TrainState(train_state.TrainState):
    """Training state with batch_stats for BatchNorm."""
    batch_stats: Dict
    metrics: Metrics


def create_train_state(rng, model, learning_rate, input_shape, query_shape):
    """Create initial training state with batch_stats."""
    # Initialize model parameters
    dummy_heightmap = jnp.ones(input_shape)
    dummy_queries = jnp.ones(query_shape)

    # Initialize with mutable batch_stats
    variables = model.init(rng, dummy_heightmap, dummy_queries, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # Create optimizer with gradient clipping for stability
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=learning_rate)
    )

    # Create state
    state = TrainState.create(
        apply_fn=model.apply,#模型前向传播函数，训练时调用state.apply_fn({'params': params, 'batch_stats': batch_stats}, x, train=True)
        params=params,#模型参数
        tx=tx,#优化器
        batch_stats=batch_stats,#BatchNorm的统计信息
        metrics=Metrics(mse=0.0, eikonal=0.0, total=0.0)#训练指标，记录了MSE,Eikonal,总损失
    )
    return state


def load_checkpoint(state: TrainState, checkpoint_path: str) -> TrainState:
    """Load params and batch_stats from checkpoint.

    Args:
        state: Current training state (will be updated)
        checkpoint_path: Path to .npz checkpoint file

    Returns:
        Updated training state with loaded params and batch_stats
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = jnp.load(checkpoint_path, allow_pickle=True)

    # Extract params and batch_stats from checkpoint
    params = checkpoint['params'].item()
    batch_stats = checkpoint['batch_stats'].item()

    # Update state
    state = state.replace(params=params, batch_stats=batch_stats)
    print("Checkpoint loaded successfully!")
    return state


@partial(jax.jit, static_argnames=['compute_eikonal'])# 使用jax.jit进行编译，static_argnames=['compute_eikonal']表示compute_eikonal是静态参数
def train_step(state: TrainState, batch: Dict, compute_eikonal: bool = False) -> Tuple[TrainState, Dict]:
    """Perform a single training step.

    Args:
        state: Current training state
        batch: Training batch
        compute_eikonal: Whether to compute Eikonal loss (expensive)

    Returns:
        Updated state and loss dict
    """
    heightmap = batch['heightmap']
    queries = batch['queries_local']
    targets = batch['sdf']

    def loss_fn(params, batch_stats):
        # Forward pass with mutable batch_stats
        predictions, new_mutables = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries, train=True, mutable=['batch_stats']
        )
        new_batch_stats = new_mutables['batch_stats']

        # MSE loss
        mse_loss = sdf_mse_loss(predictions, targets)

        # Eikonal loss (optional, expensive)
        if compute_eikonal:
            eik_loss = eikonal_loss(
                state.apply_fn,
                params,
                new_batch_stats,
                heightmap, queries, predictions,
                epsilon=1e-3,  # Larger epsilon for numerical stability
                surface_threshold=0.3
            )
        else:
            eik_loss = jnp.array(0.0)

        total_loss = mse_loss + 0.1 * eik_loss
        return total_loss, (mse_loss, eik_loss, new_batch_stats)

    # 使用jax.value_and_grad计算损失和梯度
    (total_loss, (mse_loss, eik_loss, new_batch_stats)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params, state.batch_stats)

    # 应用梯度
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    # Return losses as JAX arrays (conversion to float happens outside JIT)
    return state, (mse_loss, eik_loss)


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("SDF Pretraining with ResNet + Eikonal Loss")
    print("="*60)
    print(f"  Terrain XML: {args.xml}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num queries: {args.num_queries}")
    print(f"  Heightmap size: {args.heightmap_size}x{args.heightmap_size}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  MSE weight: {args.mse_weight}")
    print(f"  Eikonal weight: {args.eikonal_weight}")
    print(f"  Eikonal every: {args.eikonal_every} epochs")
    print(f"  Randomize every: {args.randomize_every} epochs")
    print(f"  Seed: {args.seed}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("="*60 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get absolute path
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(script_dir, args.xml)

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        return

    # Initialize generator
    print("初始化数据生成器...")
    generator = SDFOnlineGenerator(
        xml_path=xml_path,
        heightmap_size=(args.heightmap_size, args.heightmap_size),
        num_queries_per_sample=args.num_queries,
    )

    # Initialize model
    print("Initializing model...")
    rng = jax.random.PRNGKey(args.seed)
    model = EndToEndSDFNetwork(
        latent_dim=args.latent_dim,
        hidden_dims=(256, 256, 128)
    )

    # Input shapes for initialization
    input_shape = (args.batch_size, args.heightmap_size, args.heightmap_size, 3)
    query_shape = (args.batch_size, args.num_queries, 3)

    state = create_train_state(rng, model, args.lr, input_shape, query_shape)

    # Load checkpoint if resuming
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(script_dir, args.resume)
        if not os.path.exists(resume_path):
            print(f"Error: Checkpoint not found: {resume_path}")
            return
        state = load_checkpoint(state, resume_path)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model initialized with {num_params:,} parameters")

    # Training loop
    print("\nStarting training...")
    key = jax.random.PRNGKey(args.seed)

    best_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        # 随机化地形布局
        if args.randomize_every > 0 and epoch > 0 and epoch % args.randomize_every == 0:
            key, subkey = jax.random.split(key)
            print(f"Randomizing terrain layout for epoch {epoch+1}...")
            generator.randomize_terrain(subkey)

        # Generate batch
        key, subkey = jax.random.split(key)
        batch = generator.generate_batch(subkey, args.batch_size)

        # Debug info for first epoch
        if epoch == 0:
            print("\nData statistics:")
            print(f"  Heightmap shape: {batch['heightmap'].shape}")
            print(f"  Heightmap range: [{float(batch['heightmap'].min()):.4f}, {float(batch['heightmap'].max()):.4f}]")
            print(f"  SDF range: [{float(batch['sdf'].min()):.4f}, {float(batch['sdf'].max()):.4f}]")
            print(f"  SDF mean: {float(batch['sdf'].mean()):.4f}")
            print()

        # Determine if we should compute Eikonal loss
        compute_eik = (epoch % args.eikonal_every == 0)

        # Training step
        state, (mse_loss, eik_loss) = train_step(state, batch, compute_eikonal=compute_eik)

        # Convert to Python floats for logging
        mse_val = float(mse_loss)
        eik_val = float(eik_loss)
        total_val = mse_val + 0.1 * eik_val

        loss_dict = {
            'mse': mse_val,
            'eikonal': eik_val,
            'total': total_val,
        }

        # Track best loss
        if total_val < best_loss:
            best_loss = total_val

        history.append(loss_dict)

        # Print progress
        if compute_eik:
            print(f"Epoch {epoch+1:4d}/{args.epochs} - "
                  f"MSE: {mse_val:.6f} | "
                  f"Eik: {eik_val:.6f} | "
                  f"Total: {total_val:.6f}")
        else:
            print(f"Epoch {epoch+1:4d}/{args.epochs} - "
                  f"MSE: {mse_val:.6f} | "
                  f"Total: {total_val:.6f}")

    # Summary
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final MSE Loss: {mse_val:.6f}")
    print(f"Final Total Loss: {total_val:.6f}")
    print(f"Best Total Loss: {best_loss:.6f}")
    print("="*60 + "\n")

    # Save final checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'final_checkpoint.npz')
    jnp.savez(
        checkpoint_path,
        params=state.params,
        batch_stats=state.batch_stats,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
