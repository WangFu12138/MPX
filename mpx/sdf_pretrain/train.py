#!/usr/bin/env python3
"""Training script for SDF pretraining with dynamic terrain generation.

Features:
- ResNet-18 encoder with BatchNorm (properly handled mutable state)
- SIREN decoder with Tanh output and truncation distance
- Distance-weighted MSE loss (near points matter more)
- Eikonal regularization loss
- Dynamic terrain generation using WFC algorithm
- Curriculum learning support
- GPU utilization optimization: prefetch + multi-step training

Usage:
    python -m mpx.sdf_pretrain.train --epochs 100 --batch-size 32 --steps-per-epoch 10
    python -m mpx.sdf_pretrain.train --use-curriculum --regenerate-every 50 --prefetch 2
"""

import os
import sys
import argparse
from typing import Dict, Tuple, Optional
from functools import partial
import threading
import queue
import time as time_module

import jax
# 强制使用高精度的 float32 进行矩阵乘法，# 解决 Eikonal 梯度不稳定的 Bug（TF32 精度丢失）
jax.config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import struct

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator
from mpx.sdf_pretrain.data.dynamic_terrain import TerrainConfig
from mpx.sdf_pretrain.models.network import EndToEndSDFNetwork
from mpx.sdf_pretrain.models.losses import (
    sdf_mse_loss, eikonal_loss, eikonal_loss_decoupled,
    truncate_sdf, sdf_mse_loss_weighted
)


def parse_args():
    parser = argparse.ArgumentParser(description="SDF Pretraining with Dynamic Terrain")
    # Data generation
    parser.add_argument('--max-boxes', type=int, default=150,
                        help='Maximum number of terrain boxes')
    parser.add_argument('--terrain-size', type=int, default=5,
                        help='WFC terrain grid size')
    parser.add_argument('--step-height-min', type=float, default=0.04,
                        help='Minimum step height')
    parser.add_argument('--step-height-max', type=float, default=0.12,
                        help='Maximum step height')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default 32, can increase for better GPU utilization)')
    parser.add_argument('--num-queries', type=int, default=128)
    parser.add_argument('--heightmap-size', type=int, default=21)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', action='store_true',
                        help='Enable cosine learning rate decay')
    parser.add_argument('--lr-warmup', type=int, default=100,
                        help='Warmup epochs for learning rate')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Minimum learning rate after decay')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=10,
                        help='Number of training steps per epoch (more = better GPU utilization)')
    parser.add_argument('--mse-weight', type=float, default=1.0)
    parser.add_argument('--mse-scale', type=float, default=10000.0,
                        help='MSE loss scaling factor (10000 = treat SDF as centimeters in loss)')
    parser.add_argument('--eikonal-weight', type=float, default=0.1)
    parser.add_argument('--eikonal-every', type=int, default=5,
                        help='Compute eikonal loss every N steps (expensive)')
    parser.add_argument('--eikonal-points', type=int, default=64,
                        help='Number of uniform samples for decoupled Eikonal loss')
    parser.add_argument('--decoupled-eikonal', action='store_true',
                        help='Use decoupled Eikonal loss (iSDF style, recommended)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='Number of batches to prefetch (0 to disable)')

    # === NEW PARAMETERS ===
    parser.add_argument('--truncation-distance', type=float, default=0.3,
                        help='SDF truncation distance in meters (default 0.3 = ±30cm). '
                             'Network output is bounded to [-T, T]. GT must be clamped to same range.')
    parser.add_argument('--lambda-decay', type=float, default=1.0,
                        help='Distance decay coefficient for weighted MSE loss. '
                             'W = e^(-λ*D). λ=1.0: 0.5m→0.61, 1m→0.37, 2m→0.14. '
                             'λ=2.0 is more aggressive.')
    parser.add_argument('--omega-0', type=float, default=30.0,
                        help='SIREN frequency parameter (default 30.0 as recommended by SIREN paper)')
    # =======================

    # Dynamic terrain
    parser.add_argument('--regenerate-every', type=int, default=50,
                        help='Regenerate terrain every N epochs (0 to disable)')
    parser.add_argument('--use-curriculum', action='store_true',
                        help='Enable curriculum learning for terrain difficulty')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='checkpoints/sdf_pretrain')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (restores epoch and optimizer state)')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Path to checkpoint to load weights only (does NOT restore epoch or optimizer state, starts fresh)')
    return parser.parse_args()


@struct.dataclass
class Metrics:
    """Training metrics."""
    mse: float
    eikonal: float
    total: float
    near_mse: float = 0.0  # MSE for points within 0.5m of origin


class TrainState(train_state.TrainState):
    """Training state with batch_stats for BatchNorm."""
    batch_stats: Dict
    metrics: Metrics


class PrefetchWorker:
    """Background worker for prefetching training batches.

    This runs in a separate thread to generate batches while GPU is training,
    improving GPU utilization by overlapping CPU data generation with GPU computation.
    """

    def __init__(self, generator, batch_size: int, prefetch_size: int = 2):
        """
        Args:
            generator: SDFDynamicGenerator instance
            batch_size: Batch size for generation
            prefetch_size: Number of batches to keep in queue
        """
        self.generator = generator
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.key = None
        self.error: Optional[Exception] = None

    def start(self, key: jax.Array):
        """Start the prefetch worker."""
        self.key = key
        self.stop_event.clear()
        self.error = None
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the prefetch worker."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _worker_loop(self):
        """Worker loop that generates batches."""
        while not self.stop_event.is_set():
            try:
                self.key, subkey = jax.random.split(self.key)
                batch = self.generator.generate_batch(subkey, self.batch_size, verbose=False)
                self.queue.put(batch, block=True, timeout=1.0)
            except queue.Full:
                # Queue is full, wait a bit
                continue
            except Exception as e:
                self.error = e
                break

    def get_batch(self, timeout: float = 30.0) -> Dict:
        """Get a batch from the queue."""
        if self.error:
            raise self.error
        return self.queue.get(block=True, timeout=timeout)

    def update_key(self, key: jax.Array):
        """Update the random key (thread-safe)."""
        self.key = key


def create_train_state(rng, model, learning_rate, input_shape, query_shape,
                       use_lr_decay=False, total_steps=10000, warmup_steps=100, min_lr=1e-5):
    """Create initial training state with batch_stats."""
    dummy_heightmap = jnp.ones(input_shape)
    dummy_queries = jnp.ones(query_shape)

    variables = model.init(rng, dummy_heightmap, dummy_queries, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    if use_lr_decay:
        # Cosine decay with warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=min_lr
        )
        tx = optax.chain(
            optax.clip_by_global_norm(50.0),
            optax.adam(learning_rate=schedule)
        )
        print(f"Using cosine LR decay: peak={learning_rate}, warmup={warmup_steps}, decay_steps={total_steps}, min={min_lr}")
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(50.0),
            optax.adam(learning_rate=learning_rate)
        )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        metrics=Metrics(mse=0.0, eikonal=0.0, total=0.0, near_mse=0.0),
    )
    return state


def load_checkpoint(state: TrainState, checkpoint_path: str) -> Tuple[TrainState, int]:
    """Load params, batch_stats and optimizer state from checkpoint.

    Returns:
        Tuple of (updated state, start_epoch)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = jnp.load(checkpoint_path, allow_pickle=True)

    params = checkpoint['params'].item()
    batch_stats = checkpoint['batch_stats'].item()
    start_epoch = int(checkpoint.get('epoch', 0))

    # 加载优化器状态（如果存在）
    if 'opt_state' in checkpoint:
        opt_state = checkpoint['opt_state'].item()
        state = state.replace(params=params, batch_stats=batch_stats, opt_state=opt_state)
        print("Loaded optimizer state (Adam momentum restored)")
    else:
        state = state.replace(params=params, batch_stats=batch_stats)
        print("No optimizer state found, using fresh Adam state")

    print(f"Checkpoint loaded successfully! Resuming from epoch {start_epoch}")
    return state, start_epoch


def load_weights_only(state: TrainState, checkpoint_path: str) -> TrainState:
    """Load only params and batch_stats from checkpoint (no optimizer state, no epoch).

    Use this to start fresh training with pretrained weights.
    """
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = jnp.load(checkpoint_path, allow_pickle=True)

    params = checkpoint['params'].item()
    batch_stats = checkpoint['batch_stats'].item()

    state = state.replace(params=params, batch_stats=batch_stats)
    print("Weights loaded successfully! Starting fresh training (optimizer state reset).")
    return state


@partial(jax.jit, static_argnames=['compute_eikonal', 'decoupled_eikonal', 'num_eikonal_points'])
def train_step(state: TrainState, batch: Dict, key: jax.random.PRNGKey,
               compute_eikonal: bool = False,
               decoupled_eikonal: bool = False,
               num_eikonal_points: int = 64,
               mse_scale: float = 10000.0,
               eikonal_weight: float = 0.1,
               truncation_distance: float = 0.3,
               lambda_decay: float = 1.0) -> Tuple[TrainState, Dict]:
    """Perform a single training step.

    Args:
        mse_scale: MSE loss scaling factor. With SDF in meters, small errors (e.g., 5mm = 0.005m)
                    produce tiny gradients (0.01). Scaling by 10000x effectively treats
                    the loss as if SDF were in centimeters, providing stronger gradient signals.
        truncation_distance: SDF truncation distance. Both network output and GT are bounded
                            to [-T, T]. This forces network to focus on near-surface region.
        lambda_decay: Distance decay coefficient for weighted MSE. W = e^(-λ*D).
    """
    heightmap = batch['heightmap']
    queries = batch['queries_local']
    targets_raw = batch['sdf']

    def loss_fn(params, batch_stats):
        predictions, new_mutables = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, queries, train=True, mutable=['batch_stats']
        )
        new_batch_stats = new_mutables['batch_stats']

        # === CRITICAL: Truncate GT to match network output range ===
        # Network can only output [-truncation_distance, +truncation_distance]
        # If we don't truncate GT, training will be unstable!
        targets = truncate_sdf(targets_raw, truncation_distance)

        # === Distance-weighted MSE loss ===
        # Near points have higher weight, far points have lower weight
        mse_loss, loss_info = sdf_mse_loss_weighted(
            predictions, targets, queries,
            lambda_decay=lambda_decay,
            epsilon=1e-8
        )
        near_mse = loss_info['near_surface_mse']

        # === Eikonal Loss ===
        if compute_eikonal:
            if decoupled_eikonal:
                eik_loss = eikonal_loss_decoupled(
                    state.apply_fn,
                    params,
                    new_batch_stats,
                    heightmap,
                    key,
                    num_eikonal_points=num_eikonal_points,
                    bounds=(-1.0, 1.0, -1.0, 1.0, -0.5, 0.5),
                    epsilon=1e-3
                )
            else:
                eik_loss = eikonal_loss(
                    state.apply_fn,
                    params,
                    new_batch_stats,
                    heightmap, queries, predictions,
                    epsilon=1e-3,
                    surface_threshold=0.3
                )
        else:
            eik_loss = jnp.array(0.0)

        # Apply MSE scaling: This amplifies small gradients (e.g., 5mm error -> strong signal)
        # Eikonal loss is NOT scaled (physical constraint |∇SDF|=1 is unit-independent)
        total_loss = mse_scale * mse_loss + eikonal_weight * eik_loss
        return total_loss, (mse_loss, eik_loss, near_mse, new_batch_stats, predictions)

    (total_loss, (mse_loss, eik_loss, near_mse, new_batch_stats, predictions)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params, state.batch_stats)

    # Compute gradient norm for monitoring
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, (mse_loss, eik_loss, near_mse, predictions, grad_norm)


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("SDF Pretraining with Dynamic Terrain Generation")
    print("="*60)
    print(f"  Max boxes: {args.max_boxes}")
    print(f"  Terrain size: {args.terrain_size}")
    print(f"  Step height: [{args.step_height_min}, {args.step_height_max}]")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num queries: {args.num_queries}")
    print(f"  Heightmap size: {args.heightmap_size}x{args.heightmap_size}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LR decay: {args.lr_decay}")
    if args.lr_decay:
        print(f"  LR warmup: {args.lr_warmup} epochs")
        print(f"  LR min: {args.lr_min}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {args.steps_per_epoch}")
    print(f"  Eikonal every: {args.eikonal_every} steps")
    print(f"  Eikonal weight: {args.eikonal_weight}")
    print(f"  Decoupled Eikonal: {args.decoupled_eikonal}")
    if args.decoupled_eikonal:
        print(f"  Eikonal sample points: {args.eikonal_points}")
    print(f"  MSE scale: {args.mse_scale}")
    print(f"  Regenerate terrain every: {args.regenerate_every} epochs")
    print(f"  Curriculum learning: {args.use_curriculum}")
    print(f"  Prefetch: {args.prefetch}")
    # === NEW PARAMETERS ===
    print(f"  Truncation distance: {args.truncation_distance}m (±{args.truncation_distance*100:.0f}cm)")
    print(f"  Lambda decay: {args.lambda_decay}")
    print(f"  Omega_0 (SIREN freq): {args.omega_0}")
    # ======================
    print(f"  Seed: {args.seed}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("="*60 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize terrain config
    terrain_config = TerrainConfig(
        size=args.terrain_size,
        step_height_min=args.step_height_min,
        step_height_max=args.step_height_max,
        max_boxes=args.max_boxes,
    )

    # Initialize generator with dynamic terrain
    print("Initializing dynamic terrain generator...")
    generator = SDFDynamicGenerator(
        max_boxes=args.max_boxes,
        heightmap_size=(args.heightmap_size, args.heightmap_size),
        num_queries_per_sample=args.num_queries,
        terrain_config=terrain_config,
        use_curriculum=args.use_curriculum,
    )

    # Generate initial terrain
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    generator.regenerate_terrain(subkey)

    # Initialize model with new parameters
    print("Initializing model...")
    rng = jax.random.PRNGKey(args.seed)
    model = EndToEndSDFNetwork(
        latent_dim=args.latent_dim,
        hidden_dims=(256, 256, 128, 64),  # 4-layer funnel
        pos_encoding_L=6,
        omega_0=args.omega_0,
        truncation_distance=args.truncation_distance,
    )

    input_shape = (args.batch_size, args.heightmap_size, args.heightmap_size, 3)
    query_shape = (args.batch_size, args.num_queries, 3)

    total_steps = args.epochs  # Total training steps
    state = create_train_state(
        rng, model, args.lr, input_shape, query_shape,
        use_lr_decay=args.lr_decay,
        total_steps=total_steps,
        warmup_steps=args.lr_warmup,
        min_lr=args.lr_min
    )

    start_epoch = 0  # 默认从第 0 轮开始

    # 检查 --resume 和 --load-weights 是否同时使用
    if args.resume and args.load_weights:
        print("Error: --resume and --load-weights cannot be used together!")
        return

    # Load checkpoint if resuming (restores epoch and optimizer state)
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(os.path.dirname(__file__), '..', '..', args.resume)
        if not os.path.exists(resume_path):
            print(f"Error: Checkpoint not found: {resume_path}")
            return
        state, start_epoch = load_checkpoint(state, resume_path)

    # Load weights only (fresh training with pretrained weights)
    if args.load_weights:
        weights_path = args.load_weights if os.path.isabs(args.load_weights) else os.path.join(os.path.dirname(__file__), '..', '..', args.load_weights)
        if not os.path.exists(weights_path):
            print(f"Error: Checkpoint not found: {weights_path}")
            return
        state = load_weights_only(state, weights_path)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model initialized with {num_params:,} parameters")

    # Training loop
    print("\nStarting training...")
    key = jax.random.PRNGKey(args.seed)

    best_loss = float('inf')
    best_near_mse = float('inf')
    history = []

    # Initialize prefetch worker if enabled
    prefetch_worker = None
    if args.prefetch > 0:
        print(f"Initializing prefetch worker (buffer size: {args.prefetch})...")
        prefetch_worker = PrefetchWorker(generator, args.batch_size, args.prefetch)
        prefetch_worker.start(key)
        print("Prefetch worker started.\n")

    global_step = 0
    jit_compiled = False

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time_module.time()

            # Curriculum learning: update level
            if args.use_curriculum and args.regenerate_every > 0:
                level = min(4, epoch // (args.epochs // 5))
                generator.set_curriculum_level(level)

            # Regenerate terrain periodically
            if args.regenerate_every > 0 and epoch > 0 and epoch % args.regenerate_every == 0:
                if prefetch_worker:
                    prefetch_worker.stop()
                key, subkey = jax.random.split(key)
                print(f"[Epoch {epoch+1}] Regenerating terrain...")
                sys.stdout.flush()
                generator.regenerate_terrain(subkey)
                if prefetch_worker:
                    prefetch_worker.start(key)

            # Epoch metrics aggregation
            epoch_mse_sum = 0.0
            epoch_eik_sum = 0.0
            epoch_total_sum = 0.0
            epoch_grad_sum = 0.0
            epoch_near_mse_sum = 0.0  # Track near-surface MSE
            valid_steps = 0

            # Multi-step training per epoch
            for step in range(args.steps_per_epoch):
                step_start = time_module.time()

                # Get batch (with prefetch if enabled)
                if prefetch_worker:
                    batch = prefetch_worker.get_batch()
                    key = prefetch_worker.key  # Sync key from worker
                else:
                    if epoch == 0 and step == 0:
                        print(f"[Epoch {epoch+1}] Generating first batch (this may take 1-2 minutes)...")
                        sys.stdout.flush()
                    batch_gen_start = time_module.time()
                    key, subkey = jax.random.split(key)
                    batch = generator.generate_batch(subkey, args.batch_size, verbose=(epoch == 0 and step == 0))
                    if epoch == 0 and step == 0:
                        print(f"[Epoch {epoch+1}] Batch generation completed in {time_module.time() - batch_gen_start:.2f}s")
                        sys.stdout.flush()

                # Debug info for first step of every 10 epochs
                if epoch % 10 == 0 and step == 0:
                    print("\nData statistics:")
                    print(f"  Heightmap shape: {batch['heightmap'].shape}")
                    # 分别统计 X, Y, Z 三个维度的范围
                    hm = batch['heightmap']
                    print(f"  Heightmap X range: [{float(hm[..., 0].min()):.4f}, {float(hm[..., 0].max()):.4f}], mean: {float(hm[..., 0].mean()):.4f}")
                    print(f"  Heightmap Y range: [{float(hm[..., 1].min()):.4f}, {float(hm[..., 1].max()):.4f}], mean: {float(hm[..., 1].mean()):.4f}")
                    print(f"  Heightmap Z range: [{float(hm[..., 2].min()):.4f}, {float(hm[..., 2].max()):.4f}], mean: {float(hm[..., 2].mean()):.4f}")
                    print(f"  SDF range: [{float(batch['sdf'].min()):.4f}, {float(batch['sdf'].max()):.4f}]")
                    print(f"  SDF mean: {float(batch['sdf'].mean()):.4f}")
                    if jnp.any(jnp.isnan(batch['sdf'])):
                        print("  WARNING: NaN detected in SDF values!")
                    print()

                # Determine if we should compute Eikonal loss
                compute_eik = (global_step % args.eikonal_every == 0)

                # Training step
                if not jit_compiled:
                    if compute_eik:
                        print(f"[Epoch {epoch+1}, Step {step+1}] Running first training step with Eikonal loss...")
                        print(f"  - This includes JIT compilation (may take 5-15 minutes)")
                        print(f"  - Eikonal loss requires 6 forward passes (3 axes × 2 directions)")
                    else:
                        print(f"[Epoch {epoch+1}, Step {step+1}] Running first training step...")
                        print(f"  - This includes JIT compilation (may take 2-5 minutes)")
                    sys.stdout.flush()

                train_start = time_module.time()
                # Generate key for Eikonal sampling
                key, eik_key = jax.random.split(key)
                state, (mse_loss, eik_loss, near_mse, predictions, grad_norm) = train_step(
                    state, batch, eik_key,
                    compute_eikonal=compute_eik,
                    decoupled_eikonal=args.decoupled_eikonal,
                    num_eikonal_points=args.eikonal_points,
                    mse_scale=args.mse_scale,
                    eikonal_weight=args.eikonal_weight,
                    truncation_distance=args.truncation_distance,
                    lambda_decay=args.lambda_decay,
                )
                train_time = time_module.time() - train_start

                if not jit_compiled:
                    print(f"[Epoch {epoch+1}, Step {step+1}] First training step completed in {train_time:.2f}s")
                    print(f"[Epoch {epoch+1}] JIT compilation done! Subsequent steps will be faster.")
                    sys.stdout.flush()
                    jit_compiled = True

                # Check for NaN in losses
                mse_val = float(mse_loss)
                eik_val = float(eik_loss)
                near_mse_val = float(near_mse)
                grad_norm_val = float(grad_norm)

                if np.isnan(mse_val) or np.isnan(eik_val):
                    print(f"\n!!! NaN detected at epoch {epoch+1}, step {step+1} !!!")
                    print(f"  MSE: {mse_val}, Eikonal: {eik_val}")
                    print("  Skipping this step...")
                    if prefetch_worker:
                        prefetch_worker.stop()
                        key, subkey = jax.random.split(key)
                        generator.regenerate_terrain(subkey)
                        prefetch_worker.start(key)
                    continue

                total_val = mse_val * args.mse_scale + args.eikonal_weight * eik_val

                # Accumulate metrics
                epoch_mse_sum += mse_val
                epoch_eik_sum += eik_val
                epoch_total_sum += total_val
                epoch_grad_sum += grad_norm_val
                epoch_near_mse_sum += near_mse_val
                valid_steps += 1
                global_step += 1

            # Calculate epoch average metrics
            if valid_steps > 0:
                avg_mse = epoch_mse_sum / valid_steps
                avg_eik = epoch_eik_sum / valid_steps
                avg_total = epoch_total_sum / valid_steps
                avg_grad = epoch_grad_sum / valid_steps
                avg_near_mse = epoch_near_mse_sum / valid_steps
            else:
                avg_mse = avg_eik = avg_total = avg_grad = avg_near_mse = float('nan')

            # Track best loss
            if not np.isnan(avg_total) and avg_total < best_loss:
                best_loss = avg_total
            if not np.isnan(avg_near_mse) and avg_near_mse < best_near_mse:
                best_near_mse = avg_near_mse

            loss_dict = {
                'mse': avg_mse,
                'eikonal': avg_eik,
                'total': avg_total,
                'near_mse': avg_near_mse,
            }
            history.append(loss_dict)

            # Calculate epoch time
            epoch_time = time_module.time() - epoch_start_time
            steps_per_sec = valid_steps / epoch_time if epoch_time > 0 else 0

            # Print progress with monitoring info
            # NOTE: near_mse is the CORE METRIC - must be < 0.001 (1mm error) for good control
            print(f"Epoch {epoch+1:4d}/{args.epochs} - "
                  f"MSE: {avg_mse:.6f} | "
                  f"Near: {avg_near_mse:.6f} | "  # Near-surface MSE (< 0.5m)
                  f"Eik: {avg_eik:.6f} | "
                  f"Total: {avg_total:.6f} | "
                  f"Grad: {avg_grad:.4f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s | "
                  f"Time: {epoch_time:.2f}s")

            sys.stdout.flush()

            # Save intermediate checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                intermediate_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.npz')
                jnp.savez(
                    intermediate_path,
                    params=state.params,
                    batch_stats=state.batch_stats,
                    opt_state=state.opt_state,
                    epoch=epoch + 1,
                    loss=avg_total,
                    near_mse=avg_near_mse,
                )
                print(f"  [Saved intermediate checkpoint: {intermediate_path}]")
                sys.stdout.flush()

    finally:
        # Stop prefetch worker
        if prefetch_worker:
            prefetch_worker.stop()

    # Summary
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final MSE Loss: {mse_val:.6f}")
    print(f"Final Near-Surface MSE: {near_mse_val:.6f}")
    print(f"Final Total Loss: {total_val:.6f}")
    print(f"Best Total Loss: {best_loss:.6f}")
    print(f"Best Near-Surface MSE: {best_near_mse:.6f}")
    print("="*60 + "\n")

    # Save final checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'final_checkpoint.npz')
    jnp.savez(
        checkpoint_path,
        params=state.params,
        batch_stats=state.batch_stats,
        opt_state=state.opt_state,
        near_mse=avg_near_mse,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
