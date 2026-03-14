#!/usr/bin/env python3
"""Test script for SDF prediction accuracy.

Usage:
    python -m mpx.sdf_pretrain.test_sdf_prediction --checkpoint checkpoints/sdf_pretrain/checkpoint_epoch_15900.npz
"""

import os
import sys
import argparse
import time
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator
from mpx.sdf_pretrain.data.dynamic_terrain import TerrainConfig
from mpx.sdf_pretrain.models.network import EndToEndSDFNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Test SDF Prediction Accuracy")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--num-terrains', type=int, default=5,
                        help='Number of test terrains to generate')
    parser.add_argument('--num-queries', type=int, default=1000,
                        help='Number of query points per terrain')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--heightmap-size', type=int, default=21)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed (different from training)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    return parser.parse_args()


def load_model(checkpoint_path: str, latent_dim: int, heightmap_size: int):
    """Load model and checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Create model
    model = EndToEndSDFNetwork(
        latent_dim=latent_dim,
        hidden_dims=(256, 256, 128)
    )

    # Initialize model to get structure
    rng = jax.random.PRNGKey(0)
    dummy_heightmap = jnp.ones((1, heightmap_size, heightmap_size, 3))
    dummy_queries = jnp.ones((1, 100, 3))

    variables = model.init(rng, dummy_heightmap, dummy_queries, train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # Load checkpoint
    checkpoint = jnp.load(checkpoint_path, allow_pickle=True)
    params = checkpoint['params'].item()
    batch_stats = checkpoint['batch_stats'].item()

    print(f"Checkpoint loaded successfully!")
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model has {num_params:,} parameters")

    return model, params, batch_stats


def predict_sdf(model, params, batch_stats, heightmap, queries, batch_size=4):
    """Predict SDF values for given heightmap and queries."""
    num_queries = queries.shape[1]

    # Process in batches to avoid OOM
    all_predictions = []

    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        batch_queries = queries[:, i:end_idx, :]

        predictions = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            heightmap, batch_queries,
            train=False  # Inference mode
        )
        all_predictions.append(predictions)

    return jnp.concatenate(all_predictions, axis=1)


def compute_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict:
    """Compute prediction metrics."""
    errors = jnp.abs(predictions - targets)

    mae = float(jnp.mean(errors))
    mse = float(jnp.mean((predictions - targets) ** 2))
    rmse = float(jnp.sqrt(mse))
    max_error = float(jnp.max(errors))

    # Error distribution
    pct_lt_1cm = float(jnp.mean(errors < 0.01) * 100)
    pct_lt_2cm = float(jnp.mean(errors < 0.02) * 100)
    pct_lt_5cm = float(jnp.mean(errors < 0.05) * 100)
    pct_lt_10cm = float(jnp.mean(errors < 0.10) * 100)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_error': max_error,
        'pct_lt_1cm': pct_lt_1cm,
        'pct_lt_2cm': pct_lt_2cm,
        'pct_lt_5cm': pct_lt_5cm,
        'pct_lt_10cm': pct_lt_10cm,
    }


def visualize_results(all_results: List[Dict], output_path: str = "sdf_test_results.png"):
    """Visualize test results."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: MAE per terrain
        ax1 = axes[0, 0]
        terrain_ids = [f"Terrain {i+1}" for i in range(len(all_results))]
        maes = [r['mae'] * 100 for r in all_results]  # Convert to cm
        ax1.bar(terrain_ids, maes, color='steelblue')
        ax1.set_ylabel('MAE (cm)')
        ax1.set_title('Mean Absolute Error per Terrain')
        ax1.axhline(y=2, color='r', linestyle='--', label='Target: 2cm')
        ax1.legend()

        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        categories = ['<1cm', '<2cm', '<5cm', '<10cm']
        avg_pcts = [
            np.mean([r['pct_lt_1cm'] for r in all_results]),
            np.mean([r['pct_lt_2cm'] for r in all_results]),
            np.mean([r['pct_lt_5cm'] for r in all_results]),
            np.mean([r['pct_lt_10cm'] for r in all_results]),
        ]
        ax2.bar(categories, avg_pcts, color='forestgreen')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Error Distribution (Average)')
        ax2.set_ylim(0, 100)

        # Plot 3: RMSE per terrain
        ax3 = axes[1, 0]
        rmses = [r['rmse'] * 100 for r in all_results]  # Convert to cm
        ax3.bar(terrain_ids, rmses, color='coral')
        ax3.set_ylabel('RMSE (cm)')
        ax3.set_title('Root Mean Square Error per Terrain')
        ax3.axhline(y=2, color='r', linestyle='--', label='Target: 2cm')
        ax3.legend()

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = (
            f"=== Summary Statistics ===\n\n"
            f"Average MAE: {np.mean([r['mae'] for r in all_results])*100:.2f} cm\n"
            f"Average RMSE: {np.mean([r['rmse'] for r in all_results])*100:.2f} cm\n"
            f"Worst MAE: {max([r['mae'] for r in all_results])*100:.2f} cm\n"
            f"Best MAE: {min([r['mae'] for r in all_results])*100:.2f} cm\n\n"
            f"=== Error Distribution ===\n"
            f"<1cm: {np.mean([r['pct_lt_1cm'] for r in all_results]):.1f}%\n"
            f"<2cm: {np.mean([r['pct_lt_2cm'] for r in all_results]):.1f}%\n"
            f"<5cm: {np.mean([r['pct_lt_5cm'] for r in all_results]):.1f}%\n"
            f"<10cm: {np.mean([r['pct_lt_10cm'] for r in all_results]):.1f}%\n"
        )
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\nVisualization saved to: {output_path}")

    except ImportError:
        print("\nmatplotlib not available, skipping visualization")


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("SDF Prediction Accuracy Test")
    print("="*60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Num terrains: {args.num_terrains}")
    print(f"  Num queries per terrain: {args.num_queries}")
    print(f"  Random seed: {args.seed}")
    print("="*60 + "\n")

    # Load model
    model, params, batch_stats = load_model(
        args.checkpoint,
        args.latent_dim,
        args.heightmap_size
    )

    # Create terrain generator with different seed
    terrain_config = TerrainConfig(
        size=7,
        step_height_min=0.04,
        step_height_max=0.12,
        max_boxes=200,
    )

    generator = SDFDynamicGenerator(
        max_boxes=200,
        heightmap_size=(args.heightmap_size, args.heightmap_size),
        num_queries_per_sample=args.num_queries,
        terrain_config=terrain_config,
        use_curriculum=False,
    )

    # Test on multiple terrains
    key = jax.random.PRNGKey(args.seed)
    all_results = []

    print("\nRunning tests...")
    print("-" * 80)

    for terrain_idx in range(args.num_terrains):
        # Generate new terrain
        key, subkey = jax.random.split(key)
        generator.regenerate_terrain(subkey)

        # Generate test batch (single sample)
        key, subkey = jax.random.split(key)
        batch = generator.generate_batch(subkey, batch_size=1, verbose=False)

        heightmap = batch['heightmap']
        queries = batch['queries_local']
        targets = batch['sdf']

        # Predict
        start_time = time.time()
        predictions = predict_sdf(
            model, params, batch_stats,
            heightmap, queries,
            batch_size=args.batch_size
        )
        inference_time = time.time() - start_time

        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        all_results.append(metrics)

        # Print results
        print(f"Terrain {terrain_idx+1}:")
        print(f"  MAE: {metrics['mae']*100:.2f} cm | RMSE: {metrics['rmse']*100:.2f} cm | Max: {metrics['max_error']*100:.2f} cm")
        print(f"  Error <1cm: {metrics['pct_lt_1cm']:.1f}% | <2cm: {metrics['pct_lt_2cm']:.1f}% | <5cm: {metrics['pct_lt_5cm']:.1f}% | <10cm: {metrics['pct_lt_10cm']:.1f}%")
        print(f"  Inference time: {inference_time*1000:.1f}ms for {args.num_queries} queries")
        print("-" * 80)

    # Summary
    print("\n" + "="*60)
    print("=== SUMMARY ===")
    print("="*60)
    avg_mae = np.mean([r['mae'] for r in all_results]) * 100
    avg_rmse = np.mean([r['rmse'] for r in all_results]) * 100
    avg_pct_lt_2cm = np.mean([r['pct_lt_2cm'] for r in all_results])

    print(f"\nAverage MAE: {avg_mae:.2f} cm")
    print(f"Average RMSE: {avg_rmse:.2f} cm")
    print(f"Points with error <2cm: {avg_pct_lt_2cm:.1f}%")

    print("\n=== Assessment ===")
    if avg_mae < 2.0:
        print("✅ EXCELLENT: Average error <2cm, ready for MPC integration")
    elif avg_mae < 5.0:
        print("⚠️  ACCEPTABLE: Average error <5cm, may need fine-tuning")
    else:
        print("❌ NEEDS IMPROVEMENT: Average error >5cm, continue training")

    print("="*60 + "\n")

    # Visualization
    if args.visualize:
        output_dir = os.path.dirname(args.checkpoint) or '.'
        visualize_results(all_results, os.path.join(output_dir, "sdf_test_results.png"))


if __name__ == "__main__":
    main()
