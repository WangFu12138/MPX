"""
Offline Dataset Generator for VAE Training
将高程图批量提取并存为 HDF5 格式，完美适配 Two-Stage 训练的第一阶段。
"""

import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from tqdm import tqdm

from mpx.sdf_pretrain.data.dataset_generator import SDFDynamicGenerator

def create_offline_dataset(
    output_path: str,
    total_samples: int = 50000,
    batch_size: int = 100,
    heightmap_size: tuple = (128, 128),
    regenerate_every: int = 500,
    seed: int = 42
):
    """
    生成纯高程图离线数据集 (Bypass SDF computation for ultra-fast generation)
    """
    print(f"🚀 启动离线数据集生成 pipeline...")
    print(f"📦 目标样本数: {total_samples}")
    print(f"🗺️  高程图尺寸: {heightmap_size}")
    print(f"💾 输出路径: {output_path}")

    # 1. 初始化生成器 (我们不需要很高的 SDF query 数量，因为我们根本不算它)
    generator = SDFDynamicGenerator(
        max_boxes=200,
        heightmap_size=heightmap_size,
        num_queries_per_sample=2, # 设置为极小值以节省显存分配
        use_curriculum=False      # 预训练 VAE 需要最丰富的地形，不需要课程学习
    )

    key = jax.random.PRNGKey(seed)
    
    # 2. 准备 HDF5 文件
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 提前计算 shape
    H, W = heightmap_size
    
    with h5py.File(output_path, 'w') as f:
        # 创建可动态扩展的 HDF5 数据集
        # VAE 需要的是 1 通道的数据，因此 shape 为 (N, 1, H, W)
        ds_heightmaps = f.create_dataset(
            'heightmaps', 
            shape=(total_samples, 1, H, W), 
            dtype=np.float32,
            compression='lzf' # lzf 压缩解压速度极快
        )
        ds_poses = f.create_dataset(
            'poses', 
            shape=(total_samples, 6), 
            dtype=np.float32
        )
        
        num_batches = total_samples // batch_size
        samples_generated = 0
        
        # 初始生成一次地形
        key, subkey = jax.random.split(key)
        generator.regenerate_terrain(subkey)
        
        # 定义一个纯净的 JIT 函数只生成高程图
        # @jax.jit
        def generate_heightmap_batch(rng, poses):
            # 获取局部坐标系下的高程图 (B, H, W, 3)
            hms = generator.generate_heightmaps(rng, poses)
            # 物理截断 [-1.0, 1.0] 米
            hms_clipped = jnp.clip(hms, -1.0, 1.0)
            # 提取 Z 通道 (B, H, W)
            z_channel = hms_clipped[..., 2]
            
            # 【核心逻辑】：映射到 [0, 1] 以适配 VAE 的 Sigmoid!
            # -1.0 -> 0.0 (深坑)
            #  0.0 -> 0.5 (平地)
            #  1.0 -> 1.0 (高台阶)
            z_norm = (z_channel + 1.0) / 2.0
            
            # 增加通道维度使其变成 (B, 1, H, W)
            return jnp.expand_dims(z_norm, axis=1)

        # 开始批量生成
        start_time = time.time()
        pbar = tqdm(total=total_samples, desc="Generating Dataset")
        
        for i in range(num_batches):
            # 每隔 N 个样本换一次随机地形，保证数据集多样性
            if i > 0 and (i * batch_size) % regenerate_every == 0:
                key, subkey = jax.random.split(key)
                generator.regenerate_terrain(subkey)
                
            key, key_pose, key_hm = jax.random.split(key, 3)
            
            # 采样 LiDAR 视角
            poses = generator.sample_lidar_poses(key_pose, batch_size)
            
            # 生成归一化后的高程图
            hms_batch = generate_heightmap_batch(key_hm, poses)
            
            # 写入 HDF5 (从 GPU Tensor 转回 Numpy)
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            ds_heightmaps[start_idx:end_idx] = np.array(hms_batch)
            ds_poses[start_idx:end_idx] = np.array(poses)
            
            samples_generated += batch_size
            pbar.update(batch_size)
            
        pbar.close()
        
        print(f"✅ 数据集生成完毕! 总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
        print(f"📊 文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"验证形状 - Heightmaps: {ds_heightmaps.shape}, Poses: {ds_poses.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/vae_dataset/terrain_heightmaps.h5")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--regenerate_every", type=int, default=500, help="多少个样本换一次地形")
    parser.add_argument("--size", type=int, default=128, help="高程图分辨率(HxW)")
    args = parser.parse_args()
    
    create_offline_dataset(
        output_path=args.output,
        total_samples=args.samples,
        batch_size=args.batch_size,
        heightmap_size=(args.size, args.size),
        regenerate_every=args.regenerate_every
    )