"""
Stage 1: VAE Training Script
读取 HDF5 离线高程图数据集，训练变分自编码器，并将高维图像压缩为 90 维的 Latent Vector。
"""
from tqdm import tqdm
import os
import time
import argparse
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization

from mpx.sdf_pretrain.models.vae import HeightmapVAE
from mpx.sdf_pretrain.models.losses import vae_loss_fn

# def get_hdf5_batches(h5_path, batch_size, shuffle=True, seed=0):
#     """
#     高效的 HDF5 数据加载器
#     包含 I/O 优化：先排序索引进行连续读取，再在内存中打乱，极大提升读取速度
#     """
#     with h5py.File(h5_path, 'r') as f:
#         data = f['heightmaps']
#         num_samples = data.shape[0]
#         indices = np.arange(num_samples)
        
#         if shuffle:
#             rng = np.random.default_rng(seed)
#             rng.shuffle(indices)
            
#         for i in range(0, num_samples, batch_size):
#             batch_indices = indices[i:i+batch_size]
            
#             # 【I/O 优化】HDF5 连续读取速度远快于随机读取，先排序，读出来再按原顺序恢复
#             sort_args = np.argsort(batch_indices)
#             sorted_indices = batch_indices[sort_args]
            
#             # 读取数据
#             batch_data = data[sorted_indices]
            
#             # 恢复随机打乱的顺序
#             restore_args = np.argsort(sort_args)
#             batch_data = batch_data[restore_args]
            
#             # 【格式转换】从 PyTorch 的 NCHW (B, 1, H, W) 转为 JAX 的 NHWC (B, H, W, 1)
#             batch_data = np.transpose(batch_data, (0, 2, 3, 1))
#             yield batch_data

class VAETrainState(train_state.TrainState):
    """扩展标准的 TrainState，用于在训练步骤中传递随机数钥匙 (供重参数化使用)"""
    key: jax.Array

@jax.jit
def train_step(state, batch, kl_weight):
    """单个训练步的 JIT 编译函数"""
    # 拆分出一个新的随机数 key 用于这一步的噪声采样
    key, subkey = jax.random.split(state.key)
    
    def loss_fn(params):
        # 前向传播 (传入 subkey)
        recon_x, mean, logvar = HeightmapVAE().apply({'params': params}, batch, subkey)
        # 计算 Loss
        total_loss, recon_loss, kld_loss = vae_loss_fn(recon_x, batch, mean, logvar, kl_weight)
        return total_loss, (recon_loss, kld_loss)
    
    # 自动求导
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = grad_fn(state.params)
    recon_loss, kld_loss = aux
    
    # 更新优化器参数
    new_state = state.apply_gradients(grads=grads)
    # 更新 state 中的 key
    new_state = new_state.replace(key=key)
    
    return new_state, total_loss, recon_loss, kld_loss

def main(args):
    print("==========================================")
    print("🚀 Stage 1: Starting VAE Pretraining")
    print(f"📦 Dataset: {args.dataset}")
    print(f"⚙️  Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"🧠 Latent Dim: 90 | KL Weight: {args.kl_weight}")
    print("==========================================\n")

# 1. 检查并加载全量数据集到内存 (告别硬盘 I/O 瓶颈！)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"找不到数据集 {args.dataset}...")
        
    print(f"⏳ 正在将 3GB 数据集全部加载到内存，请稍候...")
    with h5py.File(args.dataset, 'r') as f:
        # 一次性读出所有数据到 Numpy
        all_data_np = f['heightmaps'][:]
        total_samples = all_data_np.shape[0]
        steps_per_epoch = total_samples // args.batch_size
        
    print(f"🔄 正在转换张量格式 (NCHW -> NHWC)...")
    all_data_np = np.transpose(all_data_np, (0, 2, 3, 1))
    print(f"✅ 数据加载完毕！形状: {all_data_np.shape}")

    # 2. 初始化网络与优化器
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, state_rng = jax.random.split(rng, 3)
    
    model = HeightmapVAE()
    # 伪造一个输入形状用于初始化网络参数 (B, H, W, 1)
    dummy_x = jnp.ones((1, args.size, args.size, 1), jnp.float32)
    variables = model.init(init_rng, dummy_x, init_rng)
    
    # 我们使用带有 Warmup 和 Cosine Decay 的学习率调度，保证重建质量
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=args.lr,
        warmup_steps=5 * steps_per_epoch,
        decay_steps=args.epochs * steps_per_epoch,
        end_value=1e-6
    )
    tx = optax.adam(learning_rate=schedule)
    
    state = VAETrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        key=state_rng
    )
# 3. 训练循环
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    print("⏳ 正在进行 JAX 网络初始化与 XLA 编译 (预计需要 30-60 秒，请耐心等待)...")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        
        # 💥 核心修改 1：每轮训练前，在内存中打乱所有数据的索引
        indices = np.arange(total_samples)
        np.random.default_rng(args.seed + epoch).shuffle(indices)
        
        # 💥 核心修改 2：进度条直接遍历步数
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch:03d}/{args.epochs}")
        
        for step in pbar:
            # 💥 核心修改 3：通过索引切片，直接从内存数组中光速捞取 Batch
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_np = all_data_np[batch_idx]
            
            batch_jnp = jnp.array(batch_np)
            
            # 第一步会卡顿编译，后面的步数起飞
            state, total_loss, recon_loss, kld_loss = train_step(state, batch_jnp, args.kl_weight)
            
            epoch_total_loss += total_loss
            epoch_recon_loss += recon_loss
            epoch_kld_loss += kld_loss
            
            # 实时更新进度条后缀显示 Loss
            pbar.set_postfix({
                'Loss': f"{total_loss:.4f}", 
                'Recon': f"{recon_loss:.4f}",
                'KL': f"{kld_loss:.4f}"
            })
            
        avg_total = epoch_total_loss / steps_per_epoch
        avg_recon = epoch_recon_loss / steps_per_epoch
        avg_kld = epoch_kld_loss / steps_per_epoch
        epoch_time = time.time() - start_time
        
        print(f"👉 Epoch {epoch:03d} 总结 | "
              f"Total: {avg_total:.5f} | "
              f"Recon MSE: {avg_recon:.5f} | "
              f"KL: {avg_kld:.5f} | "
              f"耗时: {epoch_time:.2f}s")
              
        # 4. 保存最佳 Checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            save_path = os.path.join(args.output_dir, "vae_best_encoder.npz")
            with open(save_path, "wb") as f:
                f.write(serialization.to_bytes(state.params))
            print(f"  [+] Best model saved! (Total Loss: {best_loss:.5f})")
    print("\n✅ VAE 训练完全结束！")
    print(f"🎉 权重已保存至: {os.path.join(args.output_dir, 'vae_best_encoder.npz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/vae_dataset/terrain_heightmaps.h5")
    parser.add_argument("--output_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--epochs", type=int, default=300) # 通常 VAE 100个 Epoch 就能收敛得极好
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    # KL 权重设小一点(1e-4)，防止 KL 塌陷导致 90 维特征全部变成纯噪声 0
    parser.add_argument("--kl_weight", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)