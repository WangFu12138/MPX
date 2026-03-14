"""
VAE 视觉效果验证脚本
对比: 真实高程图 vs 潜变量压缩后还原的高程图
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import serialization

from mpx.sdf_pretrain.models.vae import HeightmapVAE, VAEEncoder, HeightmapDecoder

def main():
    dataset_path = "data/vae_dataset/terrain_heightmaps.h5"
    ckpt_path = "checkpoints/vae/vae_best_encoder.npz"

    print("🛠️ 正在加载验证数据...")
    with h5py.File(dataset_path, 'r') as f:
        # 随便挑 4 张地形图看看
        sample_indices = [100, 5000, 15000, 40000]
        real_data = f['heightmaps'][sample_indices]
        real_data = np.transpose(real_data, (0, 2, 3, 1)) # NCHW -> NHWC

    print("🧠 正在加载训练好的 VAE 大脑...")
    model = HeightmapVAE()
    rng = jax.random.PRNGKey(0)
    # 初始化一个空参数外壳
    dummy_x = jnp.ones((1, 128, 128, 1), jnp.float32)
    variables = model.init(rng, dummy_x, rng)
    
    # 注入训练好的权重
    with open(ckpt_path, "rb") as f:
        params = serialization.from_bytes(variables['params'], f.read())

    print("🎨 正在让画师(Decoder)重新默写地形...")
# 我们只用 encode 提取纯净特征 (不要加噪声)，然后再 decode
    # 这一步模拟了将来 NMPC 控制器真实的使用场景
    
    # 单独实例化情报员和画师
    encoder = VAEEncoder(latent_dim=90)
    decoder = HeightmapDecoder()
    
    # 给他们各自发配对应的权重参数
    mean_latent, _ = encoder.apply({'params': params['encoder']}, real_data)
    recon_data = decoder.apply({'params': params['decoder']}, mean_latent)
    # 画图对比！
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i in range(4):
        # 恢复物理量纲 (0~1 变回 -1~1 米)
        orig = real_data[i, :, :, 0] * 2.0 - 1.0
        recon = recon_data[i, :, :, 0] * 2.0 - 1.0
        diff = np.abs(orig - recon)
        
        # 1. 原图
        im1 = axes[i, 0].imshow(orig, cmap='terrain', vmin=-0.5, vmax=0.5)
        axes[i, 0].set_title(f"Original Heightmap {i+1}")
        axes[i, 0].axis('off')
        
        # 2. 重建图
        im2 = axes[i, 1].imshow(recon, cmap='terrain', vmin=-0.5, vmax=0.5)
        axes[i, 1].set_title(f"VAE Reconstructed {i+1}")
        axes[i, 1].axis('off')
        
        # 3. 误差图 (热力图)
        im3 = axes[i, 2].imshow(diff, cmap='Reds', vmin=0, vmax=0.15)
        axes[i, 2].set_title(f"Error Map (Max {np.max(diff):.3f}m)")
        axes[i, 2].axis('off')
        
        fig.colorbar(im3, ax=axes[i, 2], shrink=0.8, label="Error (meters)")

    plt.tight_layout()
    plt.savefig("vae_reconstruction_results.png", dpi=200)
    print("✅ 对比图已保存至 vae_reconstruction_results.png！")

if __name__ == "__main__":
    main()