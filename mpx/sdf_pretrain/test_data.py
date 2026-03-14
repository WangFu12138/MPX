"""
离线高程图数据集可视化脚本
随机抽取 HDF5 数据集中的高程图，并将其还原为物理尺寸 (米) 进行可视化。
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    dataset_path = "data/vae_dataset/terrain_heightmaps.h5"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件：{dataset_path}")

    print("🛠️ 正在打开数据集...")
    with h5py.File(dataset_path, 'r') as f:
        total_samples = f['heightmaps'].shape[0]
        data_shape = f['heightmaps'].shape
        print(f"📊 数据集总览 - 样本数: {total_samples}, 形状: {data_shape}")
        
        # 随机抽取 16 张地形图展示 (4x4 网格)
        np.random.seed(42) # 固定种子，保证每次看的一样
        sample_indices = np.random.choice(total_samples, 16, replace=False)
        # 💥 修复报错：h5py 强制要求切片索引必须是从小到大排序的
        sample_indices = np.sort(sample_indices)
        # 读取数据 (N, C, H, W)
        raw_data = f['heightmaps'][sample_indices]

    print("🔄 正在还原物理量纲并渲染...")
    # 格式转换 NCHW (16, 1, 21, 21) -> NHWC (16, 21, 21, 1)
    raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    
    # 【核心逻辑】：撤销归一化，把 [0, 1] 映射回真实的 [-1.0, 1.0] 米
    physical_hms = raw_data * 2.0 - 1.0

    # 开始画图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    # 取一个全局的颜色映射范围，假设起伏在正负 0.3 米左右比较好看
    # 你可以根据实际地形的高度调整 vmin 和 vmax
    vmin, vmax = -0.3, 0.3 
    
    for i, ax in enumerate(axes.flatten()):
        # 取出第 i 张图的单通道 (21, 21)
        hm2d = physical_hms[i, :, :, 0]
        
        # 画热力图
        im = ax.imshow(hm2d, cmap='terrain', vmin=vmin, vmax=vmax)
        ax.set_title(f"Sample #{sample_indices[i]}\n(Max: {np.max(hm2d):.2f}m, Min: {np.min(hm2d):.2f}m)", fontsize=10)
        ax.axis('off')
        
    # 添加一个全局的颜色条 (Colorbar)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Height (meters)")

    output_file = "generated_data_viz.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✅ 可视化网格图已保存至 {output_file}！")

if __name__ == "__main__":
    main()