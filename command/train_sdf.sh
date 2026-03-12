#!/bin/bash
# SDF 预训练启动脚本
# 王甫12138

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpx_env

# ==================== 训练参数配置 ====================
# 数据相关
XML_PATH="mpx/data/r2-1024/mjcf/scene_terrain_test.xml"
BATCH_SIZE=128           # GPU目前算力空闲，翻倍Batch Size进一步榨干GPU
NUM_QUERIES=2048         # 大量增加每个样本的采样点(2048)，让每个场景的梯度更加丰富
HEIGHTMAP_SIZE=21

# 模型相关
LATENT_DIM=256

# 优化相关
EPOCHS=10000             # 大规模预训练需要极大的Epoch数量
LR=1e-4
MSE_WEIGHT=1.0
EIKONAL_WEIGHT=0.1

# 训练策略
EIKONAL_EVERY=1          # 建议每轮都计算平滑度惩罚，防止SDF场出现局部坑洞
RANDOMIZE_EVERY=5        # 高频随机化地形(每5轮)，逼迫网络学习真正的空间几何而不是死记硬背

# 其他
SEED=42
OUTPUT_DIR="checkpoints/sdf_pretrain"
RESUME=""                # 恢复训练的checkpoint路径，如 "checkpoints/sdf_pretrain/final_checkpoint.npz"

# ==================== 打印配置 ====================
echo "=========================================="
echo "SDF Pretraining - Started at $(date)"
echo "=========================================="
echo "Terrain XML: $XML_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Num Queries: $NUM_QUERIES"
echo "Heightmap Size: ${HEIGHTMAP_SIZE}x${HEIGHTMAP_SIZE}"
echo "Latent Dim: $LATENT_DIM"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "MSE Weight: $MSE_WEIGHT"
echo "Eikonal Weight: $EIKONAL_WEIGHT"
echo "Eikonal Every: $EIKONAL_EVERY epochs"
echo "Randomize Every: $RANDOMIZE_EVERY epochs"
echo "Seed: $SEED"
echo "Output Dir: $OUTPUT_DIR"
[ -n "$RESUME" ] && echo "Resume From: $RESUME"
echo "=========================================="

# ==================== 构建命令 ====================
CMD="python -m mpx.sdf_pretrain.train \
    --xml $XML_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-queries $NUM_QUERIES \
    --heightmap-size $HEIGHTMAP_SIZE \
    --latent-dim $LATENT_DIM \
    --lr $LR \
    --mse-weight $MSE_WEIGHT \
    --eikonal-weight $EIKONAL_WEIGHT \
    --eikonal-every $EIKONAL_EVERY \
    --randomize-every $RANDOMIZE_EVERY \
    --seed $SEED \
    --output-dir $OUTPUT_DIR"

# 添加恢复参数（如果指定）
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# ==================== 执行训练 ====================
eval $CMD

echo "=========================================="
echo "Training Complete - $(date)"
echo "=========================================="
