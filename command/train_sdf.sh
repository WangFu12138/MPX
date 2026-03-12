#!/bin/bash
# SDF 预训练启动脚本 - 动态地形版本
# 王甫12138

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpx_env

# ==================== 地形配置 ====================
MAX_BOXES=150            # 最大地形盒子数
TERRAIN_SIZE=5           # WFC 地形网格大小
STEP_HEIGHT_MIN=0.04     # 最小台阶高度 (米)
STEP_HEIGHT_MAX=0.12     # 最大台阶高度 (米)

# ==================== 训练参数配置 ====================
# 数据相关
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
REGENERATE_EVERY=50      # 每50轮重新生成地形，逼迫网络学习真正的空间几何
USE_CURRICULUM=false     # 是否启用课程学习（逐步增加地形难度）

# 其他
SEED=42
OUTPUT_DIR="checkpoints/sdf_pretrain"
RESUME=""                # 恢复训练的checkpoint路径，如 "checkpoints/sdf_pretrain/final_checkpoint.npz"

# ==================== 打印配置 ====================
echo "=========================================="
echo "SDF Pretraining (Dynamic Terrain) - $(date)"
echo "=========================================="
echo "Terrain Config:"
echo "  Max Boxes: $MAX_BOXES"
echo "  Terrain Size: $TERRAIN_SIZE x $TERRAIN_SIZE"
echo "  Step Height: [$STEP_HEIGHT_MIN, $STEP_HEIGHT_MAX] m"
echo ""
echo "Training Config:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Queries: $NUM_QUERIES"
echo "  Heightmap Size: ${HEIGHTMAP_SIZE}x${HEIGHTMAP_SIZE}"
echo "  Latent Dim: $LATENT_DIM"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  MSE Weight: $MSE_WEIGHT"
echo "  Eikonal Weight: $EIKONAL_WEIGHT"
echo "  Eikonal Every: $EIKONAL_EVERY epochs"
echo "  Regenerate Every: $REGENERATE_EVERY epochs"
echo "  Curriculum: $USE_CURRICULUM"
echo "  Seed: $SEED"
echo "  Output Dir: $OUTPUT_DIR"
[ -n "$RESUME" ] && echo "  Resume From: $RESUME"
echo "=========================================="

# ==================== 构建命令 ====================
CMD="python -m mpx.sdf_pretrain.train \
    --max-boxes $MAX_BOXES \
    --terrain-size $TERRAIN_SIZE \
    --step-height-min $STEP_HEIGHT_MIN \
    --step-height-max $STEP_HEIGHT_MAX \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-queries $NUM_QUERIES \
    --heightmap-size $HEIGHTMAP_SIZE \
    --latent-dim $LATENT_DIM \
    --lr $LR \
    --mse-weight $MSE_WEIGHT \
    --eikonal-weight $EIKONAL_WEIGHT \
    --eikonal-every $EIKONAL_EVERY \
    --regenerate-every $REGENERATE_EVERY \
    --seed $SEED \
    --output-dir $OUTPUT_DIR"

# 添加课程学习（如果启用）
if [ "$USE_CURRICULUM" = true ]; then
    CMD="$CMD --use-curriculum"
fi

# 添加恢复参数（如果指定）
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# ==================== 执行训练 ====================
eval $CMD

echo "=========================================="
echo "Training Complete - $(date)"
echo "=========================================="
