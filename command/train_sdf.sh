#!/bin/bash
# SDF 预训练启动脚本 - 动态地形版本
# 王甫12138

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpx_env

# ==================== 地形配置 ====================
MAX_BOXES=200            # 最大地形盒子数（增加到200以适应更大的地形）
TERRAIN_SIZE=7           # WFC 地形网格大小（增大到7x7，更多变化）
STEP_HEIGHT_MIN=0.04     # 最小台阶高度 (米)
STEP_HEIGHT_MAX=0.12     # 最大台阶高度 (米)

# ==================== 训练参数配置 ====================
# 数据相关
BATCH_SIZE=128           # GPU目前算力空闲，翻倍Batch Size进一步榨干GPU
NUM_QUERIES=1024         # 大量增加每个样本的采样点，让每个场景的梯度更加丰富
HEIGHTMAP_SIZE=21

# 模型相关
LATENT_DIM=256

# 优化相关
EPOCHS=2000          # 大规模预训练需要极大的Epoch数量
LR=3e-5                   # 初始学习率（降低！防止SIREN震荡）
LR_DECAY=true             # 启用余弦学习率衰减
LR_WARMUP=100             # 预热 epochs
LR_MIN=1e-7               # 最小学习率
MSE_WEIGHT=1.0
MSE_SCALE=10           # MSE Loss 放大系数（解决小梯度问题，10000=厘米级别）
EIKONAL_WEIGHT=0.001        # Eikonal Loss 权重

# 训练策略 - GPU利用率优化
STEPS_PER_EPOCH=10        # 每个epoch训练多少步（更多=更好GPU利用率）
PREFETCH=4                # 预取多少个batch（0=禁用，2-4=推荐）
EIKONAL_EVERY=1          # 每20步计算一次Eikonal loss（减少计算开销）
REGENERATE_EVERY=25       # 每50轮重新生成地形，逼迫网络学习真正的空间几何
USE_CURRICULUM=false      # 是否启用课程学习（逐步增加地形难度）

# Eikonal Loss 策略（iSDF 风格）
DECOUPLED_EIKONAL=true    # 启用解耦 Eikonal Loss（避免与 MSE 梯度冲突）
EIKONAL_POINTS=64         # 解耦模式下均匀采样点数

# ==================== 新增参数 (SIREN + 加权Loss) ====================
TRUNCATION_DISTANCE=0.3   # SDF 截断距离 (米) - 网络只关注正负30cm内的地形
LAMBDA_DECAY=2.0          # 距离衰减系数 - 近处权重高，远处权重低
OMEGA_0=30.0              # SIREN 频率参数 - 论文推荐值

# 其他
SEED=42
OUTPUT_DIR="checkpoints/sdf_pretrain/3-14"
LOAD_WEIGHTS="checkpoints/sdf_pretrain/3-14/checkpoint_epoch_100.npz"  # 只加载权重，从头训练

# ==================== 清理 JAX 缓存 ====================
echo "Cleaning JAX cache..."
rm -rf ./jax_cache
rm -rf ~/.cache/jax_catalog
echo "JAX cache cleaned."

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
echo "  Steps per Epoch: $STEPS_PER_EPOCH"
echo ""
echo "Optimizer Config:"
echo "  Learning Rate: $LR"
echo "  LR Decay: $LR_DECAY"
echo "  LR Warmup: $LR_WARMUP epochs"
echo "  LR Min: $LR_MIN"
echo ""
echo "Loss Config:"
echo "  MSE Weight: $MSE_WEIGHT"
echo "  MSE Scale: $MSE_SCALE (amplifies small SDF gradients)"
echo "  Eikonal Weight: $EIKONAL_WEIGHT"
echo "  Eikonal Every: $EIKONAL_EVERY steps"
echo "  Decoupled Eikonal: $DECOUPLED_EIKONAL"
echo "  Eikonal Points: $EIKONAL_POINTS"
echo ""
echo "SIREN + Weighted Loss Config (NEW!):"
echo "  Truncation Distance: $TRUNCATION_DISTANCE m (±$(echo "$TRUNCATION_DISTANCE * 100" | bc)cm)"
echo "  Lambda Decay: $LAMBDA_DECAY (near points weight more)"
echo "  Omega_0 (SIREN freq): $OMEGA_0"
echo ""
echo "Other Config:"
echo "  Regenerate Every: $REGENERATE_EVERY epochs"
echo "  Prefetch: $PREFETCH batches"
echo "  Curriculum: $USE_CURRICULUM"
echo "  Seed: $SEED"
echo "  Output Dir: $OUTPUT_DIR"
[ -n "$LOAD_WEIGHTS" ] && echo "  Load Weights From: $LOAD_WEIGHTS"
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
    --steps-per-epoch $STEPS_PER_EPOCH \
    --prefetch $PREFETCH \
    --mse-weight $MSE_WEIGHT \
    --mse-scale $MSE_SCALE \
    --eikonal-weight $EIKONAL_WEIGHT \
    --eikonal-every $EIKONAL_EVERY \
    --eikonal-points $EIKONAL_POINTS \
    --regenerate-every $REGENERATE_EVERY \
    --seed $SEED \
    --output-dir $OUTPUT_DIR \
    --truncation-distance $TRUNCATION_DISTANCE \
    --lambda-decay $LAMBDA_DECAY \
    --omega-0 $OMEGA_0"

# 添加学习率衰减（如果启用）
if [ "$LR_DECAY" = true ]; then
    CMD="$CMD --lr-decay --lr-warmup $LR_WARMUP --lr-min $LR_MIN"
fi

# 添加课程学习（如果启用）
if [ "$USE_CURRICULUM" = true ]; then
    CMD="$CMD --use-curriculum"
fi

# 添加解耦 Eikonal Loss（如果启用）
if [ "$DECOUPLED_EIKONAL" = true ]; then
    CMD="$CMD --decoupled-eikonal"
fi

# 添加加载权重参数（如果指定）
if [ -n "$LOAD_WEIGHTS" ]; then
    CMD="$CMD --load-weights $LOAD_WEIGHTS"
fi

# ==================== 执行训练 ====================
echo "Executing command..."
echo "$CMD"
echo ""
eval $CMD

echo "=========================================="
echo "Training Complete - $(date)"
echo "=========================================="
