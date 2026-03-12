#!/bin/bash
# SDF 预训练启动脚本 - 两阶段训练
# 王甫12138

# 激活 conda 环境
eval "$(conda shell."$CONDA_EXE" shell)" 2>/dev/null || {
    echo "Conda not found. Please install conda first."
    exit 1
}
conda activate mpx_env

# ==================== 阶段 1: 只训练 MSE ====================
STAGE1_EPOCHS=300
STAGE1_BATCH_SIZE=16
STAGE1_LR=1e-3
STAGE1_EIKONAL_EVERY=9999  # 不计算 Eikonal
STAGE1_OUTPUT="checkpoints/sdf_pretrain_stage1"

# ==================== 阶段 2: 加入 Eikonal Loss ====================
STAGE2_EPOCHS=100
STAGE2_BATCH_SIZE=16
STAGE2_LR=1e-4
STAGE2_EIKONAL_EVERY=5
STAGE2_RESUME="checkpoints/sdf_pretrain_stage1/final_checkpoint.npz"
STAGE2_OUTPUT="checkpoints/sdf_pretrain_stage2"

echo "=========================================="
echo "SDF Pretraining - Two-Stage Training"
echo "=========================================="
echo ""
echo "Stage 1: MSE only"
echo "  Epochs: $STAGE1_EPOCHS"
echo "  Batch Size: $STAGE1_BATCH_SIZE"
echo "  Learning Rate: $STAGE1_LR"
echo "  Eikonal: Disabled (every $STAGE1_EIKONAL_EVERY)"
echo ""
echo "Stage 2: MSE + Eikonal"
echo "  Epochs: $STAGE2_EPOCHS"
echo "  Batch Size: $STAGE2_BATCH_SIZE"
echo "  Learning Rate: $STAGE2_LR"
echo "  Eikonal Every: $STAGE2_EIKONAL_EVERY"
echo "=========================================="
echo ""

# ==================== 阶段 1 ====================
echo "Starting Stage 1: Training MSE only..."
python -m mpx.sdf_pretrain.train \
    --epochs $STAGE1_EPOCHS \
    --batch-size $STAGE1_BATCH_SIZE \
    --lr $STAGE1_LR \
    --eikonal-every $STAGE1_EIKONAL_EVERY \
    --output-dir $STAGE1_OUTPUT

    # ==================== 阶段 2 ====================
echo ""
echo "Starting Stage 2: Training with Eikonal Loss..."
python -m mpx.sdf_pretrain.train \
    --epochs $STAGE2_EPOCHS \
    --batch-size $STAGE2_BATCH_SIZE \
    --lr $STAGE2_LR \
    --eikonal-every $STAGE2_EIKONAL_EVERY \
    --resume $STAGE2_RESUME \
    --output-dir $STAGE2_OUTPUT

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Stage 1 checkpoint: $STAGE1_OUTPUT/final_checkpoint.npz"
echo "Stage 2 checkpoint: $STAGE2_OUTPUT/final_checkpoint.npz"
echo "=========================================="
