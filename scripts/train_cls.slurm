#!/bin/bash
#SBATCH --job-name=CausalForce_CLS
#SBATCH --output=log/slurm/cls_%j.out
#SBATCH --error=log/slurm/cls_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ─── Stage 1: Risk Category Classifier Pre-training ───
# This trains the backbone + LSTM + risk_type_head
# Output checkpoint is used as initialization for Stage 2

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Environment ───
source activate CausalForce  # or: conda activate CausalForce

# ─── Paths (EDIT THESE) ───
TRAIN_DATA="/path/to/your/training_data/"
VAL_DATA="/path/to/your/validation_data/"

# ─── Create log directories ───
mkdir -p log/slurm

# ─── Run ───
cd CausalForce

python train_cls.py \
    --id risk_category_classifier \
    --epochs 10 \
    --lr 0.0001 \
    --val_every 1 \
    --batch_size 6 \
    --gpus 1 \
    --logdir ../log

echo "Finished at: $(date)"
