#!/bin/bash
#SBATCH --job-name=CausalForce_CLS
#SBATCH --output=log/slurm/cls_%j.out
#SBATCH --error=log/slurm/cls_%j.err
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

# ═══════════════════════════════════════════════════════
# Stage 1: Risk Category Classifier Pre-training
# Cluster: Convergence (LIP6, Sorbonne Université)
# GPU: NVIDIA A100 80GB
# ═══════════════════════════════════════════════════════

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Environment ───
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"
conda activate CausalForce

# ─── Paths ───
TRAIN_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/train/"
VAL_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/val/"

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
