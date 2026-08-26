#!/bin/bash
#SBATCH --job-name=CausalForce
#SBATCH --output=log/slurm/causal_%j.out
#SBATCH --error=log/slurm/causal_%j.err
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --container-image nvcr.io\#nvidia/pytorch:22.08-py3
#SBATCH --container-mount-home

# ═══════════════════════════════════════════════════════
# Stage 2: CausalForce Full Training
# Cluster: Convergence (LIP6) — Container mode
# Container: NVIDIA PyTorch 22.08 (PyTorch 1.13 + CUDA 11.7)
# ═══════════════════════════════════════════════════════

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Install extra pure-python packages offline ───
pip install --no-index --find-links=$HOME/pip_wheels --no-deps \
            pytorch-lightning \
            online-conformal \
            efficientnet-pytorch \
            einops \
            --quiet

# ─── Paths ───
TRAIN_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/train/"
VAL_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/val/"
CLS_CKPT="$HOME/CausalForce/log/risk_category_classifier/best_epoch*.ckpt"

# ─── Causal Hyperparameters ───
EXP_ID="CausalForce_v1"
NUM_CONFOUNDERS=64
NUM_HEADS=4
CF_ALPHA=0.3
W_ORTHO=0.1
W_CF=0.5
ORTHO_RAMP=4
EPOCHS=15
LR=0.0001
BATCH_SIZE=4

# ─── Create log directories ───
mkdir -p $HOME/CausalForce/log/slurm

# ─── Run ───
cd $HOME/CausalForce/CausalForce

python train_causal.py \
    --id ${EXP_ID} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --val_every 3 \
    --batch_size ${BATCH_SIZE} \
    --gpus 1 \
    --logdir ../log \
    --pretrain_ckpt ${CLS_CKPT} \
    --num_confounders ${NUM_CONFOUNDERS} \
    --num_heads ${NUM_HEADS} \
    --cf_alpha ${CF_ALPHA} \
    --w_ortho ${W_ORTHO} \
    --w_cf ${W_CF} \
    --ortho_ramp_epochs ${ORTHO_RAMP}

echo "Finished at: $(date)"
