#!/bin/bash
#SBATCH --job-name=CausalForce
#SBATCH --output=log/slurm/causal_%j.out
#SBATCH --error=log/slurm/causal_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# ─── Stage 2: CausalForce Full Training ───
# Loads pre-trained classifier from Stage 1
# Trains CausalDisentangle + CausalMessagePassing + CausalRiskHead
# with L_score + L_htsc + L_ortho + L_cf

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
CLS_CKPT="log/risk_category_classifier/best_epoch*.ckpt"  # from Stage 1

# ─── Causal Hyperparameters ───
EXP_ID="CausalForce_v1"
NUM_CONFOUNDERS=64    # confounder dictionary size
NUM_HEADS=4           # multi-head causal attention heads
CF_ALPHA=0.3          # counterfactual branch weight
W_ORTHO=0.1           # orthogonality loss weight
W_CF=0.5              # counterfactual loss weight
ORTHO_RAMP=4          # epochs to ramp up L_ortho from 0→W_ORTHO
EPOCHS=15
LR=0.0001
BATCH_SIZE=4

# ─── Create log directories ───
mkdir -p log/slurm

# ─── Run ───
cd CausalForce

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
