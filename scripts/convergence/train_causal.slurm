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

# ═══════════════════════════════════════════════════════
# Stage 2: CausalForce Full Training
# Cluster: Convergence (LIP6, Sorbonne Université)
# GPU: NVIDIA A100 80GB
#
# Loads pre-trained classifier from Stage 1
# Trains CausalDisentangle + CausalMessagePassing + CausalRiskHead
# Loss = L_score + L_htsc + L_ortho + L_cf
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
TRAIN_DATA="$HOME/data/MCR_Dataset/train/"
VAL_DATA="$HOME/data/MCR_Dataset/val/"
CLS_CKPT="../log/risk_category_classifier/best_epoch*.ckpt"  # from Stage 1

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
mkdir -p log/slurm

# ─── Run ───
cd Conformal_Risk_Tube_Prediction

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
