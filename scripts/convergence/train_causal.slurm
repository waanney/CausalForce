#!/bin/bash
#SBATCH --job-name=CausalForce
#SBATCH --output=log/slurm/causal_%j.out
#SBATCH --error=log/slurm/causal_%j.err
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --container-image nvcr.io\#nvidia/pytorch:22.08-py3
#SBATCH --container-mount-home

# ═══════════════════════════════════════════════════════
# Stage 2: CausalForce Full Training
# Cluster: Convergence (LIP6) — Container mode
# Maximum Performance A100 Configuration (Batch Size 18)
# ═══════════════════════════════════════════════════════

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Environment Isolation ───
export PYTHONUSERBASE=$HOME/.container_packages
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
mkdir -p $PYTHONUSERBASE

# ─── Install extra packages offline ───
pip install --no-index --find-links=$HOME/pip_wheels --user --no-deps \
            pytorch-lightning \
            torchmetrics \
            pyDeprecate \
            fsspec \
            packaging \
            online-conformal \
            salesforce-merlion \
            dill \
            plotly \
            efficientnet-pytorch \
            einops \
            timm \
            --quiet

# ─── Paths ───
TRAIN_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/train/"
VAL_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/val/"
CLS_CKPT="$HOME/CausalForce/log/risk_category_classifier/epoch=9-last.ckpt"

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
BATCH_SIZE=18
NUM_WORKERS=12

# ─── Create log directories ───
mkdir -p $HOME/CausalForce/log/slurm

# ─── Run ───
cd $HOME/CausalForce/CausalForce

python train_causal.py \
    --id ${EXP_ID} \
    --train_data ${TRAIN_DATA} \
    --val_data ${VAL_DATA} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --val_every 1 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --precision 16 \
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
