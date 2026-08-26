#!/bin/bash
#SBATCH --job-name=Eval_Causal
#SBATCH --output=log/slurm/eval_%j.out
#SBATCH --error=log/slurm/eval_%j.err
#SBATCH --partition=convergence
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --container-image nvcr.io\#nvidia/pytorch:22.08-py3
#SBATCH --container-mount-home

# ═══════════════════════════════════════════════════════
# Evaluation: CausalForce Inference
# Cluster: Convergence (LIP6) — Container mode
# Metrics: Coverage, Tube Volume, TC, BA, Risk-IOU
# ═══════════════════════════════════════════════════════

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Install dependencies inside container ───
pip install pytorch-lightning==1.6.5 \
            online-conformal==1.0.2 \
            pandas==1.3.5 \
            efficientnet-pytorch==0.7.0 \
            einops==0.4.1 \
            timm==0.5.4 \
            torchmetrics==0.11.4 \
            scikit-image==0.19.3 \
            opencv-python-headless==4.5.1.48 \
            --quiet

# ─── Paths ───
TEST_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/test/"
CKPT="$HOME/CausalForce/log/CausalForce_v1/best_epoch*.ckpt"

# ─── Run ───
cd $HOME/CausalForce/CausalForce

python inference_causal.py \
    --data_root ${TEST_DATA} \
    --batch_size 10 \
    --gpus 1 \
    --checkpoint ${CKPT}

echo "Finished at: $(date)"
