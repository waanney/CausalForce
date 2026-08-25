#!/bin/bash
#SBATCH --job-name=Eval_Causal
#SBATCH --output=log/slurm/eval_%j.out
#SBATCH --error=log/slurm/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ─── Evaluation: CausalForce Inference ───
# Metrics: Coverage, Tube Volume, TC, BA, Risk-IOU

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time:   $(date)"
echo "============================================"

# ─── Environment ───
source activate CausalForce

# ─── Paths (EDIT THESE) ───
TEST_DATA="/path/to/your/testing_data/"
CKPT="log/CausalForce_v1/best_epoch*.ckpt"  # from Stage 2

# ─── Run ───
cd CausalForce

python inference_causal.py \
    --data_root ${TEST_DATA} \
    --batch_size 10 \
    --gpus 1 \
    --checkpoint ${CKPT}

echo "Finished at: $(date)"
