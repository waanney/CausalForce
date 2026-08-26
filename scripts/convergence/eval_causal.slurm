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

# ═══════════════════════════════════════════════════════
# Evaluation: CausalForce Inference
# Cluster: Convergence (LIP6, Sorbonne Université)
# Metrics: Coverage, Tube Volume, TC, BA, Risk-IOU
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
TEST_DATA="$HOME/data/MCR_Dataset/Risk-Datasets-Venue/test/"
CKPT="../log/CausalForce_v1/best_epoch*.ckpt"  # from Stage 2

# ─── Run ───
cd CausalForce

python inference_causal.py \
    --data_root ${TEST_DATA} \
    --batch_size 10 \
    --gpus 1 \
    --checkpoint ${CKPT}

echo "Finished at: $(date)"
