#!/bin/bash
#SBATCH --job-name=Eval_Paper_Metrics
#SBATCH --output=log/slurm/eval_paper_%j.out
#SBATCH --error=log/slurm/eval_paper_%j.err
#SBATCH --partition=convergence
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --container-image nvcr.io\#nvidia/pytorch:22.08-py3
#SBATCH --container-mount-home

# ═══════════════════════════════════════════════════════
# Paper Metrics Evaluation: Coverage, TV, TC, BA, Risk-IoU
# Cluster: Convergence (LIP6) — Container mode
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
mkdir -p $PYTHONUSERBASE/lib/python3.8/site-packages

# Clean up any leftover files from previous plotly versions
rm -rf $PYTHONUSERBASE/lib/python3.8/site-packages/plotly*
rm -rf $PYTHONUSERBASE/lib/python3.8/site-packages/_plotly_utils*
rm -rf $PYTHONUSERBASE/lib/python3.8/site-packages/narwhals*

# ─── Install extra packages offline ───
pip install --no-index --find-links=$HOME/pip_wheels --user --no-deps \
            pytorch-lightning \
            torchmetrics \
            pyDeprecate \
            fsspec \
            packaging \
            online-conformal \
            salesforce-merlion \
            statsmodels \
            patsy \
            dill \
            plotly \
            narwhals \
            efficientnet-pytorch \
            einops \
            timm \
            --quiet

# ─── Paths ───
VAL_DATA="${VAL_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/val/}"
# This must be a calibrated Stage-2 checkpoint. A Stage-1 classifier has no
# trained risk-score head and therefore cannot produce Risk Tube metrics.
CKPT="${CAUSAL_CKPT:-$HOME/CausalForce/log/CausalForce_v1/epoch=14-last.ckpt}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: Stage-2 checkpoint not found: ${CKPT}" >&2
    echo "Set CAUSAL_CKPT=/absolute/path/to/stage2.ckpt" >&2
    exit 2
fi

# ─── Run ───
cd $HOME/CausalForce/CausalForce

python inference_causal.py \
    --data_root ${VAL_DATA} \
    --batch_size 16 \
    --gpus 1 \
    --checkpoint ${CKPT}

echo "Finished at: $(date)"
