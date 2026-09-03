#!/bin/bash
#SBATCH --job-name=Calib_Eval_Causal
#SBATCH --output=log/slurm/calib_eval_%j.out
#SBATCH --error=log/slurm/calib_eval_%j.err
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
# CausalForce SAOCP Calibration + Paper Metrics Evaluation
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
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/train/}"
TEST_DATA="${TEST_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/test/}"
INPUT_CKPT="${CAUSAL_CKPT:-$HOME/CausalForce/log/CausalForce_v1/epoch=14-last.ckpt}"
CALIB_CKPT="$HOME/CausalForce/log/CausalForce_v1/epoch=14-calibrated.ckpt"
COVERAGE="${COVERAGE:-0.9}"

if [[ ! -f "${INPUT_CKPT}" ]]; then
    echo "ERROR: Input checkpoint not found: ${INPUT_CKPT}" >&2
    exit 2
fi

cd $HOME/CausalForce/CausalForce

# ─── Step 1: Calibrate SAOCP on training/calibration stream ───
echo "[Step 1/2] Calibrating SAOCP conformal prediction on ${TRAIN_DATA}..."
python calibrate_saocp.py \
    --data_root "${TRAIN_DATA}" \
    --checkpoint "${INPUT_CKPT}" \
    --output_ckpt "${CALIB_CKPT}" \
    --batch_size 16 \
    --num_workers 8 \
    --gpus 1 \
    --coverage "${COVERAGE}"

# ─── Step 2: Evaluate on held-out test split ───
echo "[Step 2/2] Evaluating Paper Metrics on ${TEST_DATA} with calibrated checkpoint..."
python inference_causal.py \
    --data_root "${TEST_DATA}" \
    --batch_size 16 \
    --gpus 1 \
    --num_workers 8 \
    --checkpoint "${CALIB_CKPT}"

echo "Finished at: $(date)"
