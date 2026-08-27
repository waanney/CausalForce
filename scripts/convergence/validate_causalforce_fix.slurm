#!/bin/bash

#SBATCH --job-name=causalforce_validate_fix
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=convergence
#SBATCH --ntasks-per-node=1
#SBATCH --container-image nvcr.io\#nvidia/pytorch:22.08-py3
#SBATCH --container-mount-home

set -euo pipefail

# The submission directory "logs" must exist before calling sbatch because
# Slurm opens stdout/stderr before the job body starts.

START_TIME=$(date --iso-8601=seconds)
REPO_ROOT="${REPO_ROOT:-$HOME/CausalForce}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/train}"
VAL_DATA="${VAL_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/val}"
TEST_DATA="${TEST_DATA:-$HOME/data/MCR_Dataset/Risk-Datasets-Venue/test}"
CLS_CKPT="${CLS_CKPT:-$REPO_ROOT/log/risk_category_classifier/epoch=9-last.ckpt}"
CAUSAL_CKPT="${CAUSAL_CKPT:-$REPO_ROOT/log/CausalForce_v1/epoch=14-last.ckpt}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "${REPO_ROOT}"

[[ -f "${CLS_CKPT}" ]] || { echo "Missing CLS_CKPT: ${CLS_CKPT}" >&2; exit 2; }
[[ -f "${CAUSAL_CKPT}" ]] || { echo "Missing CAUSAL_CKPT: ${CAUSAL_CKPT}" >&2; exit 2; }
[[ -d "${TRAIN_DATA}" ]] || { echo "Missing TRAIN_DATA: ${TRAIN_DATA}" >&2; exit 2; }
[[ -d "${VAL_DATA}" ]] || { echo "Missing VAL_DATA: ${VAL_DATA}" >&2; exit 2; }
[[ -d "${TEST_DATA}" ]] || { echo "Missing TEST_DATA: ${TEST_DATA}" >&2; exit 2; }

echo "================================================"
echo "CAUSALFORCE FIX VALIDATION"
echo "================================================"
echo "branch:          $(git branch --show-current)"
echo "commit:          $(git rev-parse HEAD)"
echo "hostname:        $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "Python:          $(python --version 2>&1)"
echo "PyTorch:         $(python -c 'import torch; print(torch.__version__)')"
echo "classifier ckpt: ${CLS_CKPT}"
echo "causal ckpt:     ${CAUSAL_CKPT}"
echo "train dataset:   ${TRAIN_DATA}"
echo "val dataset:     ${VAL_DATA}"
echo "test dataset:    ${TEST_DATA}"
echo "config:          batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}, debug_samples=10"
echo "start time:      ${START_TIME}"

export PYTHONUSERBASE="$HOME/.container_packages"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.8/site-packages:${PYTHONPATH:-}"
mkdir -p "$PYTHONUSERBASE/lib/python3.8/site-packages"

pip install --no-index --find-links="$HOME/pip_wheels" --user --no-deps \
    pytorch-lightning torchmetrics pyDeprecate fsspec packaging \
    online-conformal salesforce-merlion statsmodels patsy dill plotly \
    narwhals efficientnet-pytorch einops timm --quiet

echo "[1/5] Automated regression tests"
python -m pytest -q tests
python CausalForce/train_causal.py --help >/dev/null
python CausalForce/inference_cls.py --help >/dev/null
python CausalForce/inference_causal.py --help >/dev/null
(
    cd CausalForce
    python -c "import causal_model, causal_modules, data, risk_tube_metrics; print('Import test: PASS')"
)

echo "[2/5] Dataset split and label audit"
HASH_FLAG=()
if [[ "${HASH_IMAGES:-0}" == "1" ]]; then
    HASH_FLAG=(--hash_images)
fi
python CausalForce/audit_dataset.py \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --test_data "${TEST_DATA}" \
    "${HASH_FLAG[@]}"

echo "[3/5] Stage 1 validation metrics"
cd "${REPO_ROOT}/CausalForce"
python inference_cls.py \
    --data_root "${VAL_DATA}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --gpus 1 \
    --checkpoint "${CLS_CKPT}"

echo "[4/5] Stage 2 real-batch forward/loss/backward audit"
python validate_pipeline.py \
    --data_root "${VAL_DATA}" \
    --checkpoint "${CAUSAL_CKPT}" \
    --batch_size 2 \
    --num_workers 2

echo "[5/5] Final Risk Tube evaluation on held-out test split"
python inference_causal.py \
    --data_root "${TEST_DATA}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --debug_samples 10 \
    --gpus 1 \
    --checkpoint "${CAUSAL_CKPT}"

END_TIME=$(date --iso-8601=seconds)
echo "end time:        ${END_TIME}"
echo "Validation pipeline: PASS"
