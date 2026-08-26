#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CausalForce — Setup on Convergence (LIP6) Cluster
# Run this ONCE on the frontend: front.convergence.lip6.fr
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  CausalForce — Convergence Cluster Setup     ║"
echo "╚══════════════════════════════════════════════╝"

# ─── 1. Clone the repository ───
echo ""
echo ">>> [1/4] Cloning repository..."
cd ~
git clone https://github.com/hcis-lab/CRTP.git
cd CRTP

# ─── 2. Create conda environment ───
echo ""
echo ">>> [2/4] Creating conda environment..."
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda env create -f environment.yml --name CausalForce
conda activate CausalForce

echo ""
echo ">>> Environment created. Verifying PyTorch + CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ─── 3. Download dataset ───
echo ""
echo ">>> [3/4] Downloading Multiple Coexisting Risks Dataset..."
echo ""
echo "⚠️  The dataset is hosted on Google Drive."
echo "    You need to install gdown and download manually."
echo ""

pip install gdown

# Create dataset directory
mkdir -p ~/data/MCR_Dataset

# Download from Google Drive folder
# Link: https://drive.google.com/drive/folders/13hRzEaJadxPIgf_hRIaQVJq9Pr1SsGIL
echo "Downloading dataset (this may take a while)..."
gdown --folder "https://drive.google.com/drive/folders/13hRzEaJadxPIgf_hRIaQVJq9Pr1SsGIL" -O ~/data/MCR_Dataset/

# Extract all zip files
echo ""
echo ">>> Extracting zip files..."
cd ~/data/MCR_Dataset
for f in *.zip; do
    [ -f "$f" ] && echo "  Extracting $f ..." && unzip -q -o "$f"
done

# ─── 4. Organize train/val/test splits ───
echo ""
echo ">>> [4/4] Organizing data splits..."

mkdir -p ~/data/MCR_Dataset/train
mkdir -p ~/data/MCR_Dataset/val
mkdir -p ~/data/MCR_Dataset/test

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Organize your train/val/test splits in ~/data/MCR_Dataset/"
echo "  2. Edit data paths in the SLURM scripts"
echo "  3. Submit jobs:"
echo ""
echo "     # Stage 1: Risk Category Classifier"
echo "     cd ~/CRTP && sbatch scripts/convergence/train_cls.slurm"
echo ""
echo "     # Stage 2: Causal Training (after Stage 1 finishes)"
echo "     cd ~/CRTP && sbatch scripts/convergence/train_causal.slurm"
echo ""
echo "     # Evaluation"
echo "     cd ~/CRTP && sbatch scripts/convergence/eval_causal.slurm"
echo ""
