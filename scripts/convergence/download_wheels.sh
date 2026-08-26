#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download offline wheels for container execution on Convergence
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

echo "Downloading Python wheels for offline installation on compute nodes..."

pip download \
    pytorch-lightning==1.6.5 \
    online-conformal==1.0.2 \
    pandas==1.3.5 \
    efficientnet-pytorch==0.7.0 \
    einops==0.4.1 \
    timm==0.5.4 \
    torchmetrics==0.11.4 \
    scikit-image==0.19.3 \
    opencv-python-headless==4.5.1.48 \
    -d $HOME/pip_wheels

echo "✅ Wheels downloaded to $HOME/pip_wheels"
