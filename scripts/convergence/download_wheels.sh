#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download exact compatible offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading compatible offline wheels..."

pip download \
    pytorch-lightning==1.6.5 \
    torchmetrics==0.8.2 \
    online-conformal==1.0.2 \
    efficientnet-pytorch==0.7.0 \
    einops==0.4.1 \
    pyyaml==6.0 \
    fsspec==2022.5.0 \
    packaging==21.3 \
    typing-extensions==4.3.0 \
    -d $HOME/pip_wheels

echo "✅ All compatible wheels downloaded to $HOME/pip_wheels"
