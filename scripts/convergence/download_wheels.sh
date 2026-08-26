#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download pure-python offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

echo "Downloading pure-python wheels and dependencies..."

pip download \
    "pytorch-lightning>=1.6.0,<1.8.0" \
    torchmetrics \
    online-conformal==1.0.2 \
    efficientnet-pytorch==0.7.0 \
    einops==0.4.1 \
    pyyaml \
    fsspec \
    packaging \
    -d $HOME/pip_wheels

echo "✅ Wheels downloaded to $HOME/pip_wheels"
