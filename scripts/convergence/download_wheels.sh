#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download pre-built offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading pre-built offline wheels..."

# 1. Download pytorch-lightning 1.7.7 wheel directly via curl
curl -sSL "https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl" -o $HOME/pip_wheels/pytorch_lightning-1.7.7-py3-none-any.whl

# 2. Download remaining pure-python wheels without building from source
pip download \
    online-conformal==1.0.2 \
    efficientnet-pytorch==0.7.0 \
    einops==0.4.1 \
    --only-binary=:all: \
    -d $HOME/pip_wheels

echo "✅ All pre-built wheels downloaded successfully to $HOME/pip_wheels"
