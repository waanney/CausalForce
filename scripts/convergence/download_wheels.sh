#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download exact compatible offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading offline wheels..."

# Download pytorch-lightning 1.7.7 wheel directly via curl to bypass frontend pip 24.1+ metadata validation
curl -sSL "https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl" -o $HOME/pip_wheels/pytorch_lightning-1.7.7-py3-none-any.whl

pip download \
    torchmetrics==0.9.3 \
    online-conformal==1.0.2 \
    efficientnet-pytorch==0.7.0 \
    einops==0.4.1 \
    pyyaml==6.0 \
    fsspec==2022.5.0 \
    packaging==21.3 \
    typing-extensions==4.3.0 \
    -d $HOME/pip_wheels

echo "✅ All compatible wheels downloaded to $HOME/pip_wheels"
