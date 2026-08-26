#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download pre-built offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading pre-built offline packages..."

# 1. Download pytorch-lightning 1.7.7 wheel directly via curl
curl -sSL "https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl" -o $HOME/pip_wheels/pytorch_lightning-1.7.7-py3-none-any.whl

# 2. Download efficientnet_pytorch tarball directly via curl (since PyPI has no prebuilt wheel for 0.7.x)
curl -sSL "https://files.pythonhosted.org/packages/4e/4a/07ac5e2e850d999335f6063ed037df8549727653ee8840d2f09908488e04/efficientnet_pytorch-0.7.1.tar.gz" -o $HOME/pip_wheels/efficientnet_pytorch-0.7.1.tar.gz

# 3. Download remaining pure-python wheels
pip download \
    online-conformal==1.0.2 \
    einops==0.4.1 \
    --only-binary=:all: \
    -d $HOME/pip_wheels

echo "✅ All offline packages downloaded successfully to $HOME/pip_wheels"
