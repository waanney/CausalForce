#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download pre-built offline wheels and pretrained weights for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels
mkdir -p $HOME/.cache/torch/hub/checkpoints

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading pre-built offline packages..."

# 1. Download pytorch-lightning 1.7.7 wheel via curl (using -L to follow redirects)
curl -sSL -L "https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl" -o $HOME/pip_wheels/pytorch_lightning-1.7.7-py3-none-any.whl

# 2. Download packages and their dependencies
pip download timm==0.6.13 -d $HOME/pip_wheels
pip download torchmetrics==0.9.3 -d $HOME/pip_wheels
pip download pyDeprecate==0.3.2 -d $HOME/pip_wheels
pip download fsspec==2022.5.0 -d $HOME/pip_wheels
pip download packaging==21.3 -d $HOME/pip_wheels
pip download online-conformal==1.0.2 -d $HOME/pip_wheels
pip download salesforce-merlion -d $HOME/pip_wheels
pip download dill -d $HOME/pip_wheels
pip download efficientnet-pytorch==0.7.1 -d $HOME/pip_wheels
pip download einops==0.4.1 -d $HOME/pip_wheels

# 3. Download torchvision and timm ResNet-50 pretrained weights for offline execution
echo "Downloading ResNet-50 pretrained weights to torch hub cache..."
curl -sSL -L "https://download.pytorch.org/models/resnet50-0676744e.pth" -o $HOME/.cache/torch/hub/checkpoints/resnet50-0676744e.pth
curl -sSL -L "https://download.pytorch.org/models/resnet50-0676ba61.pth" -o $HOME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
curl -sSL -L "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth" -o $HOME/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth

# 4. Sanity check: ensure all files are valid binaries/archives and not HTML error pages
echo "Verifying downloaded packages in $HOME/pip_wheels:"
for f in $HOME/pip_wheels/*; do
    if head -n 1 "$f" 2>/dev/null | grep -q -E "^<|^<!DOCTYPE"; then
        echo "❌ Error: $f is an HTML error page! Removing..."
        rm -f "$f"
    else
        echo "  ✓ $(basename "$f") ($(du -h "$f" | cut -f1))"
    fi
done

echo "✅ All offline packages and pretrained weights downloaded and verified successfully!"
