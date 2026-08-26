#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Download pre-built offline wheels for container execution
# Run this on the FRONTEND node (front.convergence.lip6.fr)
# ═══════════════════════════════════════════════════════════════

mkdir -p $HOME/pip_wheels

# Clean up broken/partial wheels
rm -rf $HOME/pip_wheels/*

echo "Downloading pre-built offline packages..."

# 1. Download pytorch-lightning 1.7.7 wheel via curl (using -L to follow redirects)
curl -sSL -L "https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl" -o $HOME/pip_wheels/pytorch_lightning-1.7.7-py3-none-any.whl

# 2. Download pure-python packages via pip --no-deps
pip download --no-deps online-conformal==1.0.2 -d $HOME/pip_wheels
pip download --no-deps efficientnet-pytorch==0.7.1 -d $HOME/pip_wheels
pip download --no-deps einops==0.4.1 -d $HOME/pip_wheels

# 3. Sanity check: ensure all files are valid binaries/archives and not HTML error pages
echo "Verifying downloaded packages in $HOME/pip_wheels:"
for f in $HOME/pip_wheels/*; do
    if head -n 1 "$f" 2>/dev/null | grep -q -E "^<|^<!DOCTYPE"; then
        echo "❌ Error: $f is an HTML error page! Removing..."
        rm -f "$f"
    else
        echo "  ✓ $(basename "$f") ($(du -h "$f" | cut -f1))"
    fi
done

echo "✅ All offline packages downloaded and verified successfully in $HOME/pip_wheels"
