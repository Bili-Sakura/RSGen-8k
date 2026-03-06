#!/usr/bin/env bash
# RSGen-8k environment setup
# Creates conda env `rsgen` with Python 3.12 and PyTorch 2.8.0 (CUDA 12.6)

set -e

ENV_NAME="${RSGEN_ENV:-rsgen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Creating conda environment: $ENV_NAME (Python 3.12)"
conda create -n "$ENV_NAME" python=3.12 -y

echo "==> Activating $ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "==> Installing PyTorch 2.8.0 + CUDA 12.6"
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing requirements"
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "==> Installing swanlab"
pip install swanlab

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
echo "Optional: pip install muon-optimizer"
