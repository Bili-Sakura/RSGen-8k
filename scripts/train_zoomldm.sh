#!/bin/bash
# ZoomLDM Training Launch Script
# ===============================
# Multi-scale latent diffusion with magnification-aware SSL conditioning
# for remote sensing imagery (NAIP-style).
#
# Reference: Yellapragada et al., "ZoomLDM: Latent Diffusion Model for
#            Multi-scale Image Generation", CVPR 2025.
# Source: https://github.com/Bili-Sakura/ZoomLDM-Diffusers
#
# Prerequisites:
#   1. Clone ZoomLDM-Diffusers:
#        git clone https://github.com/Bili-Sakura/ZoomLDM-Diffusers.git zoomldm_repo
#   2. Install ZoomLDM deps: cd zoomldm_repo && pip install -r requirements.txt
#   3. Prepare data: multi-scale patches with pre-extracted VAE + DINO-v2
#      features (see ZoomLDM NAIP demo dataset / notebooks)
#
# Usage:
#   export ZOOMLDM_REPO="${ZOOMLDM_REPO:-./zoomldm_repo}"
#   export DATA_ROOT="./data/zoomldm_rs"
#   bash scripts/train_zoomldm.sh
#
# Or with custom config:
#   ZOOMLDM_REPO=./zoomldm_repo bash scripts/train_zoomldm.sh --base configs/zoomldm_naip.yaml data.params.train.params.config.root=./data/zoomldm_rs

set -euo pipefail

ZOOMLDM_REPO="${ZOOMLDM_REPO:-./zoomldm_repo}"
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="${CONFIG:-configs/zoomldm_naip.yaml}"
GPUS="${GPUS:-0}"

if [[ ! -d "$ZOOMLDM_REPO" ]]; then
    echo "Error: ZoomLDM-Diffusers repo not found at $ZOOMLDM_REPO"
    echo "Clone it with: git clone https://github.com/Bili-Sakura/ZoomLDM-Diffusers.git $ZOOMLDM_REPO"
    exit 1
fi

# Add ZoomLDM repo to PYTHONPATH so ldm.* and main.* are importable
export PYTHONPATH="${ZOOMLDM_REPO}:${PYTHONPATH:-}"

# Change to ZoomLDM repo for main.py (it expects to run from repo root)
cd "$ZOOMLDM_REPO"

EXTRA_ARGS=()
if [[ -n "$DATA_ROOT" ]]; then
    EXTRA_ARGS+=(data.params.train.params.config.root="$DATA_ROOT")
fi

echo "Running ZoomLDM training from $ZOOMLDM_REPO"
echo "  Config: $CONFIG"
echo "  GPUs: $GPUS"
echo "  Extra: ${EXTRA_ARGS[*]}"

python main.py -t --gpus "$GPUS" --base "$CONFIG" "${EXTRA_ARGS[@]}" "$@"
