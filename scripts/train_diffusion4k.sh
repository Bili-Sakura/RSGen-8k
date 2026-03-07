#!/bin/bash
# Diffusion-4K Wavelet Fine-tuning Launch Script
# ================================================
# Fine-tune a Stable Diffusion UNet with wavelet-domain loss for
# ultra-high-resolution image synthesis.
#
# Reference: Zhang et al., "Diffusion-4K: Ultra-High-Resolution Image
#            Synthesis with Latent Diffusion Models", CVPR 2025.
#            https://github.com/zhang0jhon/diffusion-4k
#
# Usage:
#   bash scripts/train_diffusion4k.sh
#
# Prerequisites:
#   pip install rsgen8k[training]
#   # or: pip install pytorch-wavelets bitsandbytes

set -euo pipefail

export INSTANCE_DIR="${INSTANCE_DIR:-./data/train}"
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/diffusion4k}"
export MODEL_NAME="${MODEL_NAME:-./models/lcybuaa/Text2Earth}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  -m rsgen8k.training.trainer \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="bf16" \
  --resolution=2048 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=5000 \
  --seed=0 \
  --wave="haar"
