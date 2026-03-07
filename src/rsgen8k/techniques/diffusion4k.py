"""Diffusion-4K: wavelet-finetuned denoising for ultra-high-resolution images.

Adapted from the Diffusion-4K project by Jinjin Zhang et al.
  Paper : "Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent
           Diffusion Models", CVPR 2025.
  Source: https://github.com/zhang0jhon/diffusion-4k
  License: Apache-2.0

A model fine-tuned with the Diffusion-4K wavelet loss learns to produce
high-frequency detail at ultra-high resolutions.  At inference time the
denoising step is standard classifier-free-guided DDIM — the improved
detail comes from the fine-tuned weights, not from a specialised per-step
algorithm.

This module provides ``diffusion4k_denoise_step`` so that the RSGen-8k
multi-stage engine can dispatch to it like any other technique.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def diffusion4k_denoise_step(
    unet: torch.nn.Module,
    scheduler: object,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float = 7.5,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perform one DDIM denoising step with a wavelet-finetuned UNet.

    The denoising itself is standard classifier-free guidance; the quality
    improvement comes from the UNet weights having been trained with the
    Diffusion-4K wavelet loss (see :mod:`rsgen8k.training.wavelet_loss`).

    Args:
        unet: The (wavelet-finetuned) UNet model.
        scheduler: DDIM scheduler with ``scale_model_input`` and ``step``.
        latents: Current noisy latents, shape ``(B, C, H, W)``.
        text_embeddings: Text encoder output, shape ``(2*B, seq, dim)`` when
            using classifier-free guidance (unconditional + conditional).
        t: Current timestep tensor.
        guidance_scale: CFG scale factor.
        class_labels: Optional class/resolution conditioning labels.

    Returns:
        Updated latents after one denoising step.
    """
    # Expand latents for classifier-free guidance (unconditional + conditional)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Build UNet kwargs
    unet_kwargs = {"encoder_hidden_states": text_embeddings}
    if class_labels is not None:
        unet_kwargs["class_labels"] = class_labels

    # Predict noise
    noise_pred = unet(latent_model_input, t, **unet_kwargs).sample

    # Classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # Scheduler step
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents
