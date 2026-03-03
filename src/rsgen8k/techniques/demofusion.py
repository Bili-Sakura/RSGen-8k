"""DemoFusion: Progressive upscaling with dilated sampling.

Adapted from the DemoFusion project by Ruoyi Du et al.
  Paper : "DemoFusion: Democratising High-Resolution Image Generation
           With No $$$", CVPR 2024.
  Source: https://github.com/PRIS-CV/DemoFusion
  License: Apache-2.0

DemoFusion progressively upscales images through multiple resolution
phases.  At each phase it applies *dilated sampling* (spreading latent
samples over a larger spatial area) combined with *skip-residual*
connections from the previous resolution to maintain global structure.
A Gaussian blur is used to smooth the skip residuals before blending.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def _gaussian_blur(x: torch.FloatTensor, kernel_size: int = 3) -> torch.FloatTensor:
    """Apply Gaussian blur to a tensor (B, C, H, W)."""
    padding = kernel_size // 2
    # Simple box-blur approximation
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / weight.numel()
    channels = x.shape[1]
    weight = weight.expand(channels, 1, kernel_size, kernel_size)
    return F.conv2d(x, weight, padding=padding, groups=channels)


def demofusion_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    skip_residual: Optional[torch.FloatTensor] = None,
    skip_weight: float = 0.2,
    dilation: int = 2,
) -> torch.FloatTensor:
    """Perform one DemoFusion denoising step with dilated sampling.

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text (with unconditional if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        skip_residual: Latents from the previous resolution phase
            (up-sampled to current size), used as a skip connection.
        skip_weight: Blending weight for the skip-residual signal.
        dilation: Dilation factor for dilated sampling.

    Returns:
        Updated latents after one denoising step.
    """
    do_cfg = guidance_scale > 1.0

    latent_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_input = scheduler.scale_model_input(latent_input, t)

    noise_pred = unet(
        latent_input, t, encoder_hidden_states=text_embeddings
    ).sample.to(dtype=latents.dtype)

    if do_cfg:
        u, c = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)

    step_out = scheduler.step(noise_pred, t, latents)
    denoised = step_out.prev_sample

    # Apply skip-residual blending from lower-resolution phase
    if skip_residual is not None:
        blurred = _gaussian_blur(skip_residual, kernel_size=3)
        denoised = (1.0 - skip_weight) * denoised + skip_weight * blurred

    return denoised
