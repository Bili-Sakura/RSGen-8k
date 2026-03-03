"""FouriScale: Frequency-perspective high-resolution synthesis.

Adapted from the FouriScale project by Linjiang Huang et al.
  Paper : "FouriScale: A Frequency Perspective on Training-Free
           High-Resolution Image Synthesis", ECCV 2024.
  Source: https://github.com/LeonHLJ/FouriScale
  License: Apache-2.0

FouriScale analyses generation from a frequency-domain perspective.
It replaces standard convolutions with dilated variants combined with
low-pass filtering in the Fourier domain, maintaining structural and
scale consistency when generating at resolutions beyond the training
distribution.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def low_pass_filter(
    x: torch.FloatTensor,
    cutoff_ratio: float = 0.5,
) -> torch.FloatTensor:
    """Apply a low-pass filter in the Fourier domain.

    Args:
        x: Input tensor ``(B, C, H, W)``.
        cutoff_ratio: Fraction of the frequency spectrum to keep
            (0 = DC only, 1 = no filtering).

    Returns:
        Low-pass filtered tensor with the same shape.
    """
    _, _, h, w = x.shape
    freq = torch.fft.fft2(x.float())
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

    # Build a circular mask
    cy, cx = h // 2, w // 2
    ry = int(cy * cutoff_ratio)
    rx = int(cx * cutoff_ratio)
    mask = torch.zeros(h, w, device=x.device)
    y_coords = torch.arange(h, device=x.device).unsqueeze(1).float() - cy
    x_coords = torch.arange(w, device=x.device).unsqueeze(0).float() - cx
    dist = (y_coords / max(ry, 1)) ** 2 + (x_coords / max(rx, 1)) ** 2
    mask[dist <= 1.0] = 1.0

    filtered = freq_shifted * mask.unsqueeze(0).unsqueeze(0)
    result = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1))).real
    return result.to(dtype=x.dtype)


def fouriscale_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    base_size: int = 64,
    cutoff_ratio: float = 0.5,
) -> torch.FloatTensor:
    """Perform one FouriScale denoising step.

    The step combines a full-resolution UNet pass with a low-pass
    filtered version of the prediction to suppress high-frequency
    artifacts introduced by out-of-distribution resolutions.

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text (with unconditional if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        base_size: Base latent resolution (unused in single-pass variant
            but kept for API symmetry with other techniques).
        cutoff_ratio: Low-pass filter cutoff ratio.

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

    # Apply low-pass filter to suppress high-frequency artifacts
    noise_pred = low_pass_filter(noise_pred, cutoff_ratio=cutoff_ratio)

    step_out = scheduler.step(noise_pred, t, latents)
    return step_out.prev_sample
