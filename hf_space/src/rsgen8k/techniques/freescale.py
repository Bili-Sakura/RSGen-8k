"""FreeScale: Tuning-free scale fusion via attention modification.

Adapted from the FreeScale project by Haonan Qiu et al.
  Paper : "FreeScale: Unleashing the Resolution of Diffusion Models via
           Tuning-Free Scale Fusion", arXiv:2412.09626, 2024.
  Source: https://github.com/ali-vilab/FreeScale
  License: Apache-2.0

FreeScale modifies the self-attention layers to apply cosine-scheduled
scale fusion, blending attention outputs computed at the native
resolution with those from a down-scaled reference.  This prevents
repetitive patterns at high resolutions while preserving fine detail.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cosine_scale_schedule(
    current_step: int,
    total_steps: int,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> float:
    """Compute a cosine-annealed blending weight for scale fusion.

    Returns a weight in ``[min_scale, max_scale]`` that starts high
    (favouring the global / low-resolution branch) and decreases toward
    ``min_scale`` (favouring the local / high-resolution branch).

    Args:
        current_step: Current denoising step index (0-based).
        total_steps: Total number of denoising steps.
        min_scale: Minimum blending weight.
        max_scale: Maximum blending weight.

    Returns:
        Blending weight for the current step.
    """
    if total_steps <= 1:
        return min_scale
    ratio = current_step / (total_steps - 1)
    weight = min_scale + 0.5 * (max_scale - min_scale) * (1 + math.cos(math.pi * ratio))
    return weight


def freescale_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    step_index: int,
    total_steps: int,
    base_size: int = 64,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    """Perform one FreeScale denoising step with scale fusion.

    At each step a blending weight is computed via cosine scheduling.
    Two UNet forward passes are run: one at the full (high) resolution
    and one at the base (low) resolution.  The noise predictions are
    blended according to the schedule before applying the scheduler
    step.

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text (with unconditional if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        step_index: Current step index in the denoising loop.
        total_steps: Total denoising steps.
        base_size: Base latent resolution for the low-res branch.

    Returns:
        Updated latents after one denoising step.
    """
    _, _, h, w = latents.shape
    do_cfg = guidance_scale > 1.0
    scale_weight = cosine_scale_schedule(step_index, total_steps)

    # ---- High-resolution (local) prediction ----
    unet_kwargs = {"encoder_hidden_states": text_embeddings}
    if class_labels is not None:
        unet_kwargs["class_labels"] = class_labels
    hi_input = torch.cat([latents] * 2) if do_cfg else latents
    hi_input = scheduler.scale_model_input(hi_input, t)
    hi_pred = unet(hi_input, t, **unet_kwargs).sample.to(dtype=latents.dtype)
    if do_cfg:
        u, c = hi_pred.chunk(2)
        hi_pred = u + guidance_scale * (c - u)

    # ---- Low-resolution (global) prediction ----
    lo_latents = F.interpolate(
        latents, size=(base_size, base_size), mode="bilinear", align_corners=False
    )
    lo_input = torch.cat([lo_latents] * 2) if do_cfg else lo_latents
    lo_input = scheduler.scale_model_input(lo_input, t)
    lo_pred = unet(lo_input, t, **unet_kwargs).sample.to(dtype=latents.dtype)
    if do_cfg:
        u, c = lo_pred.chunk(2)
        lo_pred = u + guidance_scale * (c - u)
    lo_pred_up = F.interpolate(
        lo_pred, size=(h, w), mode="bilinear", align_corners=False
    )

    # ---- Scale fusion ----
    fused_pred = scale_weight * lo_pred_up + (1.0 - scale_weight) * hi_pred

    step_out = scheduler.step(fused_pred, t, latents)
    return step_out.prev_sample
