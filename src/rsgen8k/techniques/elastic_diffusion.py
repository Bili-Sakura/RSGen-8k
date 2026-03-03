"""ElasticDiffusion: Training-free arbitrary-size generation.

Adapted from the ElasticDiffusion project by Moayed Haji Ali et al.
  Paper : "ElasticDiffusion: Training-free Arbitrary Size Image Generation
           through Global-Local Content Separation", CVPR 2024.
  Source: https://github.com/MoayedHajiAli/ElasticDiffusion-official
  License: MIT

ElasticDiffusion decouples generation into local detail signals
(processed on overlapping patches) and global structural signals
(obtained from a reduced-resolution guidance pass).  The two signals
are blended at each denoising step, enabling coherent generation at
arbitrary image sizes without retraining.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def _get_views(
    height: int,
    width: int,
    window_size: int = 64,
    stride: int = 16,
) -> List[Tuple[int, int, int, int]]:
    """Compute overlapping view coordinates in latent space."""
    views = []
    h_starts = list(range(0, max(height - window_size + 1, 1), stride))
    w_starts = list(range(0, max(width - window_size + 1, 1), stride))
    if h_starts[-1] + window_size < height:
        h_starts.append(height - window_size)
    if w_starts[-1] + window_size < width:
        w_starts.append(width - window_size)
    for h in h_starts:
        for w in w_starts:
            views.append((h, h + window_size, w, w + window_size))
    return views


def elastic_diffusion_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    base_size: int = 64,
    window_size: int = 64,
    stride: int = 16,
    global_weight: float = 0.5,
) -> torch.FloatTensor:
    """Perform one ElasticDiffusion denoising step.

    1. **Global pass**: down-sample latents to base resolution, denoise,
       and up-sample back.
    2. **Local pass**: denoise overlapping patches independently and fuse
       via weighted averaging.
    3. **Blend**: combine global and local predictions.

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler with ``step`` and ``scale_model_input``.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text (with unconditional if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        base_size: Base latent resolution for the global pass.
        window_size: Patch window size for the local pass.
        stride: Sliding-window stride for the local pass.
        global_weight: Blending weight for the global signal (0–1).

    Returns:
        Updated latents after one denoising step.
    """
    _, _, h, w = latents.shape
    do_cfg = guidance_scale > 1.0

    # ---- Global pass: reduced-resolution guidance ----
    global_latents = F.interpolate(
        latents, size=(base_size, base_size), mode="bilinear", align_corners=False
    )
    global_input = (
        torch.cat([global_latents] * 2) if do_cfg else global_latents
    )
    global_input = scheduler.scale_model_input(global_input, t)
    global_pred = unet(
        global_input, t, encoder_hidden_states=text_embeddings
    ).sample.to(dtype=latents.dtype)
    if do_cfg:
        u, c = global_pred.chunk(2)
        global_pred = u + guidance_scale * (c - u)
    global_step = scheduler.step(global_pred, t, global_latents).prev_sample
    global_upsampled = F.interpolate(
        global_step, size=(h, w), mode="bilinear", align_corners=False
    )

    # ---- Local pass: patch-wise denoising with consensus fusion ----
    views = _get_views(h, w, window_size=window_size, stride=stride)
    count = torch.zeros_like(latents)
    value = torch.zeros_like(latents)

    for r_start, r_end, c_start, c_end in views:
        patch = latents[:, :, r_start:r_end, c_start:c_end]
        patch_input = torch.cat([patch] * 2) if do_cfg else patch
        patch_input = scheduler.scale_model_input(patch_input, t)
        pred = unet(
            patch_input, t, encoder_hidden_states=text_embeddings
        ).sample.to(dtype=latents.dtype)
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + guidance_scale * (c - u)
        denoised = scheduler.step(pred, t, patch).prev_sample
        value[:, :, r_start:r_end, c_start:c_end] += denoised
        count[:, :, r_start:r_end, c_start:c_end] += 1

    local_result = torch.where(count > 0, value / count, latents)

    # ---- Blend global and local ----
    blended = global_weight * global_upsampled + (1.0 - global_weight) * local_result
    return blended
