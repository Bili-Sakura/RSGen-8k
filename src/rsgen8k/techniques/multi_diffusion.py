"""MultiDiffusion: Panoramic generation via overlapping view fusion.

Adapted from the MultiDiffusion project by Omer Bar-Tal et al.
  Paper : "MultiDiffusion: Fusing Diffusion Paths for Controlled Image
           Generation", ICML 2023.
  Source: https://github.com/omerbt/MultiDiffusion
  License: MIT

MultiDiffusion generates high-resolution images by running independent
denoising paths on overlapping views (patches) of the latent canvas,
then fusing them via a weighted-average consensus step at each
denoising iteration.  This ensures global coherence while allowing
generation at arbitrary resolutions.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _get_views(
    height: int,
    width: int,
    window_size: int = 64,
    stride: int = 16,
) -> List[Tuple[int, int, int, int]]:
    """Compute sliding-window view coordinates in latent space.

    Returns:
        List of ``(row_start, row_end, col_start, col_end)`` tuples.
    """
    views = []
    h_starts = list(range(0, max(height - window_size + 1, 1), stride))
    w_starts = list(range(0, max(width - window_size + 1, 1), stride))
    # Ensure last window reaches the edge
    if h_starts[-1] + window_size < height:
        h_starts.append(height - window_size)
    if w_starts[-1] + window_size < width:
        w_starts.append(width - window_size)
    for h in h_starts:
        for w in w_starts:
            views.append((h, h + window_size, w, w + window_size))
    return views


def multidiffusion_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    window_size: int = 64,
    stride: int = 16,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    """Perform one MultiDiffusion denoising step.

    The global latent is split into overlapping views.  Each view is
    denoised independently, and the results are fused via weighted
    averaging (consensus voting).

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler with a ``step`` method.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text prompt (with unconditional if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        window_size: View window size in latent pixels.
        stride: Sliding-window stride in latent pixels.

    Returns:
        Updated latents after one denoising step.
    """
    _, _, h, w = latents.shape
    views = _get_views(h, w, window_size=window_size, stride=stride)

    count = torch.zeros_like(latents)
    value = torch.zeros_like(latents)

    do_cfg = guidance_scale > 1.0

    for r_start, r_end, c_start, c_end in views:
        view_latents = latents[:, :, r_start:r_end, c_start:c_end]

        latent_input = (
            torch.cat([view_latents] * 2) if do_cfg else view_latents
        )
        latent_input = scheduler.scale_model_input(latent_input, t)

        unet_kwargs = {"encoder_hidden_states": text_embeddings}
        if class_labels is not None:
            unet_kwargs["class_labels"] = class_labels
        noise_pred = unet(latent_input, t, **unet_kwargs).sample.to(dtype=latents.dtype)

        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)

        step_out = scheduler.step(noise_pred, t, view_latents)
        denoised = step_out.prev_sample

        value[:, :, r_start:r_end, c_start:c_end] += denoised
        count[:, :, r_start:r_end, c_start:c_end] += 1

    latents = torch.where(count > 0, value / count, latents)
    return latents
