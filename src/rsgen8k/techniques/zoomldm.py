"""ZoomLDM: multi-scale latent diffusion with SSL conditioning.

Adapted from ZoomLDM-Diffusers by Bili-Sakura (fork of cvlab-stonybrook/ZoomLDM).
  Paper: Yellapragada et al., "ZoomLDM: Latent Diffusion Model for
         Multi-scale Image Generation", CVPR 2025.
  Source: https://github.com/Bili-Sakura/ZoomLDM-Diffusers

ZoomLDM uses self-supervised embeddings (DINO-v2 for satellite/NAIP) and
magnification levels for conditioning, not text prompts. Inference requires
pre-extracted SSL features and a magnification level (0=1x, 1=2x, 2=3x, 3=4x).

This module provides:
- zoomldm_denoise_step: Per-step denoising with context conditioning (for
  potential use in custom multi-stage flows).
- run_zoomldm_generation: Standalone generation with ZoomLDMPipeline given
  ssl_features and magnification.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

# ZoomLDM magnification levels: 0=1x, 1=2x, 2=3x, 3=4x
MAG_LEVELS = (0, 1, 2, 3)


def zoomldm_denoise_step(
    unet: torch.nn.Module,
    scheduler: object,
    latents: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float = 2.0,
    conditioning_key: str = "crossattn",
) -> torch.Tensor:
    """Perform one denoising step with ZoomLDM UNet (context conditioning).

    ZoomLDM uses SSL-derived context instead of text embeddings. The UNet
    expects ``context`` from the conditioning encoder (EmbeddingViT2_5),
    not encoder_hidden_states from a text encoder.

    Args:
        unet: ZoomLDM UNet (openaimodel.UNetModel).
        scheduler: DDIM scheduler with scale_model_input and step.
        latents: Current noisy latents, shape (B, C, H, W).
        context: Conditioning from EmbeddingViT2_5, shape (2*B, seq, dim)
            when using classifier-free guidance.
        t: Current timestep tensor.
        guidance_scale: CFG scale factor.
        conditioning_key: "crossattn" or "concat".

    Returns:
        Updated latents after one denoising step.
    """
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if conditioning_key == "crossattn":
        noise_pred = unet(latent_model_input, t, context=context)
    elif conditioning_key == "concat":
        noise_pred = unet(
            torch.cat([latent_model_input, context], dim=1),
            t,
        )
    else:
        noise_pred = unet(latent_model_input, t)

    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


def run_zoomldm_generation(
    pipeline,
    ssl_features: Union[torch.Tensor, list],
    magnification: Union[int, torch.Tensor],
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    latent_shape: tuple = (3, 64, 64),
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> list:
    """Run ZoomLDM generation with SSL features and magnification.

    Uses the ZoomLDMPipeline directly. Output is 256×256 by default
    (latent 64×64 with scale 4). For larger images, use the joint
    multi-scale sampling from the ZoomLDM notebooks.

    Args:
        pipeline: ZoomLDMPipeline instance (from_pretrained or from_single_file).
        ssl_features: SSL feature tensor(s). Shape (B, C, H, W) or list of
            tensors for variable-size conditioning.
        magnification: Integer or tensor, 0–3 (1x, 2x, 3x, 4x).
        num_inference_steps: Denoising steps.
        guidance_scale: CFG scale.
        latent_shape: Latent shape per sample (default 3×64×64 → 256px).
        generator: Optional RNG.
        device: Device for generation.

    Returns:
        List of PIL images.
    """
    if device is None:
        device = next(pipeline.unet.parameters()).device

    if isinstance(magnification, int):
        magnification = torch.tensor([magnification], dtype=torch.long, device=device)
    elif isinstance(magnification, torch.Tensor):
        magnification = magnification.to(device)
    else:
        magnification = torch.tensor([int(magnification)], dtype=torch.long, device=device)

    output = pipeline(
        ssl_features=ssl_features,
        magnification=magnification,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        latent_shape=latent_shape,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )
    images = output.images
    if not isinstance(images, list):
        images = [images]
    return images
