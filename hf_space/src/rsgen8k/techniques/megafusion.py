"""MegaFusion: Multi-stage progressive high-resolution generation.

Adapted from the MegaFusion project by Haoning Wu et al.
  Paper : "MegaFusion: Extend Diffusion Models towards Higher-resolution
           Image Generation without Further Tuning", WACV 2025.
  Source: https://github.com/haoningwu3639/MegaFusion
  License: Apache-2.0

MegaFusion decomposes the denoising process into multiple stages at
increasing resolutions.  At each stage the predicted clean image is
bicubic-upsampled, re-encoded into latent space, noise is added at the
appropriate timestep, and a few denoising steps are run.  An optional
noise-rescheduling formula adjusts the noise schedule based on the
resolution scale factor.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from rsgen8k.models.scheduler import MegaFusionDDIMScheduler

logger = logging.getLogger(__name__)

# Re-export the scheduler and pipeline used by MegaFusion for convenience.
__all__ = ["run_megafusion"]


def _encode_image(
    image: Image.Image,
    vae: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    """Encode a PIL image into VAE latent space."""
    tensor = transforms.ToTensor()(image).float() * 2.0 - 1.0
    tensor = tensor.unsqueeze(0).to(device, dtype=dtype)
    latents = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
    return latents


def run_megafusion(
    pipeline,
    vae: torch.nn.Module,
    schedulers: List[MegaFusionDDIMScheduler],
    prompt: str,
    negative_prompt: Optional[str],
    noise_latents: torch.FloatTensor,
    stage_resolutions: List[int],
    stage_timesteps_list: List[torch.Tensor],
    num_inference_steps: int,
    guidance_scale: float,
    if_reschedule: bool,
    device: torch.device,
    weight_dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
    resolution_cond: Optional[int] = None,
) -> Image.Image:
    """Execute the MegaFusion multi-stage generation loop.

    Args:
        pipeline: A :class:`MegaFusionPipeline` instance.
        vae: The VAE module (already on *device*).
        schedulers: One scheduler per resolution stage.
        prompt: Text prompt for generation.
        negative_prompt: Optional negative prompt.
        noise_latents: Initial noise tensor for the base stage.
        stage_resolutions: Resolution for each stage.
        stage_timesteps_list: Pre-computed timestep slices per stage.
        num_inference_steps: Total inference steps (for scheduler init).
        guidance_scale: Classifier-free guidance scale.
        if_reschedule: Whether to apply noise rescheduling.
        device: Torch device.
        weight_dtype: Model weight dtype.

    Returns:
        The final generated :class:`PIL.Image.Image`.
    """
    x_0_predict: Optional[Image.Image] = None

    for stage_idx, (res, stage_ts) in enumerate(
        zip(stage_resolutions, stage_timesteps_list)
    ):
        logger.info(
            "MegaFusion stage %d/%d: %d×%d (%d steps)",
            stage_idx + 1,
            len(stage_resolutions),
            res,
            res,
            len(stage_ts),
        )

        pipeline_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=res,
            width=res,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            stage_timesteps=stage_ts,
        )
        if resolution_cond is not None:
            pipeline_kwargs["resolution_cond"] = resolution_cond

        if stage_idx == 0:
            pipeline.scheduler = schedulers[0]
            pipeline_kwargs["latents"] = noise_latents
            _, x0_out = pipeline(**pipeline_kwargs)
            x_0_predict = x0_out.images[0]
        else:
            x_0_predict = x_0_predict.resize(
                (res, res), Image.Resampling.BICUBIC
            )
            latents = _encode_image(x_0_predict, vae, device, weight_dtype)
            noise = torch.randn(latents.shape, generator=generator, device="cpu", dtype=latents.dtype).to(device)

            sched = schedulers[stage_idx]
            pipeline.scheduler = sched if if_reschedule else schedulers[0]

            noisy_idx = min(4, len(stage_ts) - 1) if if_reschedule else 0
            latents_noisy = pipeline.scheduler.add_noise(
                latents, noise, stage_ts[noisy_idx]
            )

            pipeline_kwargs["latents"] = latents_noisy
            _, x0_out = pipeline(**pipeline_kwargs)
            x_0_predict = x0_out.images[0]

    return x_0_predict
