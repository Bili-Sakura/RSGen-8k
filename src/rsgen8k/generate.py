"""Multi-stage progressive generation engine for 8K remote sensing images.

This module implements the MegaFusion multi-stage upscaling strategy
on top of the Text2Earth diffusion model.  The approach generates an
image at the model's native resolution and then progressively upscales
through intermediate resolutions up to 8K (8192 × 8192).

Adapted from the MegaFusion project (Wu et al., WACV 2025):
https://github.com/haoningwu3639/MegaFusion
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default stage configuration for scaling from 512 → 8192 (8K)
# ---------------------------------------------------------------------------
DEFAULT_STAGE_RESOLUTIONS = [512, 1024, 2048, 4096, 8192]
DEFAULT_STAGE_STEPS = [40, 3, 3, 2, 2]

TEXT2EARTH_MODEL_ID = "lcybuaa/Text2Earth"


@dataclass
class GenerationConfig:
    """Configuration for multi-stage MegaFusion generation."""

    model_path: str = TEXT2EARTH_MODEL_ID
    prompt: str = "A high-resolution satellite image of an urban area."
    negative_prompt: str = ""
    output_dir: str = "./outputs"
    seed: Optional[int] = None
    mixed_precision: str = "fp16"
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    stage_resolutions: List[int] = field(default_factory=lambda: list(DEFAULT_STAGE_RESOLUTIONS))
    stage_steps: List[int] = field(default_factory=lambda: list(DEFAULT_STAGE_STEPS))
    if_reschedule: bool = False
    if_dilation: bool = False
    enable_xformers: bool = True
    vae_tiling: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "GenerationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_pipeline(config: GenerationConfig, device: torch.device):
    """Load Text2Earth model components and construct the MegaFusion pipeline.

    Args:
        config: Generation configuration.
        device: Target torch device.

    Returns:
        A tuple ``(pipeline, vae, schedulers)`` where *schedulers* is a list
        of :class:`MegaFusionDDIMScheduler` instances (one per stage).
    """
    from diffusers import AutoencoderKL
    from diffusers.utils.import_utils import is_xformers_available
    from transformers import AutoTokenizer, CLIPTextModel

    from rsgen8k.models.pipeline import MegaFusionPipeline
    from rsgen8k.models.scheduler import MegaFusionDDIMScheduler

    weight_dtype = _get_weight_dtype(config.mixed_precision)
    model_path = config.model_path
    base_res = config.stage_resolutions[0]

    logger.info("Loading tokenizer and text encoder from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    logger.info("Loading VAE from %s", model_path)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    logger.info("Loading UNet from %s", model_path)
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # Build one scheduler per resolution stage
    schedulers: list[MegaFusionDDIMScheduler] = []
    for res in config.stage_resolutions:
        sched = MegaFusionDDIMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            base_resolution=base_res,
            target_resolution=res,
        )
        schedulers.append(sched)

    pipeline = MegaFusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=schedulers[0],
    )

    if config.enable_xformers and is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning("Could not enable xformers: %s", exc)

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    vae.eval()
    unet.eval()
    text_encoder.eval()

    if config.vae_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    return pipeline, vae, schedulers


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


def generate(config: GenerationConfig) -> Image.Image:
    """Run multi-stage MegaFusion generation at progressively higher resolutions.

    This is the main entry point for generation.  It performs the following
    stages:

    1. Generate at base resolution (e.g. 512 × 512) using the full early
       portion of the denoising schedule.
    2. For each subsequent resolution, bicubic-upsample the predicted clean
       image, re-encode to latent space, add noise at the appropriate
       timestep, and denoise for a small number of steps.

    Args:
        config: Generation configuration.

    Returns:
        The final generated :class:`PIL.Image.Image`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _get_weight_dtype(config.mixed_precision)

    pipeline, vae, schedulers = load_pipeline(config, device)

    sample_seed = config.seed if config.seed is not None else random.randint(0, 100_000)
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    logger.info("Using seed %d", sample_seed)

    base_res = config.stage_resolutions[0]
    shape = (1, vae.config.latent_channels, base_res // 8, base_res // 8)
    noise_latents = torch.randn(shape, generator=generator, device=device, dtype=weight_dtype)

    resolutions = config.stage_resolutions
    steps = config.stage_steps
    num_steps = config.num_inference_steps

    # Pre-compute timestep ranges for every stage
    for sched in schedulers:
        sched.set_timesteps(num_steps, device=device)

    cumulative = 0
    stage_timesteps_list: list[torch.Tensor] = []
    for i, n_steps in enumerate(steps):
        ts = schedulers[i].timesteps[cumulative: cumulative + n_steps]
        stage_timesteps_list.append(ts)
        cumulative += n_steps

    os.makedirs(config.output_dir, exist_ok=True)
    x_0_predict: Optional[Image.Image] = None

    with torch.no_grad():
        for stage_idx, (res, stage_ts) in enumerate(zip(resolutions, stage_timesteps_list)):
            logger.info(
                "Stage %d/%d: resolution %d×%d  (%d steps)",
                stage_idx + 1,
                len(resolutions),
                res,
                res,
                len(stage_ts),
            )

            if stage_idx == 0:
                # First stage: generate from pure noise
                pipeline.scheduler = schedulers[0]
                _, x0_out = pipeline(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt or None,
                    height=res,
                    width=res,
                    latents=noise_latents,
                    num_inference_steps=num_steps,
                    guidance_scale=config.guidance_scale,
                    stage_timesteps=stage_ts,
                )
                x_0_predict = x0_out.images[0]
            else:
                # Subsequent stages: upsample → encode → add noise → denoise
                x_0_predict = x_0_predict.resize((res, res), Image.Resampling.BICUBIC)
                latents = _encode_image(x_0_predict, vae, device, weight_dtype)
                noise = torch.randn_like(latents)

                sched = schedulers[stage_idx]
                pipeline.scheduler = sched if config.if_reschedule else schedulers[0]

                noisy_timestep_idx = min(4, len(stage_ts) - 1) if config.if_reschedule else 0
                latents_noisy = pipeline.scheduler.add_noise(latents, noise, stage_ts[noisy_timestep_idx])

                _, x0_out = pipeline(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt or None,
                    height=res,
                    width=res,
                    latents=latents_noisy,
                    num_inference_steps=num_steps,
                    guidance_scale=config.guidance_scale,
                    stage_timesteps=stage_ts,
                )
                x_0_predict = x0_out.images[0]

    output_path = os.path.join(config.output_dir, f"rsgen8k_seed{sample_seed}_{resolutions[-1]}px.png")
    x_0_predict.save(output_path)
    logger.info("Saved output to %s", output_path)

    return x_0_predict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for RSGen-8k generation."""
    parser = argparse.ArgumentParser(description="RSGen-8k: Remote Sensing Image Generation at 8K Resolution")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, default=TEXT2EARTH_MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--stage_resolutions", type=int, nargs="+", default=DEFAULT_STAGE_RESOLUTIONS)
    parser.add_argument("--stage_steps", type=int, nargs="+", default=DEFAULT_STAGE_STEPS)
    parser.add_argument("--if_reschedule", action="store_true")
    parser.add_argument("--if_dilation", action="store_true")
    parser.add_argument("--no_xformers", action="store_true")
    parser.add_argument("--no_vae_tiling", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.config:
        config = GenerationConfig.from_yaml(args.config)
    else:
        config = GenerationConfig()

    # CLI arguments override config file
    for key in ("model_path", "prompt", "negative_prompt", "output_dir", "seed",
                "mixed_precision", "guidance_scale", "num_inference_steps",
                "stage_resolutions", "stage_steps", "if_reschedule", "if_dilation"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.enable_xformers = not args.no_xformers
    config.vae_tiling = not args.no_vae_tiling

    if config.prompt is None:
        logger.error("No prompt specified. Use --prompt or --config.")
        sys.exit(1)

    generate(config)


if __name__ == "__main__":
    main()
