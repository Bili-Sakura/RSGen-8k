"""Multi-stage progressive generation engine for 8K remote sensing images.

This module implements multi-stage upscaling strategies on top of various
remote sensing diffusion models (Text2Earth, DiffusionSat, GeoSynth).
The default technique is MegaFusion, but alternative methods such as
MultiDiffusion, ElasticDiffusion, FreeScale, DemoFusion, and FouriScale
are also supported.

Techniques adapted from their respective projects — see each technique
module under ``rsgen8k.techniques`` for full attribution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from rsgen8k.models.model_registry import MODEL_REGISTRY, get_model_info, resolve_model_path
from rsgen8k.techniques.registry import TECHNIQUE_REGISTRY, get_technique

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default stage configuration for scaling from 512 → 8192 (8K)
# ---------------------------------------------------------------------------
DEFAULT_STAGE_RESOLUTIONS = [512, 1024, 2048, 4096, 8192]
DEFAULT_STAGE_STEPS = [40, 3, 3, 2, 2]

TEXT2EARTH_MODEL_ID = "lcybuaa/Text2Earth"

# Upper bound for randomly chosen seeds when no explicit seed is given.
MAX_RANDOM_SEED = 2**32 - 1

# Available model and technique keys (for CLI help)
AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())
AVAILABLE_TECHNIQUES = list(TECHNIQUE_REGISTRY.keys())


@dataclass
class GenerationConfig:
    """Configuration for multi-stage high-resolution generation."""

    model_path: str = TEXT2EARTH_MODEL_ID
    model_name: str = "text2earth"
    technique: str = "megafusion"
    prompt: str = "A high-resolution satellite image of an urban area."
    negative_prompt: str = ""
    google_level: int = 18  # Text2Earth GSD: level 18 = high-res (~0.6m). Used as resolution conditioning.
    # Native technique options
    batch_size: int = 1  # Batch size for native inference (ignored by other techniques)
    native_scheduler: str = "ddim"  # Scheduler for native: ddim, euler, dpmsolver_multistep, pndm
    output_dir: str = "./outputs"
    ckpt_dir: str = "./models"
    seed: Optional[int] = None
    mixed_precision: str = "bf16"
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    stage_resolutions: List[int] = field(default_factory=lambda: list(DEFAULT_STAGE_RESOLUTIONS))
    stage_steps: List[int] = field(default_factory=lambda: list(DEFAULT_STAGE_STEPS))
    if_reschedule: bool = False
    if_dilation: bool = False
    enable_xformers: bool = True
    vae_tiling: bool = True
    deterministic: bool = False
    save_metadata: bool = True  # Save metadata.json alongside each generated image

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


def _seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Seeds ``random``, ``numpy``, and ``torch`` (CPU & CUDA) so that
    subsequent calls to any of these libraries produce the same
    sequence of random numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_generator(seed: int) -> torch.Generator:
    """Create a CPU ``torch.Generator`` for cross-platform reproducible sampling.

    Using a CPU generator is the recommended practice from diffusers
    because GPU random number generators differ across hardware.  The
    resulting tensors are then moved to the target device.
    """
    return torch.Generator(device="cpu").manual_seed(seed)


def _save_inference_metadata(
    output_path: str,
    config: GenerationConfig,
    sample_seed: int,
    resolution: int,
    prompt: Optional[str] = None,
    batch_index: Optional[int] = None,
) -> None:
    """Save inference metadata as JSON alongside the generated image.

    Writes {output_path_without_ext}_metadata.json with inference details
    for reproducibility.
    """
    if not config.save_metadata:
        return
    base, _ = os.path.splitext(output_path)
    metadata_path = f"{base}_metadata.json"
    image_filename = os.path.basename(output_path)
    metadata = {
        "image": image_filename,
        "prompt": prompt if prompt is not None else config.prompt,
        "negative_prompt": config.negative_prompt or "",
        "seed": sample_seed,
        "model_name": config.model_name,
        "model_path": config.model_path,
        "technique": config.technique,
        "resolution": resolution,
        "stage_resolutions": config.stage_resolutions,
        "stage_steps": config.stage_steps,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "if_reschedule": config.if_reschedule,
        "mixed_precision": config.mixed_precision,
        "deterministic": config.deterministic,
        "google_level": config.google_level,
    }
    if config.technique.lower() == "native":
        metadata["native_scheduler"] = config.native_scheduler
        metadata["batch_size"] = getattr(config, "batch_size", 1)
    if batch_index is not None:
        metadata["batch_index"] = batch_index
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)


# Schedulers supported for native technique (SD-compatible)
_NATIVE_SCHEDULERS = {
    "ddim": "DDIMScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "dpmsolver_multistep": "DPMSolverMultistepScheduler",
    "dpmsolver_singlestep": "DPMSolverSinglestepScheduler",
    "pndm": "PNDMScheduler",
    "lms": "LMSDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "dpm2": "KDPM2DiscreteScheduler",
    "dpm2_ancestral": "KDPM2AncestralDiscreteScheduler",
}


def _load_native_scheduler(model_path: str, scheduler_name: str):
    """Load scheduler for native technique. Uses model's DDIM config as base."""
    from diffusers import DDIMScheduler
    from diffusers.schedulers import (
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )

    name = scheduler_name.lower()
    if name not in _NATIVE_SCHEDULERS:
        available = ", ".join(sorted(_NATIVE_SCHEDULERS.keys()))
        raise ValueError(
            f"Unknown native scheduler '{scheduler_name}'. Available: {available}"
        )

    base = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    config = dict(base.config)
    # Text2Earth / SD 2.0 use epsilon prediction; DDIM config may not specify it
    if "prediction_type" not in config:
        config["prediction_type"] = "epsilon"

    cls_map = {
        "ddim": DDIMScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "dpmsolver_multistep": DPMSolverMultistepScheduler,
        "dpmsolver_singlestep": DPMSolverSinglestepScheduler,
        "pndm": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "heun": HeunDiscreteScheduler,
        "dpm2": KDPM2DiscreteScheduler,
        "dpm2_ancestral": KDPM2AncestralDiscreteScheduler,
    }
    cls = cls_map[name]
    return cls.from_config(config)


def load_pipeline(config: GenerationConfig, device: torch.device):
    """Load model components and construct the MegaFusion pipeline.

    The model is resolved from the model registry when ``config.model_name``
    is a known key; otherwise ``config.model_path`` is used directly.

    For unconditional models (e.g. ddpmcd), loads the custom pipeline and
    returns (pipeline, None, None).

    Returns:
        A tuple ``(pipeline, vae, schedulers)``. For ddpmcd, vae and schedulers
        are None; pipeline has a .generate() method.
    """
    from diffusers import AutoencoderKL, DiffusionPipeline
    from diffusers.utils.import_utils import is_xformers_available
    from transformers import AutoTokenizer, CLIPTextModel

    from rsgen8k.models.pipeline import MegaFusionPipeline
    from rsgen8k.models.scheduler import MegaFusionDDIMScheduler

    weight_dtype = _get_weight_dtype(config.mixed_precision)

    # Resolve model path from registry if possible
    model_path = config.model_path
    model_info = None
    try:
        model_info = get_model_info(config.model_name)
        model_path = model_info.model_id
        logger.info("Using registered model '%s' (%s)", model_info.name, model_path)
    except KeyError:
        logger.info("Using model path directly: %s", model_path)

    # Check for local checkpoint before downloading from Hub
    model_path = resolve_model_path(model_path, ckpt_dir=config.ckpt_dir)
    if os.path.isabs(model_path):
        logger.info("Loading from local checkpoint: %s", model_path)
    else:
        logger.info("Loading from HuggingFace Hub: %s", model_path)

    # DDPM-CD: unconditional model, uses custom pipeline (no tokenizer/text_encoder)
    if model_info and model_info.architecture == "ddpm-sr3":
        logger.info("Loading DDPM-CD pipeline (unconditional)")
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline="pipeline",
            trust_remote_code=True,
        )
        pipeline.to(device)
        # Keep in float32: custom pipeline uses float tensors internally
        if config.enable_xformers and is_xformers_available():
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                logger.warning("Could not enable xformers: %s", exc)
        return pipeline, None, None

    base_res = config.stage_resolutions[0]

    logger.info("Loading tokenizer and text encoder from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    logger.info("Loading VAE from %s", model_path)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    logger.info("Loading UNet from %s", model_path)
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    # Native technique: use selected diffusion scheduler; others use MegaFusionDDIMScheduler
    if config.technique.lower() == "native":
        schedulers = [_load_native_scheduler(model_path, config.native_scheduler)]
        logger.info("Using native %s (no rescheduling)", config.native_scheduler)
    else:
        schedulers = []
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


def _generate_ddpmcd(
    config: GenerationConfig,
    pipeline,
    device: torch.device,
    generator: Optional[torch.Generator],
    sample_seed: int,
) -> Image.Image:
    """Run unconditional DDPM-CD generation (256×256 pixel-space)."""
    os.makedirs(config.output_dir, exist_ok=True)
    num_steps = getattr(config, "num_inference_steps", 1000) or 1000
    gen = None
    if generator is not None and device.type == "cuda":
        gen = torch.Generator(device=device).manual_seed(generator.initial_seed())
    elif generator is not None:
        gen = generator

    with torch.no_grad():
        image_tensor = pipeline.generate(
            batch_size=1,
            in_channels=3,
            image_size=256,
            num_inference_steps=num_steps,
            generator=gen,
        )
    # Tensor (1,3,256,256) in [0,1] or similar; convert to PIL
    img = image_tensor.squeeze(0).cpu().float()
    if img.max() <= 1.0:
        img = (img * 255).clamp(0, 255).byte()
    else:
        img = ((img + 1) * 127.5).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()
    out = Image.fromarray(img)
    path = os.path.join(config.output_dir, f"rsgen8k_seed{sample_seed}_256px.png")
    out.save(path)
    logger.info("Saved output to %s", path)
    _save_inference_metadata(path, config, sample_seed, 256, prompt="")
    return out


def generate(config: GenerationConfig) -> Image.Image:
    """Run multi-stage generation at progressively higher resolutions.

    The technique used is determined by ``config.technique``.  The default
    is ``"megafusion"`` which uses the MegaFusion multi-stage approach.
    Other supported techniques include ``"multidiffusion"``,
    ``"elasticdiffusion"``, ``"freescale"``, ``"demofusion"``, and
    ``"fouriscale"``.

    Args:
        config: Generation configuration.

    Returns:
        The final generated :class:`PIL.Image.Image`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _get_weight_dtype(config.mixed_precision)

    # --- Reproducibility setup ---
    sample_seed = config.seed if config.seed is not None else random.randint(0, MAX_RANDOM_SEED)
    _seed_everything(sample_seed)

    if config.deterministic:
        # Enable fully deterministic CUDA algorithms (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            logger.warning("Could not enable deterministic algorithms on this platform")

    # Use a CPU generator for cross-platform reproducible noise generation.
    # GPU generators produce hardware-dependent results.
    generator = _make_generator(sample_seed)
    logger.info("Using seed %d (deterministic=%s)", sample_seed, config.deterministic)

    technique_key_lower = config.technique.lower()
    technique_key = technique_key_lower
    technique_info = get_technique(technique_key)
    logger.info("Using technique: %s", technique_info.name)

    pipeline, vae, schedulers = load_pipeline(config, device)

    # DDPM-CD: unconditional pixel-space generation
    if vae is None:
        return _generate_ddpmcd(config, pipeline, device, generator, sample_seed)

    base_res = config.stage_resolutions[0]
    if technique_key_lower == "native":
        target_res = config.stage_resolutions[-1] if config.stage_resolutions else 1024
        base_res = target_res
    shape = (1, vae.config.latent_channels, base_res // 8, base_res // 8)
    noise_latents = torch.randn(shape, generator=generator, device="cpu", dtype=weight_dtype).to(device)

    resolutions = config.stage_resolutions
    steps = config.stage_steps
    num_steps = config.num_inference_steps

    # Pre-compute timestep ranges for every stage (skipped for native, done in block)
    for sched in schedulers:
        sched.set_timesteps(num_steps, device=device)

    stage_timesteps_list: List[torch.Tensor] = []
    if technique_key_lower != "native":
        cumulative = 0
        for i, n_steps in enumerate(steps):
            ts = schedulers[i].timesteps[cumulative: cumulative + n_steps]
            stage_timesteps_list.append(ts)
            cumulative += n_steps

    os.makedirs(config.output_dir, exist_ok=True)

    used_prompts: List[str] = [config.prompt]

    with torch.no_grad():
        if technique_key == "native":
            batch_size = getattr(config, "batch_size", 1)
            res = config.stage_resolutions[-1] if config.stage_resolutions else 1024
            num_steps = config.num_inference_steps

            # Build batch prompts
            prompts = [config.prompt] * batch_size if isinstance(config.prompt, str) else config.prompt
            used_prompts = list(prompts)
            if len(prompts) != batch_size:
                prompts = (prompts * (batch_size // len(prompts) + 1))[:batch_size]

            pipeline.scheduler = schedulers[0]
            pipeline.scheduler.set_timesteps(num_steps, device=device)

            # Generator must be on same device as pipeline for prepare_latents
            gen = None
            if generator is not None and device.type == "cuda":
                gen = torch.Generator(device=device).manual_seed(generator.initial_seed())
            elif generator is not None:
                gen = generator

            image_out, x0_out = pipeline(
                prompt=prompts,
                negative_prompt=config.negative_prompt or None,
                height=res,
                width=res,
                num_inference_steps=num_steps,
                guidance_scale=config.guidance_scale,
                generator=gen,
                resolution_cond=config.google_level,
            )
            # Use decoded final latents (image_out) - canonical denoised result
            images = image_out.images  # List[PIL.Image]
            x_0_predict = images[0] if len(images) == 1 else images
        elif technique_key == "megafusion":
            from rsgen8k.techniques.megafusion import run_megafusion

            x_0_predict = run_megafusion(
                pipeline=pipeline,
                vae=vae,
                schedulers=schedulers,
                prompt=config.prompt,
                negative_prompt=config.negative_prompt or None,
                noise_latents=noise_latents,
                stage_resolutions=resolutions,
                stage_timesteps_list=stage_timesteps_list,
                num_inference_steps=num_steps,
                guidance_scale=config.guidance_scale,
                if_reschedule=config.if_reschedule,
                device=device,
                weight_dtype=weight_dtype,
                generator=generator,
                resolution_cond=config.google_level,
            )
        else:
            # For non-MegaFusion techniques, use a unified multi-stage loop
            # that delegates per-step denoising to the chosen technique.
            x_0_predict = _run_technique_multistage(
                technique_key=technique_key,
                pipeline=pipeline,
                vae=vae,
                schedulers=schedulers,
                prompt=config.prompt,
                negative_prompt=config.negative_prompt or None,
                noise_latents=noise_latents,
                stage_resolutions=resolutions,
                stage_timesteps_list=stage_timesteps_list,
                num_inference_steps=num_steps,
                guidance_scale=config.guidance_scale,
                if_reschedule=config.if_reschedule,
                resolution_cond=config.google_level,
                device=device,
                weight_dtype=weight_dtype,
                generator=generator,
            )

    out_res = resolutions[-1]
    if isinstance(x_0_predict, list):
        for i, img in enumerate(x_0_predict):
            path = os.path.join(
                config.output_dir,
                f"rsgen8k_seed{sample_seed}_{out_res}px_batch{i}.png",
            )
            img.save(path)
            logger.info("Saved output to %s", path)
            prompt_i = used_prompts[i] if i < len(used_prompts) else config.prompt
            _save_inference_metadata(
                path, config, sample_seed, out_res,
                prompt=prompt_i, batch_index=i,
            )
    else:
        output_path = os.path.join(
            config.output_dir, f"rsgen8k_seed{sample_seed}_{out_res}px.png"
        )
        x_0_predict.save(output_path)
        logger.info("Saved output to %s", output_path)
        _save_inference_metadata(output_path, config, sample_seed, out_res)

    return x_0_predict


def _run_technique_multistage(
    technique_key: str,
    pipeline,
    vae: torch.nn.Module,
    schedulers,
    prompt: str,
    negative_prompt: Optional[str],
    noise_latents: torch.FloatTensor,
    stage_resolutions: List[int],
    stage_timesteps_list: List[torch.Tensor],
    num_inference_steps: int,
    guidance_scale: float,
    if_reschedule: bool,
    resolution_cond: Optional[int] = None,
    device: torch.device = None,
    weight_dtype: torch.dtype = None,
    generator: Optional[torch.Generator] = None,
) -> Image.Image:
    """Run multi-stage generation using a per-step technique.

    This function wraps MultiDiffusion, ElasticDiffusion, FreeScale,
    DemoFusion, and FouriScale into the same multi-stage progressive
    upscaling loop used by MegaFusion (base generation → upsample →
    re-encode → denoise).  The per-step denoising behaviour is delegated
    to the selected technique's denoise function.
    """
    x_0_predict: Optional[Image.Image] = None
    prev_latents: Optional[torch.FloatTensor] = None

    for stage_idx, (res, stage_ts) in enumerate(
        zip(stage_resolutions, stage_timesteps_list)
    ):
        logger.info(
            "%s stage %d/%d: %d×%d (%d steps)",
            technique_key,
            stage_idx + 1,
            len(stage_resolutions),
            res,
            res,
            len(stage_ts),
        )

        sched = schedulers[stage_idx] if if_reschedule else schedulers[0]
        pipeline.scheduler = sched

        # Prepare latents for this stage
        if stage_idx == 0:
            latents = noise_latents
        else:
            x_0_predict = x_0_predict.resize((res, res), Image.Resampling.BICUBIC)
            latents = _encode_image(x_0_predict, vae, device, weight_dtype)
            noise = torch.randn(latents.shape, generator=generator, device="cpu", dtype=latents.dtype).to(device)
            noisy_idx = min(4, len(stage_ts) - 1) if if_reschedule else 0
            latents = pipeline.scheduler.add_noise(latents, noise, stage_ts[noisy_idx])

        # Encode text prompt
        text_embeddings = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
        )

        # Text2Earth / class conditioning: build class_labels when model requires it
        do_cfg = guidance_scale > 1.0
        class_labels = None
        unet = pipeline.unet
        needs_class_labels = (
            getattr(unet.config, "num_class_embeds", 0) > 0
            or getattr(unet, "class_embedding", None) is not None
        )
        if needs_class_labels:
            n = 1
            res_null = torch.zeros(n, dtype=latents.dtype, device=device)
            cond_val = float(resolution_cond if resolution_cond is not None else res)
            res_cond = torch.full((n,), cond_val, dtype=latents.dtype, device=device)
            class_labels = torch.cat([res_null, res_cond]) if do_cfg else res_cond

        base_latent_size = stage_resolutions[0] // 8
        total_steps = len(stage_ts)

        # Run per-step denoising with selected technique
        for step_idx, t in enumerate(stage_ts):
            if technique_key == "multidiffusion":
                from rsgen8k.techniques.multi_diffusion import multidiffusion_denoise_step

                latents = multidiffusion_denoise_step(
                    unet=pipeline.unet,
                    scheduler=pipeline.scheduler,
                    latents=latents,
                    text_embeddings=text_embeddings,
                    t=t,
                    guidance_scale=guidance_scale,
                    class_labels=class_labels,
                )
            elif technique_key == "elasticdiffusion":
                from rsgen8k.techniques.elastic_diffusion import elastic_diffusion_denoise_step

                latents = elastic_diffusion_denoise_step(
                    unet=pipeline.unet,
                    scheduler=pipeline.scheduler,
                    latents=latents,
                    text_embeddings=text_embeddings,
                    t=t,
                    guidance_scale=guidance_scale,
                    base_size=base_latent_size,
                    class_labels=class_labels,
                )
            elif technique_key == "freescale":
                from rsgen8k.techniques.freescale import freescale_denoise_step

                latents = freescale_denoise_step(
                    unet=pipeline.unet,
                    scheduler=pipeline.scheduler,
                    latents=latents,
                    text_embeddings=text_embeddings,
                    t=t,
                    guidance_scale=guidance_scale,
                    step_index=step_idx,
                    total_steps=total_steps,
                    base_size=base_latent_size,
                    class_labels=class_labels,
                )
            elif technique_key == "demofusion":
                from rsgen8k.techniques.demofusion import demofusion_denoise_step

                latents = demofusion_denoise_step(
                    unet=pipeline.unet,
                    scheduler=pipeline.scheduler,
                    latents=latents,
                    text_embeddings=text_embeddings,
                    t=t,
                    guidance_scale=guidance_scale,
                    skip_residual=prev_latents,
                    class_labels=class_labels,
                )
            elif technique_key == "fouriscale":
                from rsgen8k.techniques.fouriscale import fouriscale_denoise_step

                latents = fouriscale_denoise_step(
                    unet=pipeline.unet,
                    scheduler=pipeline.scheduler,
                    latents=latents,
                    text_embeddings=text_embeddings,
                    t=t,
                    guidance_scale=guidance_scale,
                    base_size=base_latent_size,
                    class_labels=class_labels,
                )
            else:
                raise ValueError(f"Unsupported technique: {technique_key}")

        prev_latents = latents.clone()

        # Decode latents to image
        decoded = pipeline.decode_latents(latents)
        if isinstance(decoded, torch.Tensor):
            from diffusers.pipelines.pipeline_utils import DiffusionPipeline
            x_0_predict = DiffusionPipeline.numpy_to_pil(decoded)[0]
        else:
            x_0_predict = pipeline.numpy_to_pil(decoded)[0]

    return x_0_predict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for RSGen-8k generation."""
    parser = argparse.ArgumentParser(description="RSGen-8k: Remote Sensing Image Generation at 8K Resolution")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, default=TEXT2EARTH_MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--model_name", type=str, default="text2earth",
        help=f"Registered model name. Available: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        "--technique", type=str, default="megafusion",
        help=f"Upscaling technique. Available: {', '.join(AVAILABLE_TECHNIQUES)}",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--ckpt_dir", type=str, default="./models",
                        help="Local checkpoint directory (HuggingFace repo_id layout, default: ./models)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--stage_resolutions", type=int, nargs="+", default=DEFAULT_STAGE_RESOLUTIONS)
    parser.add_argument("--stage_steps", type=int, nargs="+", default=DEFAULT_STAGE_STEPS)
    parser.add_argument("--if_reschedule", action="store_true")
    parser.add_argument("--if_dilation", action="store_true")
    parser.add_argument("--no_xformers", action="store_true")
    parser.add_argument("--no_vae_tiling", action="store_true")
    parser.add_argument("--no_save_metadata", action="store_true",
                        help="Disable saving metadata.json alongside generated images (default: save metadata)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable fully deterministic CUDA algorithms for reproducibility (may be slower)")
    parser.add_argument("--list_models", action="store_true", help="List available base models and exit")
    parser.add_argument("--list_techniques", action="store_true", help="List available techniques and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.list_models:
        from rsgen8k.models.model_registry import list_models
        print("\nAvailable base models:")
        for key, info in list_models().items():
            print(f"  {key:20s} {info.name} ({info.model_id})")
        sys.exit(0)

    if args.list_techniques:
        from rsgen8k.techniques.registry import list_techniques
        print("\nAvailable techniques:")
        for key, info in list_techniques().items():
            print(f"  {key:20s} {info.name}")
            print(f"  {'':20s} {info.description[:80]}")
        sys.exit(0)

    if args.config:
        config = GenerationConfig.from_yaml(args.config)
    else:
        config = GenerationConfig()

    # CLI arguments override config file
    for key in ("model_path", "model_name", "technique", "prompt",
                "negative_prompt", "output_dir", "ckpt_dir", "seed",
                "mixed_precision", "guidance_scale", "num_inference_steps",
                "stage_resolutions", "stage_steps", "if_reschedule", "if_dilation",
                "deterministic"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.enable_xformers = not args.no_xformers
    config.vae_tiling = not args.no_vae_tiling
    config.save_metadata = not args.no_save_metadata

    if config.prompt is None:
        logger.error("No prompt specified. Use --prompt or --config.")
        sys.exit(1)

    generate(config)


if __name__ == "__main__":
    main()
