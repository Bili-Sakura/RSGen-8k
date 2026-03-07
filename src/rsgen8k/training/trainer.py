"""Diffusion-4K training orchestrator for Stable Diffusion models.

Adapted from the Diffusion-4K project by Jinjin Zhang et al.
  Paper : "Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent
           Diffusion Models", CVPR 2025.
  Source: https://github.com/zhang0jhon/diffusion-4k
  License: Apache-2.0

This module adapts the wavelet-based fine-tuning approach from Diffusion-4K
for Stable Diffusion v1.x/v2.x UNet models used by RSGen-8k.  The original
Diffusion-4K targets FLUX and SD3 architectures with flow matching; this
adaptation uses DDPM/DDIM noise scheduling while retaining the wavelet-domain
loss that is core to the technique.

Usage (via accelerate + DeepSpeed)::

    accelerate launch --config_file configs/ds_config.yaml \\
        -m rsgen8k.training.trainer \\
        --pretrained_model_name_or_path ./models/lcybuaa/Text2Earth \\
        --instance_data_dir ./data/train \\
        --output_dir ./checkpoints/diffusion4k \\
        --resolution 2048 \\
        --train_batch_size 2 \\
        --max_train_steps 10000 \\
        --learning_rate 1e-6
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageCaptionDataset(Dataset):
    """Simple dataset that loads images and their text captions.

    Images are read from *data_dir*.  Captions are loaded from a
    sidecar ``.txt`` file with the same stem, or from the EXIF
    ``ImageDescription`` tag.  When no caption is available the filename
    (without extension) is used.

    Args:
        data_dir: Directory containing image files.
        resolution: Target resolution; images are center-cropped to this size.
        file_extensions: Allowed image suffixes.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        file_extensions: Optional[set] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        exts = file_extensions or self.SUPPORTED_EXTENSIONS

        self.image_paths: List[Path] = sorted(
            p for p in self.data_dir.rglob("*") if p.suffix.lower() in exts
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {data_dir} with extensions {exts}"
            )

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_caption(self, image_path: Path) -> str:
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()
        return image_path.stem.replace("_", " ")

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        caption = self._load_caption(img_path)
        return {"pixel_values": pixel_values, "caption": caption}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_args(input_args=None) -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Stable Diffusion UNet with Diffusion-4K wavelet loss."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path to pretrained SD model or HuggingFace model ID.",
    )
    parser.add_argument(
        "--instance_data_dir", type=str, required=True,
        help="Directory containing training images (with optional .txt captions).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/diffusion4k",
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "cosine", "linear"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard"])
    parser.add_argument(
        "--wave", type=str, default="haar",
        help="Wavelet family for the DWT loss (default: haar).",
    )
    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Run the Diffusion-4K wavelet fine-tuning loop.

    This function sets up the Accelerate-based training loop for
    fine-tuning a Stable Diffusion UNet with wavelet-domain loss.
    """
    if args is None:
        args = parse_args()

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    from rsgen8k.training.wavelet_loss import WaveletLoss

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model components ----
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )

    # Freeze everything except UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # ---- Optimizer ----
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, install bitsandbytes: pip install bitsandbytes"
            )
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # ---- Dataset & DataLoader ----
    dataset = ImageCaptionDataset(
        data_dir=args.instance_data_dir,
        resolution=args.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- LR scheduler ----
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.max_train_steps or (
        args.num_train_epochs * num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ---- Wavelet loss ----
    wavelet_loss_fn = WaveletLoss(wave=args.wave)

    # ---- Prepare with Accelerator ----
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler,
    )

    # ---- Tokenisation helper ----
    def tokenize_caption(caption: str) -> torch.Tensor:
        inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # ---- Training ----
    global_step = 0
    logger.info("***** Running Diffusion-4K wavelet fine-tuning *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Max train steps = %d", max_train_steps)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Wavelet = %s", args.wave)

    unet.train()

    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                captions = batch["caption"]

                # Encode images → latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode text
                input_ids = torch.cat(
                    [tokenize_caption(c) for c in captions]
                ).to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0].to(
                    dtype=weight_dtype
                )

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )

                # Add noise (DDPM forward process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Convert prediction to x0 estimate for wavelet loss
                # (following Diffusion-4K: compare x0 predictions in wavelet domain)
                if noise_scheduler.config.prediction_type == "epsilon":
                    # x0 = (noisy - sqrt(1-alpha) * eps) / sqrt(alpha)
                    alpha_prod_t = noise_scheduler.alphas_cumprod.to(
                        device=timesteps.device
                    )[timesteps]
                    while alpha_prod_t.ndim < noisy_latents.ndim:
                        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
                    pred_x0 = (
                        noisy_latents - (1 - alpha_prod_t).sqrt() * model_pred
                    ) / alpha_prod_t.sqrt()
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    alpha_prod_t = noise_scheduler.alphas_cumprod.to(
                        device=timesteps.device
                    )[timesteps]
                    while alpha_prod_t.ndim < noisy_latents.ndim:
                        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
                    pred_x0 = (
                        alpha_prod_t.sqrt() * noisy_latents
                        - (1 - alpha_prod_t).sqrt() * model_pred
                    )
                else:
                    # sample prediction — model directly predicts x0
                    pred_x0 = model_pred

                # Wavelet loss: compare predicted x0 vs ground-truth latents
                loss = wavelet_loss_fn(pred_x0, latents)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 100 == 0:
                    logger.info(
                        "Step %d / %d  loss=%.6f  lr=%.2e",
                        global_step,
                        max_train_steps,
                        loss.detach().item(),
                        lr_scheduler.get_last_lr()[0],
                    )

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info("Saved checkpoint to %s", save_path)

            if global_step >= max_train_steps:
                break

    # ---- Save final model ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet_unwrapped,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info("Saved final model to %s", args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
