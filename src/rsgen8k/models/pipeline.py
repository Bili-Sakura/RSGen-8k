"""MegaFusion Stable Diffusion pipeline with multi-stage progressive generation.

Adapted from the MegaFusion project (Wu et al., WACV 2025):
https://github.com/haoningwu3639/MegaFusion
"""

import inspect
import torch
from typing import Callable, List, Optional, Union

from diffusers import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging
from diffusers.utils.import_utils import is_accelerate_available
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from rsgen8k.models.scheduler import MegaFusionDDIMScheduler

logger = logging.get_logger(__name__)


class MegaFusionPipeline(DiffusionPipeline):
    """Stable Diffusion pipeline extended with MegaFusion stage-based timestep support.

    This pipeline wraps the standard Stable Diffusion components and exposes a
    ``stage_timesteps`` parameter in :meth:`__call__` so that callers can run
    partial denoising schedules required by the MegaFusion multi-stage approach.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: torch.nn.Module,
        scheduler: MegaFusionDDIMScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_sequential_cpu_offload(self, gpu_id: int = 0):
        """Offload models to CPU, loading to GPU only when needed."""
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")
        device = torch.device(f"cuda:{gpu_id}")
        for model in [self.unet, self.text_encoder]:
            if model is not None:
                cpu_offload(model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device"):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt=None,
    ):
        """Encode text prompt into CLIP embeddings."""
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents: torch.FloatTensor):
        """Decode latent representations to pixel space."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, "b c h w -> b h w c").cpu().float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta: float):
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [torch.randn(shape, generator=gen, device=rand_device, dtype=dtype) for gen in generator]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        stage_timesteps: Optional[torch.Tensor] = None,
    ):
        """Run a (partial) denoising pass.

        When *stage_timesteps* is provided the denoising loop iterates only
        over those timesteps instead of the full schedule, which is the core
        mechanism that enables MegaFusion's multi-stage approach.

        Returns:
            A tuple ``(output, x0_prediction)`` of
            :class:`StableDiffusionPipelineOutput` when ``return_dict=True``.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        loop_timesteps = stage_timesteps if stage_timesteps is not None else self.scheduler.timesteps
        z_0_predict = None

        with self.progress_bar(total=len(loop_timesteps)) as progress_bar:
            for t in loop_timesteps:
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample.to(dtype=latents.dtype)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                step_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = step_output.prev_sample
                z_0_predict = step_output.pred_original_sample

                progress_bar.update(1)

        image = self.decode_latents(latents)
        x_0_predict = self.decode_latents(z_0_predict) if z_0_predict is not None else image

        if output_type == "pil":
            image = self.numpy_to_pil(image)
            x_0_predict = self.numpy_to_pil(x_0_predict)

        if not return_dict:
            return image, x_0_predict

        return (
            StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None),
            StableDiffusionPipelineOutput(images=x_0_predict, nsfw_content_detected=None),
        )
