"""Custom DDIM scheduler with noise rescheduling for MegaFusion.

Adapted from the MegaFusion project (Wu et al., WACV 2025):
https://github.com/haoningwu3639/MegaFusion
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """Output class for the scheduler's step function.

    Attributes:
        prev_sample: Computed sample ``(x_{t-1})`` of previous timestep.
        pred_original_sample: The predicted denoised sample ``(x_0)``.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Create a beta schedule that discretizes the given alpha_t_bar function."""

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Noise rescheduling formulas from the MegaFusion paper.
# Each resolution factor maps to coefficients (a, b) such that:
#     alphas_cumprod_new = alphas_cumprod * 1.0 / (a - b * alphas_cumprod)
# For base resolution (factor=1), no rescheduling is applied.
RESCHEDULE_COEFFICIENTS = {
    1.0: None,       # base resolution – no rescheduling
    1.5: (2.25, 1.25),
    2.0: (4.0, 3.0),
    3.0: (9.0, 8.0),
    4.0: (16.0, 15.0),
}


def compute_rescheduled_alphas_cumprod(
    alphas_cumprod: torch.Tensor,
    base_resolution: int,
    target_resolution: int,
) -> torch.Tensor:
    """Apply noise rescheduling for a given resolution scaling factor.

    The formula follows the MegaFusion paper: for a resolution scale factor
    *r*, the rescheduled cumulative product of alphas is::

        alphas_cumprod' = alphas_cumprod / (r^2 - (r^2 - 1) * alphas_cumprod)

    Args:
        alphas_cumprod: Original cumulative product of alphas.
        base_resolution: Base training resolution of the diffusion model.
        target_resolution: Target generation resolution for this stage.

    Returns:
        Rescheduled cumulative product of alphas.
    """
    factor = target_resolution / base_resolution
    if factor <= 1.0:
        return alphas_cumprod

    # Use pre-defined coefficients when available, otherwise compute from
    # the general formula  r^2 / (r^2 - (r^2 - 1) * alpha).
    if factor in RESCHEDULE_COEFFICIENTS and RESCHEDULE_COEFFICIENTS[factor] is not None:
        a, b = RESCHEDULE_COEFFICIENTS[factor]
    else:
        r2 = factor * factor
        a, b = r2, r2 - 1.0

    return alphas_cumprod * 1.0 / (a - b * alphas_cumprod)


class MegaFusionDDIMScheduler(SchedulerMixin, ConfigMixin):
    """DDIM scheduler extended with MegaFusion noise rescheduling.

    This scheduler wraps the standard DDIM algorithm and adds support for
    resolution-dependent noise rescheduling as described in MegaFusion
    (Wu et al., WACV 2025).

    Args:
        num_train_timesteps: Number of diffusion steps used to train the model.
        beta_start: The starting beta value of inference.
        beta_end: The final beta value.
        beta_schedule: The beta schedule type.
        trained_betas: Array of betas to bypass ``beta_start`` / ``beta_end``.
        clip_sample: Whether to clip predicted sample between -1 and 1.
        set_alpha_to_one: Fix the previous alpha product to 1 for the final step.
        steps_offset: Offset added to the inference steps.
        prediction_type: Prediction type of the scheduler function.
        base_resolution: Base training resolution of the diffusion model.
        target_resolution: Target resolution; triggers rescheduling when > base.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        base_resolution: int = 512,
        target_resolution: int = 512,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Apply noise rescheduling based on target resolution
        self.alphas_cumprod = compute_rescheduled_alphas_cumprod(
            self.alphas_cumprod, base_resolution, target_resolution
        )

        # Recompute alphas and betas from rescheduled alphas_cumprod
        alphas_cumprod_restore = torch.empty_like(self.alphas_cumprod)
        alphas_cumprod_restore[0] = self.alphas_cumprod[0]
        alphas_cumprod_restore[1:] = self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
        self.alphas = alphas_cumprod_restore
        self.betas = 1.0 - self.alphas

        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """Ensures interchangeability with schedulers that need to scale the denoising model input."""
        return sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> float:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Set the discrete timesteps used for the diffusion chain."""
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than "
                f"`self.config.num_train_timesteps`: {self.config.num_train_timesteps}."
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """Predict the sample at the previous timestep by reversing the SDE."""
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', run 'set_timesteps' after creating the scheduler."
            )

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of "
                "`epsilon`, `sample`, or `v_prediction`."
            )

        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        if use_clipped_model_output:
            model_output = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError("Cannot pass both generator and variance_noise.")
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                )
            prev_sample = prev_sample + std_dev_t * variance_noise

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """Add noise to the original samples according to the noise schedule."""
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
