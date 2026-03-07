"""∞-Diff: Infinite Resolution Diffusion with Subsampled Mollified States.

Adapted from the ∞-Diff project by Sam Bond-Taylor and Chris G. Willcocks.
  Paper : "∞-Diff: Infinite Resolution Diffusion with Subsampled Mollified
           States", ICLR 2024.
  Source: https://github.com/samb-t/infty-diff
  License: MIT

∞-Diff introduces a diffusion process defined in infinite-dimensional
Hilbert space.  Two key ideas are adapted here for standard latent
diffusion pipelines:

1. **Mollified diffusion** – a DCT-domain Gaussian blur is applied to
   noise predictions so that the denoising process smooths out artifacts
   introduced by operating at resolutions beyond the training
   distribution.
2. **Coordinate subsampling** – only a random subset of spatial
   positions is denoised per step, reducing memory and compute while
   retaining quality through Monte-Carlo integration.

Together these allow existing Stable-Diffusion UNets to produce higher
resolution images with reduced memory and improved structural coherence.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCT-based Gaussian Blur (Mollifier)
# ---------------------------------------------------------------------------

def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an orthonormal 1-D DCT-II matrix of size ``n``.

    Rows correspond to frequency indices *k* and columns to spatial
    positions *i*, i.e. ``C[k, i] = alpha_k * cos(pi * k * (2i+1) / (2n))``.
    """
    grid = torch.arange(n, device=device, dtype=dtype)
    # C[k, i] = cos(pi * k * (2i + 1) / (2n))
    basis = torch.cos(math.pi * grid.unsqueeze(1) * (2.0 * grid.unsqueeze(0) + 1.0) / (2.0 * n))
    # Orthonormal scaling
    basis[0] *= math.sqrt(1.0 / n)
    basis[1:] *= math.sqrt(2.0 / n)
    return basis


def dct_gaussian_blur(
    x: torch.FloatTensor,
    std: float = 1.0,
) -> torch.FloatTensor:
    """Apply Gaussian blur in DCT domain (separable, 2-D).

    The blur is implemented by multiplying DCT coefficients with a
    Gaussian envelope ``exp(-0.5 * (pi * k * std / N)^2)`` along each
    spatial axis.  This avoids large spatial-domain kernels and is exact
    for the chosen standard deviation.

    Args:
        x: Input tensor ``(B, C, H, W)``.
        std: Standard deviation of the Gaussian kernel in pixel units.

    Returns:
        Blurred tensor with the same shape and dtype.
    """
    if std <= 0:
        return x

    _, _, h, w = x.shape
    orig_dtype = x.dtype
    x_f = x.float()

    # Construct 1-D DCT matrices for height and width
    dct_h = _dct_matrix(h, x.device, torch.float32)  # (H, H)
    dct_w = _dct_matrix(w, x.device, torch.float32)  # (W, W)

    # Gaussian filter in DCT domain
    freq_h = torch.arange(h, device=x.device, dtype=torch.float32)
    freq_w = torch.arange(w, device=x.device, dtype=torch.float32)
    env_h = torch.exp(-0.5 * (math.pi * freq_h * std / h) ** 2)
    env_w = torch.exp(-0.5 * (math.pi * freq_w * std / w) ** 2)

    # 2-D separable DCT: Y = C_h @ X @ C_w^T
    # Then filter:        Y' = Y * env_h[:, None] * env_w[None, :]
    # Inverse:            X' = C_h^T @ Y' @ C_w
    x_dct = dct_h @ x_f @ dct_w.T
    x_dct = x_dct * (env_h.view(1, 1, h, 1) * env_w.view(1, 1, 1, w))
    x_out = dct_h.T @ x_dct @ dct_w

    return x_out.to(dtype=orig_dtype)


def wiener_deconvolution(
    x: torch.FloatTensor,
    std: float = 1.0,
    snr: float = 100.0,
) -> torch.FloatTensor:
    """Approximate Wiener deconvolution to undo :func:`dct_gaussian_blur`.

    This sharpens the final image by inverting the Gaussian envelope in
    DCT domain, regularised by a signal-to-noise ratio to avoid
    amplifying noise.

    Args:
        x: Blurred tensor ``(B, C, H, W)``.
        std: Standard deviation used during blurring.
        snr: Assumed signal-to-noise ratio for regularisation.

    Returns:
        Sharpened tensor with the same shape and dtype.
    """
    if std <= 0:
        return x

    _, _, h, w = x.shape
    orig_dtype = x.dtype
    x_f = x.float()

    dct_h = _dct_matrix(h, x.device, torch.float32)
    dct_w = _dct_matrix(w, x.device, torch.float32)

    freq_h = torch.arange(h, device=x.device, dtype=torch.float32)
    freq_w = torch.arange(w, device=x.device, dtype=torch.float32)
    env_h = torch.exp(-0.5 * (math.pi * freq_h * std / h) ** 2)
    env_w = torch.exp(-0.5 * (math.pi * freq_w * std / w) ** 2)

    # Wiener filter: H* / (|H|^2 + 1/SNR)
    wiener_h = env_h / (env_h ** 2 + 1.0 / snr)
    wiener_w = env_w / (env_w ** 2 + 1.0 / snr)

    # 2-D separable DCT, apply Wiener filter, inverse DCT
    x_dct = dct_h @ x_f @ dct_w.T
    x_dct = x_dct * (wiener_h.view(1, 1, h, 1) * wiener_w.view(1, 1, 1, w))
    x_out = dct_h.T @ x_dct @ dct_w

    return x_out.to(dtype=orig_dtype)


# ---------------------------------------------------------------------------
# Coordinate subsampling helpers
# ---------------------------------------------------------------------------

def subsample_coordinates(
    h: int,
    w: int,
    ratio: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample a random subset of spatial coordinate indices.

    Returns a 1-D tensor of flat indices into an ``(H, W)`` grid.
    Following ∞-Diff, the default ratio is 0.25 (= 4× subsampling).

    Args:
        h: Spatial height.
        w: Spatial width.
        ratio: Fraction of coordinates to keep (0, 1].
        device: Target device.

    Returns:
        1-D ``LongTensor`` of length ``int(h * w * ratio)``.
    """
    total = h * w
    n_samples = max(1, int(total * ratio))
    n_samples = min(n_samples, total)
    indices = torch.randperm(total, device=device)[:n_samples]
    return indices


def apply_sparse_mask(
    x: torch.FloatTensor,
    indices: torch.Tensor,
) -> torch.FloatTensor:
    """Zero-out unsampled spatial positions.

    Args:
        x: Tensor ``(B, C, H, W)``.
        indices: 1-D flat indices to **keep**.

    Returns:
        Masked tensor with unsampled positions set to zero and sampled
        positions scaled by ``total / n_samples`` for unbiased estimation.
    """
    b, c, h, w = x.shape
    total = h * w
    n_samples = indices.numel()
    mask = torch.zeros(total, device=x.device, dtype=x.dtype)
    mask[indices] = float(total) / float(n_samples)
    mask = mask.view(1, 1, h, w).expand_as(x)
    return x * mask


# ---------------------------------------------------------------------------
# Per-step denoising function
# ---------------------------------------------------------------------------

def inftydiff_denoise_step(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    t: torch.Tensor,
    guidance_scale: float,
    gaussian_std: float = 1.0,
    subsample_ratio: float = 1.0,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    """Perform one ∞-Diff inspired denoising step.

    Combines mollified noise prediction (DCT Gaussian blur) with
    optional coordinate subsampling.  When ``subsample_ratio < 1`` only
    a random subset of latent positions contributes to the noise
    prediction, reducing memory and compute.

    Args:
        unet: The UNet denoiser module.
        scheduler: Diffusion scheduler.
        latents: Current noisy latents ``(B, C, H, W)``.
        text_embeddings: Encoded text (with unconditional part if CFG).
        t: Current timestep.
        guidance_scale: Classifier-free guidance scale.
        gaussian_std: Standard deviation for DCT Gaussian mollification.
            Set to ``0`` to disable mollification.
        subsample_ratio: Fraction of spatial coordinates to use per step
            (1.0 = full resolution, 0.25 = 4× subsampling as in paper).
        class_labels: Optional class conditioning tensor (e.g. for
            Text2Earth resolution conditioning).

    Returns:
        Updated latents after one denoising step.
    """
    do_cfg = guidance_scale > 1.0

    latent_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_input = scheduler.scale_model_input(latent_input, t)

    unet_kwargs = {"encoder_hidden_states": text_embeddings}
    if class_labels is not None:
        unet_kwargs["class_labels"] = class_labels

    noise_pred = unet(latent_input, t, **unet_kwargs).sample.to(dtype=latents.dtype)

    if do_cfg:
        u, c = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)

    # ---- Mollification: smooth the noise prediction in DCT domain ----
    if gaussian_std > 0:
        noise_pred = dct_gaussian_blur(noise_pred, std=gaussian_std)

    # ---- Coordinate subsampling (Monte-Carlo integration) ----
    # Only subsample when ratio is strictly between 0 and 1; ratios >= 1
    # mean "use all coordinates" and are skipped for efficiency.
    if 0 < subsample_ratio < 1.0:
        _, _, lh, lw = noise_pred.shape
        indices = subsample_coordinates(lh, lw, ratio=subsample_ratio, device=noise_pred.device)
        noise_pred = apply_sparse_mask(noise_pred, indices)

    step_out = scheduler.step(noise_pred, t, latents)
    return step_out.prev_sample
