"""Wavelet-based training loss for ultra-high-resolution diffusion models.

Adapted from the Diffusion-4K project by Jinjin Zhang et al.
  Paper : "Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent
           Diffusion Models", CVPR 2025.
  Source: https://github.com/zhang0jhon/diffusion-4k
  License: Apache-2.0

The key idea is to decompose both the model prediction and the ground-truth
latent into wavelet sub-bands (LL, LH, HL, HH) using a single-level Haar
discrete wavelet transform (DWT) and compute the training loss in the
wavelet domain.  This forces the model to attend to both low-frequency
structure (LL) and high-frequency detail (LH, HL, HH), producing sharper
and more detailed images at ultra-high resolutions.

Two implementations are provided:

1. A **pure-PyTorch** Haar DWT (``haar_dwt_2d``) that requires no extra
   dependencies.
2. A thin ``WaveletLoss`` wrapper that optionally delegates to the
   ``pytorch_wavelets`` library when installed, falling back to the
   pure-PyTorch path otherwise.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-PyTorch Haar DWT (no external dependency)
# ---------------------------------------------------------------------------

def haar_dwt_2d(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2-D Haar discrete wavelet transform.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``.  ``H`` and ``W`` must
           be even.

    Returns:
        Tuple of four tensors ``(LL, LH, HL, HH)`` each with shape
        ``(B, C, H/2, W/2)``.

        - **LL**: approximation (low-pass rows, low-pass cols)
        - **LH**: horizontal detail (low-pass rows, high-pass cols)
        - **HL**: vertical detail (high-pass rows, low-pass cols)
        - **HH**: diagonal detail (high-pass rows, high-pass cols)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4-D input (B, C, H, W), got {x.ndim}-D")
    if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
        raise ValueError(
            f"Spatial dimensions must be even, got H={x.shape[2]}, W={x.shape[3]}"
        )

    # Split into even/odd rows and columns
    x_ll = x[:, :, 0::2, 0::2]  # even rows, even cols
    x_lh = x[:, :, 0::2, 1::2]  # even rows, odd cols
    x_hl = x[:, :, 1::2, 0::2]  # odd rows, even cols
    x_hh = x[:, :, 1::2, 1::2]  # odd rows, odd cols

    # Haar wavelet: average / difference with 1/2 normalization
    ll = (x_ll + x_lh + x_hl + x_hh) * 0.5
    lh = (x_ll - x_lh + x_hl - x_hh) * 0.5
    hl = (x_ll + x_lh - x_hl - x_hh) * 0.5
    hh = (x_ll - x_lh - x_hl + x_hh) * 0.5

    return ll, lh, hl, hh


def haar_idwt_2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """Inverse single-level 2-D Haar DWT.

    Args:
        ll, lh, hl, hh: Sub-band tensors each of shape ``(B, C, H/2, W/2)``.

    Returns:
        Reconstructed tensor of shape ``(B, C, H, W)``.
    """
    # Inverse Haar: recover the four polyphase components
    x_ll = (ll + lh + hl + hh) * 0.5
    x_lh = (ll - lh + hl - hh) * 0.5
    x_hl = (ll + lh - hl - hh) * 0.5
    x_hh = (ll - lh - hl + hh) * 0.5

    B, C, Hh, Wh = ll.shape
    out = ll.new_empty(B, C, Hh * 2, Wh * 2)
    out[:, :, 0::2, 0::2] = x_ll
    out[:, :, 0::2, 1::2] = x_lh
    out[:, :, 1::2, 0::2] = x_hl
    out[:, :, 1::2, 1::2] = x_hh
    return out


# ---------------------------------------------------------------------------
# Wavelet Loss
# ---------------------------------------------------------------------------

class WaveletLoss(torch.nn.Module):
    """Wavelet-domain MSE loss for diffusion model training.

    Decomposes both ``pred`` and ``target`` into Haar wavelet sub-bands and
    computes the MSE in the concatenated wavelet coefficient space.  This
    encourages the model to reproduce both low-frequency structure and
    high-frequency detail.

    When ``pytorch_wavelets`` is installed, the DWT from that library is
    used (it supports additional wavelet families via the *wave* parameter).
    Otherwise a pure-PyTorch Haar implementation is used.

    Args:
        wave: Wavelet name passed to ``pytorch_wavelets.DWTForward``
            (default ``"haar"``).  Ignored when falling back to pure-PyTorch.
        levels: Number of DWT decomposition levels (default 1).
            Only level 1 is supported by the built-in implementation.
        weighting: Optional per-sample weighting tensor of shape ``(B, 1, 1, 1)``
            or broadcastable.
    """

    def __init__(self, wave: str = "haar", levels: int = 1) -> None:
        super().__init__()
        self.wave = wave
        self.levels = levels
        self._dwt = None  # Lazy-initialised on first call

        if levels != 1:
            raise ValueError("Only single-level DWT is supported (levels=1)")

    # -- helpers -------------------------------------------------------------

    def _ensure_dwt(self, device: torch.device, dtype: torch.dtype) -> None:
        """Lazy-initialise the DWT operator on the correct device."""
        if self._dwt is not None:
            return

        try:
            from pytorch_wavelets import DWTForward

            self._dwt = DWTForward(
                J=self.levels, mode="zero", wave=self.wave,
            ).to(device=device, dtype=dtype)
            self._use_pytorch_wavelets = True
            logger.debug("Using pytorch_wavelets DWT (%s)", self.wave)
        except ImportError:
            self._dwt = "builtin"
            self._use_pytorch_wavelets = False
            logger.debug("pytorch_wavelets not found; using built-in Haar DWT")

    def _decompose(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose *x* into wavelet sub-bands and concatenate along C."""
        if self._use_pytorch_wavelets:
            xll, xh = self._dwt(x)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
        else:
            xll, xlh, xhl, xhh = haar_dwt_2d(x)
        return torch.cat([xll, xlh, xhl, xhh], dim=1)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute wavelet-domain MSE loss.

        Args:
            pred: Model prediction, shape ``(B, C, H, W)``.
            target: Ground-truth latent, shape ``(B, C, H, W)``.
            weighting: Optional per-sample weight, broadcastable to ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        self._ensure_dwt(pred.device, pred.dtype)

        pred_w = self._decompose(pred)
        target_w = self._decompose(target)

        if weighting is not None:
            loss = torch.mean(
                (weighting.float() * (pred_w.float() - target_w.float()) ** 2).reshape(
                    target_w.shape[0], -1
                ),
                dim=1,
            )
        else:
            loss = torch.mean(
                (pred_w.float() - target_w.float()) ** 2,
                dim=(1, 2, 3),
            )
        return loss.mean()
