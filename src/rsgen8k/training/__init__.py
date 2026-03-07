"""Training utilities for fine-tuning diffusion models with wavelet-based loss.

Implements the Diffusion-4K wavelet fine-tuning technique from:
  Zhang et al., "Diffusion-4K: Ultra-High-Resolution Image Synthesis with
  Latent Diffusion Models", CVPR 2025.
"""

from rsgen8k.training.wavelet_loss import WaveletLoss, haar_dwt_2d

__all__ = ["WaveletLoss", "haar_dwt_2d"]
