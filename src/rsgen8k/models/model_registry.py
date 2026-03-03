"""Supported base model definitions.

This module provides a registry of supported remote sensing diffusion models
with their HuggingFace model IDs and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class BaseModelInfo:
    """Metadata for a supported base diffusion model.

    Attributes:
        model_id: HuggingFace model repository ID.
        name: Short human-readable name.
        architecture: Model architecture type (e.g. ``"sd1.5"``, ``"sdxl"``).
        base_resolution: Native training resolution.
        description: Brief description of the model.
        url: HuggingFace model page URL.
    """

    model_id: str
    name: str
    architecture: str
    base_resolution: int
    description: str
    url: str


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, BaseModelInfo] = {
    "text2earth": BaseModelInfo(
        model_id="lcybuaa/Text2Earth",
        name="Text2Earth",
        architecture="sd1.5",
        base_resolution=512,
        description=(
            "Text-to-remote-sensing-image generation model based on Stable "
            "Diffusion 1.5, trained to generate high-quality remote sensing "
            "images from text descriptions."
        ),
        url="https://huggingface.co/lcybuaa/Text2Earth",
    ),
    "diffusionsat": BaseModelInfo(
        model_id="BiliSakura/DiffusionSat-Single-512",
        name="DiffusionSat-Single-512",
        architecture="sd1.5",
        base_resolution=512,
        description=(
            "DiffusionSat single-image generation model at 512×512 resolution, "
            "designed for satellite image synthesis conditioned on text and "
            "metadata."
        ),
        url="https://huggingface.co/BiliSakura/DiffusionSat-Single-512",
    ),
    "geosynth": BaseModelInfo(
        model_id="MVRL/GeoSynth",
        name="GeoSynth",
        architecture="sd1.5",
        base_resolution=512,
        description=(
            "GeoSynth generates realistic satellite imagery from text "
            "descriptions, fine-tuned on geospatial datasets."
        ),
        url="https://huggingface.co/MVRL/GeoSynth",
    ),
}


def get_model_info(name: str) -> BaseModelInfo:
    """Look up a model by its registry key (case-insensitive).

    Raises:
        KeyError: If the model name is not found in the registry.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return MODEL_REGISTRY[key]


def list_models() -> Dict[str, BaseModelInfo]:
    """Return a copy of the full model registry."""
    return dict(MODEL_REGISTRY)
