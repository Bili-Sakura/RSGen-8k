"""Supported base model definitions.

This module provides a registry of supported remote sensing diffusion models
with their HuggingFace model IDs and metadata.

Models can be loaded from a local checkpoint directory (default ``./ckpt``)
using the same ``{org}/{repo}`` structure as HuggingFace, or downloaded
automatically from the Hub when no local copy exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

# Default local checkpoint root — mirrors HuggingFace ``{org}/{repo}`` layout.
DEFAULT_CKPT_DIR = os.path.join(".", "ckpt")


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


def resolve_model_path(
    model_id: str,
    ckpt_dir: Optional[str] = None,
) -> str:
    """Resolve a model ID to a local path or HuggingFace repo ID.

    Checks ``{ckpt_dir}/{model_id}`` first. If the directory exists and
    contains a ``model_index.json`` or ``unet/`` subfolder it is returned
    as-is (local path). Otherwise the original *model_id* is returned so
    that diffusers will download from the HuggingFace Hub.

    Args:
        model_id: HuggingFace-style ``org/repo`` identifier.
        ckpt_dir: Root directory for local checkpoints.  Defaults to
            :data:`DEFAULT_CKPT_DIR` (``./ckpt``).

    Returns:
        Absolute local path when available, otherwise the original
        *model_id* string.
    """
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_CKPT_DIR

    local_path = os.path.join(ckpt_dir, model_id)
    if os.path.isdir(local_path):
        # Minimal sanity check: a diffusers model dir typically has model_index.json or unet/
        if (
            os.path.isfile(os.path.join(local_path, "model_index.json"))
            or os.path.isdir(os.path.join(local_path, "unet"))
        ):
            return os.path.abspath(local_path)
    return model_id
