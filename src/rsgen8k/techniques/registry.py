"""Technique registry for high-resolution generation methods.

All techniques share a common :class:`BaseTechnique` interface so the
generation engine can select and apply any method by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TechniqueInfo:
    """Metadata for a supported high-resolution generation technique.

    Attributes:
        key: Short identifier used in configs and CLI.
        name: Human-readable technique name.
        description: Brief description of the approach.
        paper: Paper citation string.
        github_url: Original source code repository.
        supported_architectures: Model architectures this technique works with.
        module: Python module path within ``rsgen8k.techniques``.
    """

    key: str
    name: str
    description: str
    paper: str
    github_url: str
    supported_architectures: List[str]
    module: str


# ---------------------------------------------------------------------------
# Technique registry
# ---------------------------------------------------------------------------

TECHNIQUE_REGISTRY: Dict[str, TechniqueInfo] = {
    "native": TechniqueInfo(
        key="native",
        name="Native DDIM",
        description=(
            "Standard single-stage generation with DDIM scheduler. No "
            "multi-resolution or upscaling techniques."
        ),
        paper='Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.',
        github_url="https://github.com/CompVis/stable-diffusion",
        supported_architectures=["sd1.5", "sdxl"],
        module="rsgen8k.techniques.native",
    ),
    "megafusion": TechniqueInfo(
        key="megafusion",
        name="MegaFusion",
        description=(
            "Multi-stage progressive generation that decomposes the denoising "
            "process into stages at increasing resolutions with noise "
            "rescheduling."
        ),
        paper=(
            'Wu et al., "MegaFusion: Extend Diffusion Models towards '
            'Higher-resolution Image Generation without Further Tuning", '
            "WACV 2025."
        ),
        github_url="https://github.com/haoningwu3639/MegaFusion",
        supported_architectures=["sd1.5", "sdxl", "sd3"],
        module="rsgen8k.techniques.megafusion",
    ),
    "elasticdiffusion": TechniqueInfo(
        key="elasticdiffusion",
        name="ElasticDiffusion",
        description=(
            "Training-free arbitrary-size image generation through "
            "global-local content separation, using view-based patch "
            "sampling with reduced-resolution guidance."
        ),
        paper=(
            'Haji Ali et al., "ElasticDiffusion: Training-free Arbitrary '
            'Size Image Generation through Global-Local Content Separation", '
            "CVPR 2024."
        ),
        github_url="https://github.com/MoayedHajiAli/ElasticDiffusion-official",
        supported_architectures=["sd1.5", "sdxl"],
        module="rsgen8k.techniques.elastic_diffusion",
    ),
    "multidiffusion": TechniqueInfo(
        key="multidiffusion",
        name="MultiDiffusion",
        description=(
            "Fuses multiple diffusion generation paths over overlapping "
            "patches using sliding-window consensus voting for spatial "
            "consistency."
        ),
        paper=(
            'Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for '
            'Controlled Image Generation", ICML 2023.'
        ),
        github_url="https://github.com/omerbt/MultiDiffusion",
        supported_architectures=["sd1.5", "sdxl"],
        module="rsgen8k.techniques.multi_diffusion",
    ),
    "freescale": TechniqueInfo(
        key="freescale",
        name="FreeScale",
        description=(
            "Tuning-free scale fusion via modified attention mechanisms with "
            "cosine scheduling to achieve multi-resolution generation up to "
            "8K."
        ),
        paper=(
            'Qiu et al., "FreeScale: Unleashing the Resolution of Diffusion '
            'Models via Tuning-Free Scale Fusion", arXiv:2412.09626, 2024.'
        ),
        github_url="https://github.com/ali-vilab/FreeScale",
        supported_architectures=["sdxl"],
        module="rsgen8k.techniques.freescale",
    ),
    "demofusion": TechniqueInfo(
        key="demofusion",
        name="DemoFusion",
        description=(
            "Progressive upscaling with skip-residual connections and dilated "
            "sampling to extend latent diffusion models to higher resolutions."
        ),
        paper=(
            'Du et al., "DemoFusion: Democratising High-Resolution Image '
            'Generation With No $$$", CVPR 2024.'
        ),
        github_url="https://github.com/PRIS-CV/DemoFusion",
        supported_architectures=["sdxl"],
        module="rsgen8k.techniques.demofusion",
    ),
    "fouriscale": TechniqueInfo(
        key="fouriscale",
        name="FouriScale",
        description=(
            "Frequency-domain perspective on high-resolution synthesis: "
            "replaces convolutions with dilated variants plus low-pass "
            "filtering for structural consistency."
        ),
        paper=(
            'Huang et al., "FouriScale: A Frequency Perspective on '
            'Training-Free High-Resolution Image Synthesis", ECCV 2024.'
        ),
        github_url="https://github.com/LeonHLJ/FouriScale",
        supported_architectures=["sd1.5", "sdxl"],
        module="rsgen8k.techniques.fouriscale",
    ),
    "inftydiff": TechniqueInfo(
        key="inftydiff",
        name="∞-Diff",
        description=(
            "Infinite resolution diffusion with mollified noise predictions "
            "via DCT-domain Gaussian blur and optional coordinate "
            "subsampling for efficient high-resolution generation."
        ),
        paper=(
            'Bond-Taylor and Willcocks, "∞-Diff: Infinite Resolution '
            'Diffusion with Subsampled Mollified States", ICLR 2024.'
        ),
        github_url="https://github.com/samb-t/infty-diff",
        supported_architectures=["sd1.5", "sdxl"],
        module="rsgen8k.techniques.infty_diff",
    ),
    "diffusion4k": TechniqueInfo(
        key="diffusion4k",
        name="Diffusion-4K",
        description=(
            "Wavelet-based fine-tuning for direct ultra-high-resolution "
            "image synthesis. A Haar DWT loss forces the model to learn "
            "both low-frequency structure and high-frequency detail, "
            "enabling direct 4K generation from fine-tuned weights."
        ),
        paper=(
            'Zhang et al., "Diffusion-4K: Ultra-High-Resolution Image '
            'Synthesis with Latent Diffusion Models", CVPR 2025.'
        ),
        github_url="https://github.com/zhang0jhon/diffusion-4k",
        supported_architectures=["sd1.5", "sdxl", "sd3"],
        module="rsgen8k.techniques.diffusion4k",
    ),
}


def get_technique(name: str) -> TechniqueInfo:
    """Look up a technique by its registry key (case-insensitive).

    Raises:
        KeyError: If the technique name is not found in the registry.
    """
    key = name.lower()
    if key not in TECHNIQUE_REGISTRY:
        available = ", ".join(sorted(TECHNIQUE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown technique '{name}'. Available techniques: {available}"
        )
    return TECHNIQUE_REGISTRY[key]


def list_techniques() -> Dict[str, TechniqueInfo]:
    """Return a copy of the full technique registry."""
    return dict(TECHNIQUE_REGISTRY)
