# RSGen-8k

**Remote Sensing Image Generation Scaling to 8K Resolution**

RSGen-8k applies tuning-free progressive upscaling techniques to remote sensing diffusion models, enabling text-to-image generation of satellite imagery at up to 8192 × 8192 (8K) resolution. Text prompts are sourced from the [XLRS-Bench](https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en) ultra-high-resolution remote sensing caption dataset.

## Supported Base Models

| Model | HuggingFace ID | Architecture |
|-------|---------------|--------------|
| [Text2Earth](https://huggingface.co/lcybuaa/Text2Earth) | `lcybuaa/Text2Earth` | SD 1.5 |
| [DiffusionSat-Single-512](https://huggingface.co/BiliSakura/DiffusionSat-Single-512) | `BiliSakura/DiffusionSat-Single-512` | SD 1.5 |
| [GeoSynth](https://huggingface.co/MVRL/GeoSynth) | `MVRL/GeoSynth` | SD 1.5 |
| [DDPM-CD Pretrained 256](https://huggingface.co/BiliSakura/ddpm-cd-pretrained-256) | `BiliSakura/ddpm-cd-pretrained-256` | DDPM-SR3 (unconditional) |

## Supported Upscaling Techniques

| Technique | Paper | Source |
|-----------|-------|--------|
| [MegaFusion](https://github.com/haoningwu3639/MegaFusion) | Wu et al., WACV 2025 | Multi-stage progressive denoising with noise rescheduling |
| [ElasticDiffusion](https://github.com/MoayedHajiAli/ElasticDiffusion-official) | Haji Ali et al., CVPR 2024 | Global-local content separation for arbitrary-size generation |
| [MultiDiffusion](https://github.com/omerbt/MultiDiffusion) | Bar-Tal et al., ICML 2023 | Sliding-window view fusion with consensus voting |
| [FreeScale](https://github.com/ali-vilab/FreeScale) | Qiu et al., arXiv 2024 | Cosine-scheduled scale fusion via attention modification |
| [DemoFusion](https://github.com/PRIS-CV/DemoFusion) | Du et al., CVPR 2024 | Progressive upscaling with dilated sampling and skip-residuals |
| [FouriScale](https://github.com/LeonHLJ/FouriScale) | Huang et al., ECCV 2024 | Frequency-domain low-pass filtering for structural consistency |

## Project Structure

```
RSGen-8k/
├── app.py                # Gradio web demo
├── configs/              # YAML configuration files
│   └── default.yaml      # Default generation config (512 → 8K)
├── ckpt/                 # Local model checkpoints (HuggingFace repo_id layout)
│   └── lcybuaa/Text2Earth/  # Example: place model files here
├── src/rsgen8k/          # Main Python package
│   ├── models/           # Model components & registry
│   │   ├── model_registry.py  # Supported base model definitions + local path resolution
│   │   ├── scheduler.py  # DDIM scheduler with noise rescheduling
│   │   └── pipeline.py   # SD pipeline with stage-based timesteps
│   ├── techniques/       # Upscaling technique implementations
│   │   ├── registry.py   # Technique registry & metadata
│   │   ├── megafusion.py # MegaFusion (Wu et al., WACV 2025)
│   │   ├── elastic_diffusion.py  # ElasticDiffusion (Haji Ali et al., CVPR 2024)
│   │   ├── multi_diffusion.py    # MultiDiffusion (Bar-Tal et al., ICML 2023)
│   │   ├── freescale.py  # FreeScale (Qiu et al., 2024)
│   │   ├── demofusion.py # DemoFusion (Du et al., CVPR 2024)
│   │   └── fouriscale.py # FouriScale (Huang et al., ECCV 2024)
│   ├── data/             # Dataset loading utilities
│   │   └── xlrs_bench.py # XLRS-Bench prompt loader
│   ├── metrics.py       # Image quality (FID, KID, CMMD, DINO sim, CLIP-Score, LPIPS, PSNR, SSIM)
│   └── generate.py      # Multi-stage generation engine & CLI
├── scripts/              # Standalone scripts
│   ├── generate.py           # Generation script with dataset support
│   ├── download_prompts.py   # Preview/download XLRS-Bench prompts
│   ├── benchmark_single.sh   # Single model+technique benchmark
│   ├── benchmark_case_study.sh  # Multi-resolution case study with metrics
│   ├── benchmark_all.sh      # Grid benchmark (all models × techniques)
│   ├── benchmark_seeds.sh    # Multi-seed reproducibility benchmark
│   ├── generate_dataset.sh   # Batch generation from XLRS-Bench
│   └── evaluate.sh           # Image quality evaluation (FID, CLIP-Score, etc.)
├── tests/                # Unit tests (86 tests)
├── manuscript/           # Paper / manuscript materials
└── outputs/              # Generated images (gitignored contents)
```

## Installation

### Conda Setup (Recommended)

Recommended environment with PyTorch 2.8.0 (CUDA 12.6):

```bash
conda create -n rsgen python=3.12
conda activate rsgen
# PyTorch 2.8.0 + CUDA 12.6 from https://download.pytorch.org/whl/cu126
# Other versions generally work — follow https://pytorch.org/get-started/previous-versions/
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# Install other packages
pip install -r requirements.txt
pip install swanlab
```

Or use the setup script:

```bash
bash scripts/setup_env.sh
```

Optional (extra optimizers):

```bash
# pip install muon-optimizer  # Uncomment if needed
```

### Post-install: Memory-efficient Attention (xformers / flash-attn)

For faster inference and lower VRAM usage at high resolutions, install one of:

**xformers** (recommended, easiest):

```bash
pip install xformers
# Or with editable install:
pip install -e ".[dev,xformers]"
```

xformers works with most PyTorch/CUDA combos. If you hit compatibility issues, check [xformers releases](https://github.com/facebookresearch/xformers/releases) for your PyTorch version.

**flash-attn** (often faster, stricter requirements):

```bash
# Option A: Prebuilt wheels (recommended) — visit https://flashattn.dev/
# Select your PyTorch, CUDA, and Python versions, then run the generated pip command.

# Option B: From source (requires CUDA toolkit + ninja)
pip install flash-attn --no-build-isolation
# On low RAM: MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

flash-attn supports PyTorch 2.3–2.9 and CUDA 11.8, 12.1–12.8. Prebuilt wheels skip ~30 min compile time.

RSGen-8k auto-enables xformers when available. Disable with `--no_xformers` if needed.

### Editable Install (After Conda Setup)

```bash
pip install -e ".[dev]"
```

For memory-efficient attention (recommended for high-resolution generation):

```bash
pip install -e ".[dev,xformers]"   # xformers (easiest)
# pip install -e ".[dev,flash-attn]"  # flash-attn — see Post-install section
```

For the Gradio web demo:

```bash
pip install -e ".[dev,demo]"
```

## Quick Start

### Generate with a Custom Prompt

```bash
python scripts/generate.py \
    --prompt "A high-resolution satellite image of a coastal city with harbors and ships" \
    --output_dir ./outputs \
    --seed 42
```

### Use a Different Base Model

```bash
# DiffusionSat
python scripts/generate.py \
    --model_name diffusionsat \
    --prompt "Aerial view of agricultural fields"

# GeoSynth
python scripts/generate.py \
    --model_name geosynth \
    --prompt "Satellite image of a mountain range"
```

### Use a Different Upscaling Technique

```bash
# MultiDiffusion (sliding-window consensus)
python scripts/generate.py \
    --technique multidiffusion \
    --prompt "Dense urban area"

# FouriScale (frequency-domain filtering)
python scripts/generate.py \
    --technique fouriscale \
    --prompt "Industrial zone with factories"
```

### List Available Models and Techniques

```bash
python scripts/generate.py --list_models
python scripts/generate.py --list_techniques
```

### Generate with XLRS-Bench Prompts

```bash
python scripts/generate.py \
    --from_dataset \
    --num_prompts 5 \
    --output_dir ./outputs
```

### Generate at Custom Resolution Stages

```bash
python scripts/generate.py \
    --prompt "Dense urban area with skyscrapers and parks" \
    --stage_resolutions 512 1024 2048 \
    --stage_steps 40 5 5
```

### Using YAML Configuration

```bash
python scripts/generate.py --config configs/default.yaml
```

### Loading Models from Local Checkpoints

Models are automatically loaded from the local `./models` directory if available,
using the same `{org}/{repo}` layout as HuggingFace. If no local copy is found,
models are downloaded from the Hub.

```bash
# Directory layout (mirrors HuggingFace repo_id):
models/
├── lcybuaa/Text2Earth/
│   ├── model_index.json
│   ├── unet/
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
├── BiliSakura/DiffusionSat-Single-512/
└── MVRL/GeoSynth/

# Use a custom checkpoint directory
python scripts/generate.py --ckpt_dir /data/models --prompt "Aerial view"
```

### Deploy to Hugging Face Spaces

A self-contained demo is in `hf_space/` — upload that folder to create a Space. Models load from Hugging Face Hub.

```bash
cd hf_space
# Create a new Space at huggingface.co/new-space, then:
# git clone your-space-url .
# Copy hf_space/* into the Space repo and push
```

### Gradio Web Demo

Launch an interactive web interface for generation:

```bash
python app.py                         # Default: http://localhost:7860
python app.py --port 8080 --share     # Public link via Gradio
python app.py --ckpt_dir /data/models # Custom checkpoint directory
```

### Benchmark Scripts

Production-ready shell scripts with all parameters defaulted and overridable:

```bash
# Single model+technique benchmark
bash scripts/benchmark_single.sh
bash scripts/benchmark_single.sh --model_name diffusionsat --technique multidiffusion

# Full grid benchmark (3 models × 6 techniques = 18 combinations)
bash scripts/benchmark_all.sh

# Multi-seed reproducibility (default: 5 seeds)
bash scripts/benchmark_seeds.sh --model_name text2earth --technique megafusion

# Batch generation from XLRS-Bench dataset
bash scripts/generate_dataset.sh --num_prompts 50

# Image quality evaluation
bash scripts/evaluate.sh --generated_dir ./outputs/benchmark/text2earth_megafusion_seed42

# Resolution case study (1K → 2K → 4K → 8K) with metrics
bash scripts/benchmark_case_study.sh
bash scripts/benchmark_case_study.sh --quick   # 1024 and 2048 only
bash scripts/benchmark_case_study.sh --reference_dir ./data/reference  # FID, LPIPS, PSNR, SSIM

# Override via environment variables
MODEL_NAME=geosynth TECHNIQUE=fouriscale bash scripts/benchmark_single.sh
```

## How It Works

All techniques share a common multi-stage framework: generate at base resolution, then progressively upscale through intermediate resolutions to 8K. The per-step denoising behavior varies by technique:

### MegaFusion (default)
Multi-stage progressive generation with noise rescheduling. At each stage: bicubic-upsample → VAE encode → add noise → denoise for a few steps.

### MultiDiffusion
At each denoising step, splits the latent canvas into overlapping sliding-window views, denoises each independently, and fuses via weighted-average consensus.

### ElasticDiffusion
Combines a reduced-resolution global guidance pass with patch-wise local denoising, blending global structure with local detail.

### FreeScale
Runs dual UNet passes (high-res and low-res) at each step, blending predictions via a cosine-annealed schedule.

### DemoFusion
Applies dilated sampling with skip-residual connections from previous resolution phases, smoothed by Gaussian filtering.

### FouriScale
Applies low-pass Fourier filtering to noise predictions, suppressing high-frequency artifacts at out-of-distribution resolutions.

### Noise Rescheduling

MegaFusion optionally applies resolution-dependent noise rescheduling that adjusts the noise schedule based on the upscale factor, preventing artifacts at high resolutions. Enable with `--if_reschedule`.

## Reproducibility

Diffusion is a stochastic process — different runs produce different outputs. RSGen-8k provides built-in controls for deterministic, reproducible generation:

### Seed Control

All random number generators (`random`, `numpy`, `torch`) are seeded when a `--seed` is provided. A CPU `torch.Generator` is used for noise generation to ensure cross-platform consistency (GPU generators produce hardware-dependent results).

```bash
# Same seed → same output (within platform tolerance)
python scripts/generate.py --prompt "Coastal city" --seed 42
python scripts/generate.py --prompt "Coastal city" --seed 42  # identical output
```

> **Tip:** The CPU generator is used automatically — you don't need to do anything special. Just provide `--seed` for reproducible results.

### Full Determinism

For bit-exact reproducibility (e.g. testing, benchmarking), enable `--deterministic`:

```bash
python scripts/generate.py --prompt "Coastal city" --seed 42 --deterministic
```

This enables:
- `torch.use_deterministic_algorithms(True)` — forces deterministic CUDA kernels
- `torch.backends.cudnn.deterministic = True` — disables non-deterministic cuDNN ops
- `torch.backends.cudnn.benchmark = False` — disables auto-tuning
- `CUBLAS_WORKSPACE_CONFIG=:16:8` — single buffer size for cuBLAS

> **Note:** Deterministic mode may reduce performance. See PyTorch's [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) guide for details.

## Running Tests

```bash
pytest tests/ -v
```

## Key Dependencies

| Package      | Purpose                              |
|--------------|--------------------------------------|
| diffusers    | Diffusion model pipeline & components |
| transformers | Text encoder & tokenizer             |
| accelerate   | Mixed-precision inference            |
| datasets     | XLRS-Bench prompt loading            |
| torch        | Deep learning framework              |
| einops       | Tensor rearrangement                 |

## References

### Base Models
- **Text2Earth:** [huggingface.co/lcybuaa/Text2Earth](https://huggingface.co/lcybuaa/Text2Earth)
- **DiffusionSat:** [huggingface.co/BiliSakura/DiffusionSat-Single-512](https://huggingface.co/BiliSakura/DiffusionSat-Single-512)
- **GeoSynth:** [huggingface.co/MVRL/GeoSynth](https://huggingface.co/MVRL/GeoSynth)
- **DDPM-CD Pretrained 256:** [huggingface.co/BiliSakura/ddpm-cd-pretrained-256](https://huggingface.co/BiliSakura/ddpm-cd-pretrained-256)

### Techniques
- **MegaFusion:** Wu et al., "MegaFusion: Extend Diffusion Models towards Higher-resolution Image Generation without Further Tuning", WACV 2025. [GitHub](https://github.com/haoningwu3639/MegaFusion)
- **ElasticDiffusion:** Haji Ali et al., "ElasticDiffusion: Training-free Arbitrary Size Image Generation through Global-Local Content Separation", CVPR 2024. [GitHub](https://github.com/MoayedHajiAli/ElasticDiffusion-official)
- **MultiDiffusion:** Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", ICML 2023. [GitHub](https://github.com/omerbt/MultiDiffusion)
- **FreeScale:** Qiu et al., "FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion", arXiv 2024. [GitHub](https://github.com/ali-vilab/FreeScale)
- **DemoFusion:** Du et al., "DemoFusion: Democratising High-Resolution Image Generation With No $$$", CVPR 2024. [GitHub](https://github.com/PRIS-CV/DemoFusion)
- **FouriScale:** Huang et al., "FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis", ECCV 2024. [GitHub](https://github.com/LeonHLJ/FouriScale)

### Dataset
- **XLRS-Bench:** [huggingface.co/datasets/initiacms/XLRS-Bench_caption_en](https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en)

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.