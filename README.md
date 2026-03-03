# RSGen-8k

**Remote Sensing Image Generation Scaling to 8K Resolution**

RSGen-8k applies [MegaFusion](https://github.com/haoningwu3639/MegaFusion) — a tuning-free progressive upscaling technique — to the [Text2Earth](https://huggingface.co/lcybuaa/Text2Earth) diffusion model, enabling text-to-image generation of remote sensing imagery at up to 8192 × 8192 (8K) resolution. Text prompts are sourced from the [XLRS-Bench](https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en) ultra-high-resolution remote sensing caption dataset.

## Project Structure

```
RSGen-8k/
├── configs/              # YAML configuration files
│   └── default.yaml      # Default generation config (512 → 8K)
├── src/rsgen8k/          # Main Python package
│   ├── models/           # MegaFusion model components
│   │   ├── scheduler.py  # DDIM scheduler with noise rescheduling
│   │   └── pipeline.py   # SD pipeline with stage-based timesteps
│   ├── data/             # Dataset loading utilities
│   │   └── xlrs_bench.py # XLRS-Bench prompt loader
│   └── generate.py       # Multi-stage generation engine & CLI
├── scripts/              # Standalone scripts
│   ├── generate.py       # Generation script with dataset support
│   └── download_prompts.py # Preview/download XLRS-Bench prompts
├── tests/                # Unit tests
├── links/                # References and links
├── manuscript/           # Paper / manuscript materials
└── outputs/              # Generated images (gitignored contents)
```

## Installation

```bash
pip install -e ".[dev]"
```

For memory-efficient attention (recommended for high-resolution generation):

```bash
pip install -e ".[dev,xformers]"
```

## Quick Start

### Generate with a Custom Prompt

```bash
python scripts/generate.py \
    --prompt "A high-resolution satellite image of a coastal city with harbors and ships" \
    --output_dir ./outputs \
    --seed 42
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

## How It Works

RSGen-8k uses MegaFusion's multi-stage progressive generation:

1. **Stage 1 (Base Resolution):** Generate at the model's native 512 × 512 resolution using the majority of denoising steps.
2. **Stage 2+ (Progressive Upscaling):** For each subsequent target resolution:
   - Bicubic-upsample the previous stage's output.
   - Re-encode into VAE latent space.
   - Add noise at the appropriate timestep.
   - Denoise for a small number of steps.

This process repeats until reaching 8K (8192 × 8192), producing coherent ultra-high-resolution remote sensing images without any model fine-tuning.

### Noise Rescheduling

MegaFusion optionally applies resolution-dependent noise rescheduling that adjusts the noise schedule based on the upscale factor, preventing artifacts at high resolutions. Enable with `--if_reschedule`.

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

- **Text2Earth:** [huggingface.co/lcybuaa/Text2Earth](https://huggingface.co/lcybuaa/Text2Earth)
- **MegaFusion:** Wu et al., "MegaFusion: Extend Diffusion Models towards Higher-resolution Image Generation without Further Tuning", WACV 2025. [GitHub](https://github.com/haoningwu3639/MegaFusion)
- **XLRS-Bench:** [huggingface.co/datasets/initiacms/XLRS-Bench_caption_en](https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en)

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.