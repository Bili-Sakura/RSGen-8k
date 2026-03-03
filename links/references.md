# References & Links

## Base Models

### Text2Earth
- **HuggingFace Model**: <https://huggingface.co/lcybuaa/Text2Earth>
- **Description**: Text-to-remote-sensing-image generation model based on Stable Diffusion, trained to generate high-quality remote sensing images from text descriptions.

### DiffusionSat-Single-512
- **HuggingFace Model**: <https://huggingface.co/BiliSakura/DiffusionSat-Single-512>
- **Description**: DiffusionSat single-image generation model at 512×512 resolution, designed for satellite image synthesis conditioned on text and metadata.

### GeoSynth
- **HuggingFace Model**: <https://huggingface.co/MVRL/GeoSynth>
- **Description**: GeoSynth generates realistic satellite imagery from text descriptions, fine-tuned on geospatial datasets.

## Upscaling Techniques

### MegaFusion
- **GitHub Repository**: <https://github.com/haoningwu3639/MegaFusion>
- **Paper**: Wu et al., "MegaFusion: Extend Diffusion Models towards Higher-resolution Image Generation without Further Tuning", WACV 2025.
- **Description**: Multi-stage progressive generation that decomposes the denoising process into stages at increasing resolutions with noise rescheduling.

### ElasticDiffusion
- **GitHub Repository**: <https://github.com/MoayedHajiAli/ElasticDiffusion-official>
- **Paper**: Haji Ali et al., "ElasticDiffusion: Training-free Arbitrary Size Image Generation through Global-Local Content Separation", CVPR 2024.
- **Description**: Decouples generation into local detail signals (overlapping patches) and global structural signals (reduced-resolution guidance), blending them at each step.

### MultiDiffusion
- **GitHub Repository**: <https://github.com/omerbt/MultiDiffusion>
- **Paper**: Bar-Tal et al., "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", ICML 2023.
- **Description**: Fuses multiple diffusion paths over overlapping sliding-window views using consensus voting for spatial consistency.

### FreeScale
- **GitHub Repository**: <https://github.com/ali-vilab/FreeScale>
- **Paper**: Qiu et al., "FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion", arXiv:2412.09626, 2024.
- **Description**: Tuning-free scale fusion via cosine-scheduled blending of high-resolution and low-resolution attention branches.

### DemoFusion
- **GitHub Repository**: <https://github.com/PRIS-CV/DemoFusion>
- **Paper**: Du et al., "DemoFusion: Democratising High-Resolution Image Generation With No $$$", CVPR 2024.
- **Description**: Progressive upscaling with skip-residual connections and dilated sampling for higher-resolution generation.

### FouriScale
- **GitHub Repository**: <https://github.com/LeonHLJ/FouriScale>
- **Paper**: Huang et al., "FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis", ECCV 2024.
- **Description**: Frequency-domain approach replacing convolutions with dilated variants plus low-pass Fourier filtering for structural consistency.

### XLRS-Bench (Text Prompt Dataset)
- **HuggingFace Dataset**: <https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en>
- **Description**: English text captions describing ultra-high-resolution remote sensing images, used as prompts for text-to-image generation.

## Libraries & Frameworks

| Library        | Purpose                                          | Link                                                       |
|----------------|--------------------------------------------------|------------------------------------------------------------|
| Diffusers      | Hugging Face diffusion model toolkit             | <https://github.com/huggingface/diffusers>                 |
| Transformers   | Hugging Face model loading and tokenization      | <https://github.com/huggingface/transformers>               |
| Accelerate     | Mixed-precision and distributed training/inference | <https://github.com/huggingface/accelerate>                |
| Datasets       | Hugging Face dataset loading                     | <https://github.com/huggingface/datasets>                   |
| PyTorch        | Deep learning framework                          | <https://pytorch.org>                                       |
| einops         | Tensor operations                                | <https://github.com/arogozhnikov/einops>                    |
| xformers       | Memory-efficient attention                       | <https://github.com/facebookresearch/xformers>              |
