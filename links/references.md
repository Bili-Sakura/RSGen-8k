# References & Links

## Core Components

### Text2Earth (Base Model)
- **HuggingFace Model**: <https://huggingface.co/lcybuaa/Text2Earth>
- **Description**: Text-to-remote-sensing-image generation model based on Stable Diffusion, trained to generate high-quality remote sensing images from text descriptions.

### MegaFusion (Upscaling Technique)
- **GitHub Repository**: <https://github.com/haoningwu3639/MegaFusion>
- **Paper**: Wu et al., "MegaFusion: Extend Diffusion Models towards Higher-resolution Image Generation without Further Tuning", WACV 2025.
- **Description**: A tuning-free technique that progressively generates higher-resolution images by decomposing the denoising process into multiple stages at increasing resolutions.

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
