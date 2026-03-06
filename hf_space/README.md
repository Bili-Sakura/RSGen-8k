---
title: RSGen-8k
sdk: gradio
app_file: app.py
---

# 🛰️ RSGen-8k: Remote Sensing Image Generation at 8K

Generate satellite/aerial imagery from text prompts using progressive upscaling. Models load from **Hugging Face Hub**.

## Models (from HuggingFace)

| Model | HF ID |
|-------|-------|
| Text2Earth | `lcybuaa/Text2Earth` |
| DiffusionSat | `BiliSakura/DiffusionSat-Single-512` |
| GeoSynth | `MVRL/GeoSynth` |
| DDPM-CD | `BiliSakura/ddpm-cd-pretrained-256` |

## Techniques

MegaFusion (default), MultiDiffusion, ElasticDiffusion, FreeScale, DemoFusion, FouriScale.

---

**ZeroGPU:** This Space uses [ZeroGPU](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) for free dynamic GPU allocation. Select **ZeroGPU** in Space settings when creating.
