#!/usr/bin/env python
"""RSGen-8k Gradio Web Demo.

Launch an interactive web interface for remote sensing image generation
at up to 8K resolution.

Usage:
    python app.py
    python app.py --port 7860 --share
    python app.py --ckpt_dir /data/models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

from rsgen8k.generate import (
    GenerationConfig,
    generate,
    DEFAULT_STAGE_RESOLUTIONS,
    DEFAULT_STAGE_STEPS,
    AVAILABLE_MODELS,
    AVAILABLE_TECHNIQUES,
)
from rsgen8k.models.model_registry import list_models
from rsgen8k.techniques.registry import list_techniques

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presets for resolution stages
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "512 → 8K (full)": {
        "resolutions": "512 1024 2048 4096 8192",
        "steps": "40 3 3 2 2",
    },
    "512 → 4K": {
        "resolutions": "512 1024 2048 4096",
        "steps": "40 3 3 4",
    },
    "512 → 2K": {
        "resolutions": "512 1024 2048",
        "steps": "40 5 5",
    },
    "512 → 1K": {
        "resolutions": "512 1024",
        "steps": "40 10",
    },
    "512 only (base)": {
        "resolutions": "512",
        "steps": "50",
    },
}


def _make_run_generation(ckpt_dir: str = "./ckpt"):
    """Create a generation function bound to the given checkpoint directory."""

    def _run_generation(
        model_name: str,
        technique: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        resolution_preset: str,
        custom_resolutions: str,
        custom_steps: str,
        mixed_precision: str,
        enable_reschedule: bool,
        enable_vae_tiling: bool,
        deterministic: bool,
    ):
        """Run generation and return the resulting image."""
        # Resolve resolution stages
        if resolution_preset != "Custom":
            preset = RESOLUTION_PRESETS[resolution_preset]
            res_str = preset["resolutions"]
            steps_str = preset["steps"]
        else:
            res_str = custom_resolutions.strip()
            steps_str = custom_steps.strip()

        try:
            stage_resolutions = [int(x) for x in res_str.split()]
            stage_steps = [int(x) for x in steps_str.split()]
        except ValueError:
            raise gr.Error("Invalid resolution or steps format. Use space-separated integers.")

        if len(stage_resolutions) != len(stage_steps):
            raise gr.Error(
                f"Number of resolutions ({len(stage_resolutions)}) must match "
                f"number of step values ({len(stage_steps)})."
            )

        config = GenerationConfig(
            model_name=model_name.lower(),
            technique=technique.lower(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed >= 0 else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            stage_resolutions=stage_resolutions,
            stage_steps=stage_steps,
            mixed_precision=mixed_precision,
            if_reschedule=enable_reschedule,
            vae_tiling=enable_vae_tiling,
            deterministic=deterministic,
            output_dir="./outputs/gradio",
            ckpt_dir=ckpt_dir,
        )

        image = generate(config)
        return image

    return _run_generation


def _update_preset(preset_name):
    """Update custom fields when a preset is selected."""
    if preset_name == "Custom":
        return gr.update(interactive=True), gr.update(interactive=True)
    preset = RESOLUTION_PRESETS[preset_name]
    return (
        gr.update(value=preset["resolutions"], interactive=False),
        gr.update(value=preset["steps"], interactive=False),
    )


def build_demo(ckpt_dir: str = "./ckpt") -> gr.Blocks:
    """Construct the Gradio interface.

    Args:
        ckpt_dir: Local checkpoint directory for model loading.
    """
    run_generation = _make_run_generation(ckpt_dir)
    model_choices = list(AVAILABLE_MODELS)
    technique_choices = list(AVAILABLE_TECHNIQUES)
    preset_choices = list(RESOLUTION_PRESETS.keys()) + ["Custom"]

    # Build model and technique descriptions for info display
    model_info_lines = []
    for key, info in list_models().items():
        model_info_lines.append(f"**{key}** → `{info.model_id}` ({info.architecture})")
    model_info_md = "\n\n".join(model_info_lines)

    tech_info_lines = []
    for key, info in list_techniques().items():
        desc = info.description
        if len(desc) > 100:
            desc = desc[:100].rsplit(" ", 1)[0] + "…"
        tech_info_lines.append(f"**{key}** — {info.name}: {desc}")
    tech_info_md = "\n\n".join(tech_info_lines)

    with gr.Blocks(
        title="RSGen-8k: Remote Sensing Image Generation",
    ) as demo:
        gr.Markdown(
            "# 🛰️ RSGen-8k: Remote Sensing Image Generation at 8K Resolution\n"
            "Generate satellite/aerial imagery from text prompts using progressive "
            "upscaling techniques. Models load from `./ckpt` (local) or HuggingFace Hub."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A high-resolution satellite image of a coastal city with harbors and ships",
                    lines=3,
                    value="A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted",
                    lines=1,
                    value="",
                )

                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=model_choices,
                        value="text2earth",
                        label="Base Model",
                    )
                    technique = gr.Dropdown(
                        choices=technique_choices,
                        value="megafusion",
                        label="Upscaling Technique",
                    )

                with gr.Row():
                    seed = gr.Number(
                        value=42,
                        label="Seed (-1 = random)",
                        precision=0,
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=7.0,
                        step=0.5, label="Guidance Scale",
                    )

                num_inference_steps = gr.Slider(
                    minimum=10, maximum=100, value=50,
                    step=1, label="Total Inference Steps",
                )

                resolution_preset = gr.Dropdown(
                    choices=preset_choices,
                    value="512 → 2K",
                    label="Resolution Preset",
                )

                with gr.Row():
                    custom_resolutions = gr.Textbox(
                        label="Stage Resolutions",
                        value="512 1024 2048",
                        interactive=False,
                    )
                    custom_steps = gr.Textbox(
                        label="Stage Steps",
                        value="40 5 5",
                        interactive=False,
                    )

                with gr.Row():
                    mixed_precision = gr.Dropdown(
                        choices=["fp16", "bf16", "no"],
                        value="fp16",
                        label="Precision",
                    )
                    enable_reschedule = gr.Checkbox(
                        label="Noise Rescheduling",
                        value=False,
                    )
                    enable_vae_tiling = gr.Checkbox(
                        label="VAE Tiling",
                        value=True,
                    )
                    deterministic = gr.Checkbox(
                        label="Deterministic",
                        value=False,
                    )

                generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                )

        # Wire preset selector to update custom fields
        resolution_preset.change(
            fn=_update_preset,
            inputs=[resolution_preset],
            outputs=[custom_resolutions, custom_steps],
        )

        # Wire generate button
        generate_btn.click(
            fn=run_generation,
            inputs=[
                model_name,
                technique,
                prompt,
                negative_prompt,
                seed,
                guidance_scale,
                num_inference_steps,
                resolution_preset,
                custom_resolutions,
                custom_steps,
                mixed_precision,
                enable_reschedule,
                enable_vae_tiling,
                deterministic,
            ],
            outputs=[output_image],
        )

        with gr.Accordion("ℹ️ Available Models", open=False):
            gr.Markdown(model_info_md)

        with gr.Accordion("ℹ️ Available Techniques", open=False):
            gr.Markdown(tech_info_md)

    return demo


def main():
    parser = argparse.ArgumentParser(description="RSGen-8k Gradio Web Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt",
                        help="Local checkpoint directory (default: ./ckpt)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Checkpoint directory: %s", os.path.abspath(args.ckpt_dir))

    demo = build_demo(ckpt_dir=args.ckpt_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
