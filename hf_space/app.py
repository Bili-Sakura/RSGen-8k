#!/usr/bin/env python
"""RSGen-8k Gradio Demo for Hugging Face Spaces.

Standalone app that loads models from Hugging Face Hub.
Uses ZeroGPU for free dynamic GPU allocation.
Upload this folder to create a Space.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Must import spaces before any GPU/CUDA imports (required for ZeroGPU)
import spaces
import gradio as gr

from rsgen8k import generate as gen_module
from rsgen8k.generate import (
    GenerationConfig,
    generate,
    AVAILABLE_MODELS,
    AVAILABLE_TECHNIQUES,
)
from rsgen8k.models.model_registry import list_models
from rsgen8k.techniques.registry import list_techniques

logger = logging.getLogger(__name__)

SCHEDULER_CHOICES = [
    "ddim", "euler", "euler_ancestral", "dpmsolver_multistep",
    "dpmsolver_singlestep", "pndm", "lms", "heun", "dpm2", "dpm2_ancestral",
]
MAX_HISTORY_IMAGES = 24

RESOLUTION_PRESETS = {
    "512 → 8K (full)": {"resolutions": "512 1024 2048 4096 8192", "steps": "40 3 3 2 2"},
    "512 → 4K": {"resolutions": "512 1024 2048 4096", "steps": "40 3 3 4"},
    "512 → 2K": {"resolutions": "512 1024 2048", "steps": "40 5 5"},
    "512 → 1K": {"resolutions": "512 1024", "steps": "40 10"},
    "512 only (base)": {"resolutions": "512", "steps": "50"},
}


def _zerogpu_duration(
    model_name, technique, prompt, negative_prompt, seed, guidance_scale,
    num_inference_steps, resolution_preset, custom_resolutions, custom_steps,
    mixed_precision, enable_reschedule, enable_vae_tiling, deterministic,
    native_scheduler, batch_size, history,
):
    """Estimate GPU duration from resolution preset / steps."""
    if resolution_preset != "Custom":
        steps = [int(x) for x in RESOLUTION_PRESETS[resolution_preset]["steps"].split()]
    else:
        try:
            steps = [int(x) for x in custom_steps.strip().split()]
        except ValueError:
            steps = [40, 5, 5]
    total_steps = sum(steps)
    return min(300, max(120, int(total_steps * 2.5)))  # ~2.5s per step, 2–5 min cap


@spaces.GPU(duration=_zerogpu_duration)
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
    native_scheduler: str,
    batch_size: int,
    history: list,
):
    if resolution_preset != "Custom":
        preset = RESOLUTION_PRESETS[resolution_preset]
        res_str, steps_str = preset["resolutions"], preset["steps"]
    else:
        res_str, steps_str = custom_resolutions.strip(), custom_steps.strip()

    try:
        stage_resolutions = [int(x) for x in res_str.split()]
        stage_steps = [int(x) for x in steps_str.split()]
    except ValueError:
        raise gr.Error("Invalid resolution or steps format. Use space-separated integers.")

    if len(stage_resolutions) != len(stage_steps):
        raise gr.Error("Number of resolutions must match number of step values.")

    batch_size = int(batch_size)
    if batch_size < 1 or batch_size > 8:
        raise gr.Error("Batch size must be between 1 and 8.")

    # ZeroGPU releases GPU after each call; clear cache to avoid stale GPU refs
    gen_module._PIPELINE_CACHE.clear()

    output_dir = f"./outputs/run_{int(time.time() * 1000)}"
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
        native_scheduler=(native_scheduler or "ddim").lower(),
        batch_size=batch_size,
        output_dir=output_dir,
        ckpt_dir=".",  # Models from HuggingFace Hub only
    )

    result = generate(config)
    images = result if isinstance(result, list) else [result]
    sample_seed = config.seed if config.seed is not None else "rand"

    new_items = []
    for i, img in enumerate(images):
        caption = f"Seed {sample_seed}" + (f" • Batch {i + 1}/{len(images)}" if len(images) > 1 else "")
        new_items.append((img, caption))

    history = history or []
    updated_history = new_items + history
    updated_history = updated_history[:MAX_HISTORY_IMAGES]
    return updated_history, updated_history


def _update_preset(preset_name):
    if preset_name == "Custom":
        return gr.update(interactive=True), gr.update(interactive=True)
    preset = RESOLUTION_PRESETS[preset_name]
    return (
        gr.update(value=preset["resolutions"], interactive=False),
        gr.update(value=preset["steps"], interactive=False),
    )


def build_demo():
    run_generation = _run_generation
    model_choices = list(AVAILABLE_MODELS)
    technique_choices = list(AVAILABLE_TECHNIQUES)
    preset_choices = list(RESOLUTION_PRESETS.keys()) + ["Custom"]

    model_info_lines = [f"**{k}** → `{v.model_id}` ({v.architecture})" for k, v in list_models().items()]
    model_info_md = "\n\n".join(model_info_lines)
    tech_info_lines = []
    for k, v in list_techniques().items():
        desc = v.description[:100].rsplit(" ", 1)[0] + "…" if len(v.description) > 100 else v.description
        tech_info_lines.append(f"**{k}** — {v.name}: {desc}")
    tech_info_md = "\n\n".join(tech_info_lines)

    with gr.Blocks(title="RSGen-8k: Remote Sensing Image Generation") as demo:
        gr.Markdown(
            "# 🛰️ RSGen-8k: Remote Sensing Image Generation at 8K\n"
            "Generate satellite/aerial imagery from text prompts. "
            "Models load from **Hugging Face Hub**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    value="A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.",
                )
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=1, value="")
                with gr.Row():
                    model_name = gr.Dropdown(choices=model_choices, value="text2earth", label="Base Model")
                    technique = gr.Dropdown(choices=technique_choices, value="megafusion", label="Upscaling Technique")
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
                num_inference_steps = gr.Slider(10, 100, value=50, step=1, label="Total Inference Steps")
                resolution_preset = gr.Dropdown(choices=preset_choices, value="512 → 2K", label="Resolution Preset")
                with gr.Row():
                    custom_resolutions = gr.Textbox(label="Stage Resolutions", value="512 1024 2048", interactive=False)
                    custom_steps = gr.Textbox(label="Stage Steps", value="40 5 5", interactive=False)
                with gr.Row():
                    mixed_precision = gr.Dropdown(choices=["fp16", "bf16", "no"], value="bf16", label="Precision")
                    enable_reschedule = gr.Checkbox(value=False, label="Noise Rescheduling")
                    enable_vae_tiling = gr.Checkbox(value=True, label="VAE Tiling")
                    deterministic = gr.Checkbox(value=False, label="Deterministic")
                with gr.Accordion("Advanced (Native technique)", open=False):
                    native_scheduler = gr.Dropdown(choices=SCHEDULER_CHOICES, value="ddim", label="Scheduler")
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            with gr.Column(scale=1):
                history_state = gr.State(value=[])
                output_gallery = gr.Gallery(label="Generated Images", columns=4, rows=2, height="auto", object_fit="contain")

        resolution_preset.change(_update_preset, [resolution_preset], [custom_resolutions, custom_steps])
        generate_btn.click(
            run_generation,
            inputs=[
                model_name, technique, prompt, negative_prompt, seed, guidance_scale,
                num_inference_steps, resolution_preset, custom_resolutions, custom_steps,
                mixed_precision, enable_reschedule, enable_vae_tiling, deterministic,
                native_scheduler, batch_size, history_state,
            ],
            outputs=[output_gallery, history_state],
        )
        clear_btn.click(lambda: ([], []), outputs=[output_gallery, history_state])
        with gr.Accordion("ℹ️ Available Models", open=False):
            gr.Markdown(model_info_md)
        with gr.Accordion("ℹ️ Available Techniques", open=False):
            gr.Markdown(tech_info_md)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
