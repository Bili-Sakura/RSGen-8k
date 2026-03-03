#!/usr/bin/env python
"""Generate remote sensing images using RSGen-8k.

Usage examples:

    # Generate with default settings (8K output)
    python scripts/generate.py --prompt "A satellite image of a coastal city"

    # Generate at custom resolution stages
    python scripts/generate.py \\
        --prompt "Dense urban area with skyscrapers" \\
        --stage_resolutions 512 1024 2048 \\
        --stage_steps 40 5 5

    # Generate using YAML configuration
    python scripts/generate.py --config configs/default.yaml

    # Generate with XLRS-Bench prompts
    python scripts/generate.py --from_dataset --num_prompts 5
"""

import argparse
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsgen8k.generate import GenerationConfig, generate, DEFAULT_STAGE_RESOLUTIONS, DEFAULT_STAGE_STEPS


def main():
    parser = argparse.ArgumentParser(
        description="RSGen-8k: Generate remote sensing images at up to 8K resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, default="lcybuaa/Text2Earth")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--stage_resolutions", type=int, nargs="+", default=DEFAULT_STAGE_RESOLUTIONS)
    parser.add_argument("--stage_steps", type=int, nargs="+", default=DEFAULT_STAGE_STEPS)
    parser.add_argument("--if_reschedule", action="store_true")
    parser.add_argument("--if_dilation", action="store_true")
    parser.add_argument("--no_xformers", action="store_true")
    parser.add_argument("--no_vae_tiling", action="store_true")

    # Dataset-based generation
    parser.add_argument("--from_dataset", action="store_true", help="Load prompts from XLRS-Bench dataset")
    parser.add_argument("--num_prompts", type=int, default=1, help="Number of prompts to sample from dataset")
    parser.add_argument("--dataset_seed", type=int, default=42, help="Seed for sampling prompts from dataset")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.config:
        config = GenerationConfig.from_yaml(args.config)
    else:
        config = GenerationConfig()

    for key in ("model_path", "negative_prompt", "output_dir", "seed",
                "mixed_precision", "guidance_scale", "num_inference_steps",
                "stage_resolutions", "stage_steps", "if_reschedule", "if_dilation"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    config.enable_xformers = not args.no_xformers
    config.vae_tiling = not args.no_vae_tiling

    if args.from_dataset:
        from rsgen8k.data import load_xlrs_bench_prompts

        prompts = load_xlrs_bench_prompts(num_prompts=args.num_prompts, seed=args.dataset_seed)
        logging.info("Loaded %d prompts from XLRS-Bench", len(prompts))
        for i, prompt in enumerate(prompts):
            logging.info("Generating image %d/%d: %s", i + 1, len(prompts), prompt[:80])
            config.prompt = prompt
            generate(config)
    else:
        if args.prompt:
            config.prompt = args.prompt
        if config.prompt is None:
            logging.error("No prompt specified. Use --prompt, --config, or --from_dataset.")
            sys.exit(1)
        generate(config)


if __name__ == "__main__":
    main()
