#!/usr/bin/env python
"""Download and preview prompts from the XLRS-Bench caption dataset.

Usage:
    python scripts/download_prompts.py --num_prompts 10 --output prompts.txt
    python scripts/download_prompts.py --preview 5
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsgen8k.data import load_xlrs_bench_prompts


def main():
    parser = argparse.ArgumentParser(description="Download XLRS-Bench remote sensing prompts")
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to download (all if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Save prompts to file")
    parser.add_argument("--preview", type=int, default=5, help="Number of prompts to preview")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (avoids full download)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("Loading prompts from XLRS-Bench dataset...")
    prompts = load_xlrs_bench_prompts(
        num_prompts=args.num_prompts,
        seed=args.seed,
        streaming=args.streaming,
    )
    logging.info("Loaded %d prompts", len(prompts))

    # Preview
    preview_count = min(args.preview, len(prompts))
    for i, prompt in enumerate(prompts[:preview_count]):
        print(f"\n[{i + 1}] {prompt}")

    # Save to file
    if args.output:
        with open(args.output, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")
        logging.info("Saved %d prompts to %s", len(prompts), args.output)


if __name__ == "__main__":
    main()
