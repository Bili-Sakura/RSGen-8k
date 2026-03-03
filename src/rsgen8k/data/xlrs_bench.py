"""Utilities for loading text prompts from the XLRS-Bench dataset.

The XLRS-Bench caption dataset provides English text descriptions of
ultra-high-resolution remote sensing images.  This module wraps the
HuggingFace ``datasets`` library to load and sample prompts for generation.

Dataset: https://huggingface.co/datasets/initiacms/XLRS-Bench_caption_en
"""

from __future__ import annotations

import random
from typing import List, Optional

DATASET_ID = "initiacms/XLRS-Bench_caption_en"


def load_xlrs_bench_prompts(
    num_prompts: Optional[int] = None,
    split: str = "train",
    seed: Optional[int] = None,
    caption_column: str = "caption_en",
    streaming: bool = False,
) -> List[str]:
    """Load text prompts from the XLRS-Bench caption dataset.

    Args:
        num_prompts: Maximum number of prompts to return.  If *None* the
            entire split is returned.
        split: Dataset split to load (default ``"train"``).
        seed: Random seed used when sampling a subset of prompts.
        caption_column: Name of the column containing text captions.  The
            loader falls back to ``"text"`` or the first string column
            when the specified column is not present.
        streaming: When *True*, load the dataset in streaming mode to
            avoid downloading the entire dataset at once.

    Returns:
        A list of prompt strings.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to load XLRS-Bench prompts. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(DATASET_ID, split=split, streaming=streaming)

    # Determine the correct caption column name
    if streaming:
        first_item = next(iter(dataset))
        columns = list(first_item.keys())
    else:
        columns = dataset.column_names

    if caption_column not in columns:
        # Fallback to common column names
        for fallback in ("text", "caption", "prompt"):
            if fallback in columns:
                caption_column = fallback
                break
        else:
            # Use the first string-valued column
            caption_column = columns[0]

    if streaming:
        prompts: List[str] = []
        for item in dataset:
            prompts.append(str(item[caption_column]))
            if num_prompts is not None and len(prompts) >= num_prompts:
                break
    else:
        prompts = [str(row) for row in dataset[caption_column]]

    if num_prompts is not None and num_prompts < len(prompts):
        rng = random.Random(seed)
        prompts = rng.sample(prompts, num_prompts)

    return prompts
