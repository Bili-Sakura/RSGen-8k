"""Tests for rsgen8k.metrics module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from rsgen8k.metrics import (
    compute_psnr,
    compute_ssim,
    evaluate_directory,
)


@pytest.fixture
def gen_dir_with_image():
    """Create temp dir with a minimal placeholder PNG for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a 64x64 gray PNG
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        path = os.path.join(tmp, "test.png")
        img.save(path)
        yield tmp, path


def test_evaluate_directory_empty_dir():
    """evaluate_directory on empty dir returns num_images=0."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "results.json")
        result = evaluate_directory(
            gen_dir=tmp,
            out_file=out,
            verbose=False,
        )
        assert result["num_images"] == 0
        assert result["metrics"] == {}
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert data["num_images"] == 0


def test_evaluate_directory_with_image(gen_dir_with_image):
    """evaluate_directory with image runs without error."""
    gen_dir, _ = gen_dir_with_image
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        result = evaluate_directory(
            gen_dir=gen_dir,
            out_file=out,
            metrics=["clip_score"],
            verbose=False,
        )
        assert result["num_images"] == 1
        assert "metrics" in result
        with open(out) as f:
            json.load(f)
    finally:
        os.unlink(out)


def test_compute_psnr_ssim_no_reference():
    """PSNR/SSIM with empty ref_dir returns None."""
    with tempfile.TemporaryDirectory() as gen_tmp:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(os.path.join(gen_tmp, "gen.png"))
        paths = [os.path.join(gen_tmp, "gen.png")]
    with tempfile.TemporaryDirectory() as ref_tmp:
        # ref_tmp is empty
        assert compute_psnr(paths, ref_tmp) is None
        assert compute_ssim(paths, ref_tmp) is None


def test_metrics_module_imports():
    """All metric functions are importable."""
    from rsgen8k.metrics import (
        compute_clip_score,
        compute_cmmd,
        compute_dino_similarity,
        compute_fid,
        compute_kid,
        compute_lpips,
        compute_psnr,
        compute_ssim,
        evaluate_directory,
    )
    assert callable(compute_fid)
    assert callable(compute_kid)
    assert callable(compute_cmmd)
    assert callable(compute_dino_similarity)
    assert callable(compute_clip_score)
    assert callable(compute_lpips)
    assert callable(compute_psnr)
    assert callable(compute_ssim)
    assert callable(evaluate_directory)
