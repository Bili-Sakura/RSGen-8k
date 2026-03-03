"""Tests for individual technique module imports and core functions."""

import pytest
import torch


class TestMultiDiffusion:
    """Tests for MultiDiffusion technique."""

    def test_get_views_basic(self):
        from rsgen8k.techniques.multi_diffusion import _get_views
        views = _get_views(128, 128, window_size=64, stride=32)
        assert len(views) > 0
        for r_start, r_end, c_start, c_end in views:
            assert r_end - r_start == 64
            assert c_end - c_start == 64
            assert r_end <= 128
            assert c_end <= 128

    def test_get_views_covers_entire_canvas(self):
        from rsgen8k.techniques.multi_diffusion import _get_views
        h, w = 128, 128
        views = _get_views(h, w, window_size=64, stride=32)
        covered = torch.zeros(h, w)
        for r_start, r_end, c_start, c_end in views:
            covered[r_start:r_end, c_start:c_end] = 1
        assert covered.all(), "Not all pixels are covered by views"


class TestElasticDiffusion:
    """Tests for ElasticDiffusion technique."""

    def test_get_views_basic(self):
        from rsgen8k.techniques.elastic_diffusion import _get_views
        views = _get_views(128, 128, window_size=64, stride=32)
        assert len(views) > 0

    def test_module_imports(self):
        from rsgen8k.techniques.elastic_diffusion import elastic_diffusion_denoise_step
        assert callable(elastic_diffusion_denoise_step)


class TestFreeScale:
    """Tests for FreeScale technique."""

    def test_cosine_schedule_bounds(self):
        from rsgen8k.techniques.freescale import cosine_scale_schedule
        for step in range(50):
            w = cosine_scale_schedule(step, 50, min_scale=0.0, max_scale=1.0)
            assert 0.0 <= w <= 1.0

    def test_cosine_schedule_decreasing(self):
        from rsgen8k.techniques.freescale import cosine_scale_schedule
        weights = [cosine_scale_schedule(i, 50) for i in range(50)]
        # Should generally decrease (not necessarily strictly)
        assert weights[0] >= weights[-1]

    def test_module_imports(self):
        from rsgen8k.techniques.freescale import freescale_denoise_step
        assert callable(freescale_denoise_step)


class TestDemoFusion:
    """Tests for DemoFusion technique."""

    def test_gaussian_blur_shape(self):
        from rsgen8k.techniques.demofusion import _gaussian_blur
        x = torch.randn(1, 4, 64, 64)
        out = _gaussian_blur(x, kernel_size=3)
        assert out.shape == x.shape

    def test_module_imports(self):
        from rsgen8k.techniques.demofusion import demofusion_denoise_step
        assert callable(demofusion_denoise_step)


class TestFouriScale:
    """Tests for FouriScale technique."""

    def test_low_pass_filter_shape(self):
        from rsgen8k.techniques.fouriscale import low_pass_filter
        x = torch.randn(1, 4, 64, 64)
        out = low_pass_filter(x, cutoff_ratio=0.5)
        assert out.shape == x.shape

    def test_low_pass_filter_full_passthrough(self):
        from rsgen8k.techniques.fouriscale import low_pass_filter
        x = torch.randn(1, 4, 64, 64)
        # With cutoff_ratio=1.0, the circular mask covers nearly all
        # frequencies; verify the output is close to the input (the
        # circular mask leaves corner frequencies slightly outside).
        out = low_pass_filter(x, cutoff_ratio=1.0)
        # Most energy should be preserved
        assert (out - x).abs().mean() < 0.5

    def test_module_imports(self):
        from rsgen8k.techniques.fouriscale import fouriscale_denoise_step
        assert callable(fouriscale_denoise_step)


class TestMegaFusionTechnique:
    """Tests for MegaFusion technique module."""

    def test_module_imports(self):
        from rsgen8k.techniques.megafusion import run_megafusion
        assert callable(run_megafusion)
