"""Tests for the MegaFusion DDIM scheduler with noise rescheduling."""

import math
import pytest
import torch

from rsgen8k.models.scheduler import (
    MegaFusionDDIMScheduler,
    compute_rescheduled_alphas_cumprod,
    RESCHEDULE_COEFFICIENTS,
)


class TestComputeRescheduledAlphasCumprod:
    """Tests for the rescheduling formula."""

    def test_no_reschedule_at_base_resolution(self):
        """At base resolution the alphas_cumprod should remain unchanged."""
        alphas = torch.linspace(0.999, 0.01, 100)
        result = compute_rescheduled_alphas_cumprod(alphas, base_resolution=512, target_resolution=512)
        assert torch.allclose(result, alphas)

    def test_reschedule_reduces_alphas(self):
        """Rescheduling at higher resolution should reduce alphas_cumprod values."""
        alphas = torch.linspace(0.999, 0.01, 100)
        result = compute_rescheduled_alphas_cumprod(alphas, base_resolution=512, target_resolution=1024)
        # Higher resolution means more noise, so alphas should be smaller
        assert (result <= alphas).all()

    def test_higher_resolution_means_more_reschedule(self):
        """Larger upscale factors should produce smaller alphas_cumprod."""
        alphas = torch.linspace(0.999, 0.01, 100)
        result_2x = compute_rescheduled_alphas_cumprod(alphas, 512, 1024)
        result_4x = compute_rescheduled_alphas_cumprod(alphas, 512, 2048)
        assert (result_4x <= result_2x).all()

    def test_known_coefficients_1_5x(self):
        """Verify the 1.5× scaling formula matches pre-defined coefficients."""
        alphas = torch.tensor([0.9, 0.5, 0.1])
        result = compute_rescheduled_alphas_cumprod(alphas, 512, 768)
        # Manual: alpha / (2.25 - 1.25 * alpha)
        expected = alphas / (2.25 - 1.25 * alphas)
        assert torch.allclose(result, expected)

    def test_known_coefficients_2x(self):
        """Verify the 2× scaling formula matches pre-defined coefficients."""
        alphas = torch.tensor([0.9, 0.5, 0.1])
        result = compute_rescheduled_alphas_cumprod(alphas, 512, 1024)
        expected = alphas / (4.0 - 3.0 * alphas)
        assert torch.allclose(result, expected)

    def test_arbitrary_factor(self):
        """Verify the general formula for non-standard factors."""
        alphas = torch.tensor([0.9, 0.5, 0.1])
        # Factor = 1200/512 ~ 2.34
        result = compute_rescheduled_alphas_cumprod(alphas, 512, 1200)
        factor = 1200 / 512
        r2 = factor * factor
        expected = alphas / (r2 - (r2 - 1.0) * alphas)
        assert torch.allclose(result, expected)

    def test_output_bounded(self):
        """Rescheduled alphas_cumprod should stay in (0, 1)."""
        alphas = torch.linspace(0.999, 0.001, 1000)
        for factor in [1.5, 2.0, 4.0, 8.0, 16.0]:
            result = compute_rescheduled_alphas_cumprod(alphas, 512, int(512 * factor))
            assert (result > 0).all()
            assert (result < 1).all()


class TestMegaFusionDDIMScheduler:
    """Tests for the scheduler class."""

    def test_init_default(self):
        scheduler = MegaFusionDDIMScheduler()
        assert scheduler.config.num_train_timesteps == 1000
        assert scheduler.config.base_resolution == 512
        assert scheduler.config.target_resolution == 512

    def test_set_timesteps(self):
        scheduler = MegaFusionDDIMScheduler()
        scheduler.set_timesteps(50)
        assert len(scheduler.timesteps) == 50
        # Timesteps should be decreasing
        for i in range(len(scheduler.timesteps) - 1):
            assert scheduler.timesteps[i] > scheduler.timesteps[i + 1]

    def test_step_output_shape(self):
        scheduler = MegaFusionDDIMScheduler()
        scheduler.set_timesteps(50)
        sample = torch.randn(1, 4, 64, 64)
        model_output = torch.randn(1, 4, 64, 64)
        t = scheduler.timesteps[0]
        output = scheduler.step(model_output, t, sample)
        assert output.prev_sample.shape == sample.shape
        assert output.pred_original_sample.shape == sample.shape

    def test_add_noise_shape(self):
        scheduler = MegaFusionDDIMScheduler()
        scheduler.set_timesteps(50)
        original = torch.randn(1, 4, 64, 64)
        noise = torch.randn(1, 4, 64, 64)
        t = scheduler.timesteps[10]
        noisy = scheduler.add_noise(original, noise, t)
        assert noisy.shape == original.shape

    def test_rescheduled_scheduler_differs(self):
        """A scheduler configured for higher resolution should have different alphas."""
        base = MegaFusionDDIMScheduler(base_resolution=512, target_resolution=512)
        high = MegaFusionDDIMScheduler(base_resolution=512, target_resolution=1024)
        assert not torch.allclose(base.alphas_cumprod, high.alphas_cumprod)

    def test_from_pretrained_like_init(self):
        """Verify that from_config round-trips correctly."""
        sched = MegaFusionDDIMScheduler(
            beta_schedule="scaled_linear",
            base_resolution=512,
            target_resolution=1024,
        )
        config = dict(sched.config)
        sched2 = MegaFusionDDIMScheduler(**config)
        assert torch.allclose(sched.alphas_cumprod, sched2.alphas_cumprod)
