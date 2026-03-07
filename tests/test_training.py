"""Tests for the Diffusion-4K wavelet loss and training module."""

import pytest
import torch

from rsgen8k.training.wavelet_loss import WaveletLoss, haar_dwt_2d, haar_idwt_2d


# ---------------------------------------------------------------------------
# Haar DWT / IDWT tests
# ---------------------------------------------------------------------------

class TestHaarDWT:
    """Tests for the pure-PyTorch Haar DWT implementation."""

    def test_output_shapes(self):
        x = torch.randn(2, 4, 64, 64)
        ll, lh, hl, hh = haar_dwt_2d(x)
        assert ll.shape == (2, 4, 32, 32)
        assert lh.shape == (2, 4, 32, 32)
        assert hl.shape == (2, 4, 32, 32)
        assert hh.shape == (2, 4, 32, 32)

    def test_single_channel(self):
        x = torch.randn(1, 1, 8, 8)
        ll, lh, hl, hh = haar_dwt_2d(x)
        assert ll.shape == (1, 1, 4, 4)

    def test_non_square_input(self):
        x = torch.randn(1, 4, 16, 32)
        ll, lh, hl, hh = haar_dwt_2d(x)
        assert ll.shape == (1, 4, 8, 16)
        assert lh.shape == (1, 4, 8, 16)

    def test_rejects_odd_dimensions(self):
        x = torch.randn(1, 4, 7, 8)
        with pytest.raises(ValueError, match="even"):
            haar_dwt_2d(x)

    def test_rejects_non_4d(self):
        x = torch.randn(64, 64)
        with pytest.raises(ValueError, match="4-D"):
            haar_dwt_2d(x)

    def test_constant_input_all_energy_in_ll(self):
        """A constant image should have all energy in LL (the approximation)."""
        x = torch.ones(1, 1, 8, 8) * 5.0
        ll, lh, hl, hh = haar_dwt_2d(x)
        assert torch.allclose(lh, torch.zeros_like(lh), atol=1e-6)
        assert torch.allclose(hl, torch.zeros_like(hl), atol=1e-6)
        assert torch.allclose(hh, torch.zeros_like(hh), atol=1e-6)
        assert ll.abs().mean() > 0

    def test_energy_preservation(self):
        """DWT should preserve total energy (Parseval's theorem for Haar)."""
        x = torch.randn(2, 4, 16, 16)
        ll, lh, hl, hh = haar_dwt_2d(x)
        energy_input = (x ** 2).sum()
        energy_wavelet = (ll ** 2 + lh ** 2 + hl ** 2 + hh ** 2).sum()
        assert torch.allclose(energy_input, energy_wavelet, rtol=1e-4)


class TestHaarIDWT:
    """Tests for the inverse Haar DWT."""

    def test_perfect_reconstruction(self):
        x = torch.randn(2, 4, 32, 32)
        ll, lh, hl, hh = haar_dwt_2d(x)
        reconstructed = haar_idwt_2d(ll, lh, hl, hh)
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_output_shape(self):
        ll = torch.randn(1, 4, 16, 16)
        lh = torch.randn(1, 4, 16, 16)
        hl = torch.randn(1, 4, 16, 16)
        hh = torch.randn(1, 4, 16, 16)
        out = haar_idwt_2d(ll, lh, hl, hh)
        assert out.shape == (1, 4, 32, 32)


# ---------------------------------------------------------------------------
# WaveletLoss tests
# ---------------------------------------------------------------------------

class TestWaveletLoss:
    """Tests for the WaveletLoss module."""

    def test_zero_loss_for_identical_inputs(self):
        loss_fn = WaveletLoss()
        x = torch.randn(2, 4, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss_for_different_inputs(self):
        loss_fn = WaveletLoss()
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randn(2, 4, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0

    def test_output_is_scalar(self):
        loss_fn = WaveletLoss()
        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_with_weighting(self):
        loss_fn = WaveletLoss()
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        weighting = torch.ones(2, 1, 1, 1) * 2.0
        loss_weighted = loss_fn(pred, target, weighting=weighting)
        loss_unweighted = loss_fn(pred, target)
        # Weighted loss should be ~2x the unweighted loss
        assert loss_weighted.item() > loss_unweighted.item()

    def test_symmetric(self):
        loss_fn = WaveletLoss()
        a = torch.randn(1, 4, 16, 16)
        b = torch.randn(1, 4, 16, 16)
        assert loss_fn(a, b).item() == pytest.approx(loss_fn(b, a).item(), rel=1e-5)

    def test_gradient_flows(self):
        loss_fn = WaveletLoss()
        pred = torch.randn(1, 4, 16, 16, requires_grad=True)
        target = torch.randn(1, 4, 16, 16)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_rejects_multi_level(self):
        with pytest.raises(ValueError, match="levels=1"):
            WaveletLoss(levels=2)

    def test_different_batch_sizes(self):
        loss_fn = WaveletLoss()
        for bs in [1, 4, 8]:
            pred = torch.randn(bs, 4, 16, 16)
            target = torch.randn(bs, 4, 16, 16)
            loss = loss_fn(pred, target)
            assert loss.ndim == 0

    def test_different_channel_counts(self):
        loss_fn = WaveletLoss()
        for c in [1, 3, 4, 8]:
            pred = torch.randn(1, c, 16, 16)
            target = torch.randn(1, c, 16, 16)
            loss = loss_fn(pred, target)
            assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Training module import tests
# ---------------------------------------------------------------------------

class TestTrainingModuleImports:
    """Verify that training module components are importable."""

    def test_import_wavelet_loss(self):
        from rsgen8k.training import WaveletLoss, haar_dwt_2d
        assert callable(WaveletLoss)
        assert callable(haar_dwt_2d)

    def test_import_trainer(self):
        from rsgen8k.training.trainer import parse_args, ImageCaptionDataset
        assert callable(parse_args)
        assert callable(ImageCaptionDataset)

    def test_parse_args_defaults(self):
        from rsgen8k.training.trainer import parse_args
        args = parse_args([
            "--pretrained_model_name_or_path", "./model",
            "--instance_data_dir", "./data",
        ])
        assert args.resolution == 2048
        assert args.learning_rate == 1e-6
        assert args.wave == "haar"
        assert args.mixed_precision == "bf16"
        assert args.train_batch_size == 2
