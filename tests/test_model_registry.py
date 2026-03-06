"""Tests for the model registry."""

import os
import tempfile
from unittest.mock import patch

import pytest

from rsgen8k.models.model_registry import (
    MODEL_REGISTRY,
    get_model_info,
    list_models,
    resolve_model_path,
)


class TestModelRegistry:
    """Tests for the base model registry."""

    def test_registry_contains_text2earth(self):
        assert "text2earth" in MODEL_REGISTRY

    def test_registry_contains_diffusionsat(self):
        assert "diffusionsat" in MODEL_REGISTRY

    def test_registry_contains_geosynth(self):
        assert "geosynth" in MODEL_REGISTRY

    def test_get_model_info_text2earth(self):
        info = get_model_info("text2earth")
        assert info.model_id == "lcybuaa/Text2Earth"
        assert info.architecture == "sd1.5"
        assert info.base_resolution == 512

    def test_get_model_info_diffusionsat(self):
        info = get_model_info("diffusionsat")
        assert info.model_id == "BiliSakura/DiffusionSat-Single-512"
        assert info.base_resolution == 512

    def test_get_model_info_geosynth(self):
        info = get_model_info("geosynth")
        assert info.model_id == "MVRL/GeoSynth"
        assert info.base_resolution == 512

    def test_get_model_info_case_insensitive(self):
        info = get_model_info("Text2Earth")
        assert info.model_id == "lcybuaa/Text2Earth"

    def test_get_model_info_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_info("nonexistent_model")

    def test_list_models_returns_all(self):
        models = list_models()
        assert len(models) >= 3
        assert "text2earth" in models
        assert "diffusionsat" in models
        assert "geosynth" in models

    def test_all_models_have_url(self):
        for key, info in MODEL_REGISTRY.items():
            assert info.url.startswith("https://"), f"Model {key} missing URL"

    def test_all_models_have_architecture(self):
        for key, info in MODEL_REGISTRY.items():
            assert info.architecture, f"Model {key} missing architecture"


class TestResolveModelPath:
    """Tests for resolve_model_path local checkpoint resolution."""

    def test_returns_hub_id_when_no_local_dir(self):
        """When no local dir exists, return the original model ID."""
        result = resolve_model_path("lcybuaa/Text2Earth", ckpt_dir="/nonexistent")
        assert result == "lcybuaa/Text2Earth"

    def test_returns_hub_id_when_dir_exists_but_empty(self):
        """An empty local dir should not be treated as a valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "lcybuaa", "Text2Earth")
            os.makedirs(model_dir)
            result = resolve_model_path("lcybuaa/Text2Earth", ckpt_dir=tmpdir)
            assert result == "lcybuaa/Text2Earth"

    def test_returns_local_path_with_model_index(self):
        """A local dir with model_index.json should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "lcybuaa", "Text2Earth")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "model_index.json"), "w") as f:
                f.write("{}")
            result = resolve_model_path("lcybuaa/Text2Earth", ckpt_dir=tmpdir)
            assert os.path.isabs(result)
            assert result == os.path.abspath(model_dir)

    def test_returns_local_path_with_unet_subfolder(self):
        """A local dir with unet/ subfolder should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "MVRL", "GeoSynth")
            os.makedirs(os.path.join(model_dir, "unet"))
            result = resolve_model_path("MVRL/GeoSynth", ckpt_dir=tmpdir)
            assert os.path.isabs(result)
            assert result == os.path.abspath(model_dir)

    def test_default_ckpt_dir(self):
        """When ckpt_dir is None the default ckpt_dir is used. If no local dir
        exists, return the Hub ID."""
        with patch("rsgen8k.models.model_registry.DEFAULT_CKPT_DIR", tempfile.mkdtemp()):
            result = resolve_model_path("lcybuaa/Text2Earth")
        assert result == "lcybuaa/Text2Earth"
