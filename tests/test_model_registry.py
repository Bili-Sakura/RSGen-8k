"""Tests for the model registry."""

import pytest

from rsgen8k.models.model_registry import (
    MODEL_REGISTRY,
    get_model_info,
    list_models,
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
