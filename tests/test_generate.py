"""Tests for the generation configuration."""

import os
import tempfile

import pytest
import yaml

from rsgen8k.generate import GenerationConfig, DEFAULT_STAGE_RESOLUTIONS, DEFAULT_STAGE_STEPS


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self):
        config = GenerationConfig()
        assert config.model_path == "lcybuaa/Text2Earth"
        assert config.guidance_scale == 7.0
        assert config.num_inference_steps == 50
        assert config.stage_resolutions == list(DEFAULT_STAGE_RESOLUTIONS)
        assert config.stage_steps == list(DEFAULT_STAGE_STEPS)

    def test_stage_steps_sum_within_budget(self):
        """Total steps across stages should not exceed num_inference_steps."""
        config = GenerationConfig()
        assert sum(config.stage_steps) <= config.num_inference_steps

    def test_resolutions_and_steps_match(self):
        """Number of resolution stages must match number of step allocations."""
        config = GenerationConfig()
        assert len(config.stage_resolutions) == len(config.stage_steps)

    def test_resolutions_are_increasing(self):
        config = GenerationConfig()
        for i in range(len(config.stage_resolutions) - 1):
            assert config.stage_resolutions[i] < config.stage_resolutions[i + 1]

    def test_resolutions_divisible_by_8(self):
        """All resolutions must be divisible by 8 for the VAE."""
        config = GenerationConfig()
        for res in config.stage_resolutions:
            assert res % 8 == 0, f"Resolution {res} is not divisible by 8"

    def test_from_yaml(self):
        data = {
            "model_path": "test/model",
            "prompt": "test prompt",
            "guidance_scale": 5.0,
            "stage_resolutions": [512, 1024],
            "stage_steps": [40, 10],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            config = GenerationConfig.from_yaml(tmp_path)
            assert config.model_path == "test/model"
            assert config.prompt == "test prompt"
            assert config.guidance_scale == 5.0
            assert config.stage_resolutions == [512, 1024]
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_ignores_unknown_keys(self):
        data = {"model_path": "test/model", "unknown_key": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            config = GenerationConfig.from_yaml(tmp_path)
            assert config.model_path == "test/model"
            assert not hasattr(config, "unknown_key")
        finally:
            os.unlink(tmp_path)

    def test_custom_resolutions(self):
        """Verify custom stage configurations work."""
        config = GenerationConfig(
            stage_resolutions=[512, 1024, 2048],
            stage_steps=[40, 5, 5],
        )
        assert config.stage_resolutions[-1] == 2048
        assert sum(config.stage_steps) == 50
