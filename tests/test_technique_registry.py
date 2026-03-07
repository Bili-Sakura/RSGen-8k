"""Tests for the technique registry."""

import pytest

from rsgen8k.techniques.registry import (
    TECHNIQUE_REGISTRY,
    get_technique,
    list_techniques,
)


class TestTechniqueRegistry:
    """Tests for the technique registry."""

    EXPECTED_TECHNIQUES = [
        "native",
        "megafusion",
        "elasticdiffusion",
        "multidiffusion",
        "freescale",
        "demofusion",
        "fouriscale",
        "inftydiff",
    ]

    def test_all_expected_techniques_registered(self):
        for key in self.EXPECTED_TECHNIQUES:
            assert key in TECHNIQUE_REGISTRY, f"Technique '{key}' not found"

    def test_get_technique_native(self):
        info = get_technique("native")
        assert info.name == "Native DDIM"
        assert "DDIM" in info.description

    def test_get_technique_megafusion(self):
        info = get_technique("megafusion")
        assert info.name == "MegaFusion"
        assert "sd1.5" in info.supported_architectures

    def test_get_technique_multidiffusion(self):
        info = get_technique("multidiffusion")
        assert info.name == "MultiDiffusion"
        assert info.github_url == "https://github.com/omerbt/MultiDiffusion"

    def test_get_technique_elasticdiffusion(self):
        info = get_technique("elasticdiffusion")
        assert info.name == "ElasticDiffusion"
        assert "CVPR 2024" in info.paper

    def test_get_technique_freescale(self):
        info = get_technique("freescale")
        assert info.name == "FreeScale"
        assert "sdxl" in info.supported_architectures

    def test_get_technique_demofusion(self):
        info = get_technique("demofusion")
        assert info.name == "DemoFusion"
        assert "CVPR 2024" in info.paper

    def test_get_technique_fouriscale(self):
        info = get_technique("fouriscale")
        assert info.name == "FouriScale"
        assert "ECCV 2024" in info.paper

    def test_get_technique_inftydiff(self):
        info = get_technique("inftydiff")
        assert info.name == "∞-Diff"
        assert "ICLR 2024" in info.paper
        assert "sd1.5" in info.supported_architectures

    def test_get_technique_case_insensitive(self):
        info = get_technique("MegaFusion")
        assert info.key == "megafusion"

    def test_get_technique_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown technique"):
            get_technique("nonexistent")

    def test_list_techniques_returns_all(self):
        techniques = list_techniques()
        assert len(techniques) >= 6
        for key in self.EXPECTED_TECHNIQUES:
            assert key in techniques

    def test_all_techniques_have_github_url(self):
        for key, info in TECHNIQUE_REGISTRY.items():
            assert info.github_url.startswith("https://"), f"{key} missing URL"

    def test_all_techniques_have_paper_citation(self):
        for key, info in TECHNIQUE_REGISTRY.items():
            assert len(info.paper) > 10, f"{key} missing paper citation"

    def test_all_techniques_have_supported_architectures(self):
        for key, info in TECHNIQUE_REGISTRY.items():
            assert len(info.supported_architectures) > 0, f"{key} has no architectures"
