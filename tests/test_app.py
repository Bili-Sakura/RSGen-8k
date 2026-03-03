"""Tests for the Gradio web demo."""

import os
import sys

import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture()
def app_module():
    """Import app.py as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", _APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGradioDemo:
    """Tests for the Gradio web demo."""

    def test_import_app(self, app_module):
        """Verify app.py imports without error."""
        assert hasattr(app_module, "build_demo")
        assert hasattr(app_module, "RESOLUTION_PRESETS")

    def test_build_demo_returns_blocks(self, app_module):
        """Verify build_demo() creates a valid Gradio Blocks."""
        demo = app_module.build_demo()
        import gradio as gr
        assert isinstance(demo, gr.Blocks)

    def test_resolution_presets_valid(self, app_module):
        """Verify all presets have matching resolution/step counts."""
        for name, preset in app_module.RESOLUTION_PRESETS.items():
            resolutions = preset["resolutions"].split()
            steps = preset["steps"].split()
            assert len(resolutions) == len(steps), (
                f"Preset '{name}': {len(resolutions)} resolutions vs {len(steps)} steps"
            )
