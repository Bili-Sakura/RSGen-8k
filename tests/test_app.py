"""Tests for the Gradio web demo."""

import sys
import os

import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestGradioDemo:
    """Tests for the Gradio web demo."""

    def test_import_app(self):
        """Verify app.py imports without error."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
        )
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        assert hasattr(app, "build_demo")
        assert hasattr(app, "RESOLUTION_PRESETS")

    def test_build_demo_returns_blocks(self):
        """Verify build_demo() creates a valid Gradio Blocks."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
        )
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        demo = app.build_demo()
        import gradio as gr
        assert isinstance(demo, gr.Blocks)

    def test_resolution_presets_valid(self):
        """Verify all presets have matching resolution/step counts."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
        )
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        for name, preset in app.RESOLUTION_PRESETS.items():
            resolutions = preset["resolutions"].split()
            steps = preset["steps"].split()
            assert len(resolutions) == len(steps), (
                f"Preset '{name}': {len(resolutions)} resolutions vs {len(steps)} steps"
            )
