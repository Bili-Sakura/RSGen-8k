"""Tests for the XLRS-Bench data loading module."""

import pytest
from unittest.mock import patch, MagicMock

# Patch sys.modules so "datasets" resolves to a mock with load_dataset.
# The project has a local datasets/ package that shadows HuggingFace's.
def _mock_datasets_module(load_return_value):
    mock_mod = MagicMock()
    mock_mod.load_dataset = MagicMock(return_value=load_return_value)
    return mock_mod


class TestLoadXlrsBenchPrompts:
    """Tests for the prompt loading function."""

    def _make_mock_dataset(self, captions):
        """Create a mock dataset with the given captions."""
        ds = MagicMock()
        ds.column_names = ["caption_en"]
        ds.__getitem__ = lambda self_ds, key: captions if key == "caption_en" else None
        return ds

    def test_loads_all_prompts(self):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = ["caption A", "caption B", "caption C"]
        with patch.dict("sys.modules", {"datasets": _mock_datasets_module(self._make_mock_dataset(captions))}):
            result = load_xlrs_bench_prompts()
        assert len(result) == 3
        assert result == captions

    def test_limits_num_prompts(self):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = [f"caption {i}" for i in range(20)]
        with patch.dict("sys.modules", {"datasets": _mock_datasets_module(self._make_mock_dataset(captions))}):
            result = load_xlrs_bench_prompts(num_prompts=5, seed=42)
        assert len(result) == 5

    def test_deterministic_with_seed(self):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = [f"caption {i}" for i in range(50)]
        mock_mod = _mock_datasets_module(self._make_mock_dataset(captions))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            result1 = load_xlrs_bench_prompts(num_prompts=10, seed=42)
            result2 = load_xlrs_bench_prompts(num_prompts=10, seed=42)
        assert result1 == result2

    def test_fallback_column_name(self):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        ds = MagicMock()
        ds.column_names = ["text"]
        ds.__getitem__ = lambda self_ds, key: ["prompt A", "prompt B"] if key == "text" else None

        with patch.dict("sys.modules", {"datasets": _mock_datasets_module(ds)}):
            result = load_xlrs_bench_prompts()
        assert len(result) == 2

    def test_import_error_without_datasets(self):
        """Verify a clear error when the datasets library is missing."""
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                load_xlrs_bench_prompts()
