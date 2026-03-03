"""Tests for the XLRS-Bench data loading module."""

import pytest
from unittest.mock import patch, MagicMock


class TestLoadXlrsBenchPrompts:
    """Tests for the prompt loading function."""

    def _make_mock_dataset(self, captions):
        """Create a mock dataset with the given captions."""
        ds = MagicMock()
        ds.column_names = ["caption_en"]
        ds.__getitem__ = lambda self_ds, key: captions if key == "caption_en" else None
        return ds

    @patch("datasets.load_dataset")
    def test_loads_all_prompts(self, mock_load):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = ["caption A", "caption B", "caption C"]
        mock_load.return_value = self._make_mock_dataset(captions)

        result = load_xlrs_bench_prompts()
        assert len(result) == 3
        assert result == captions

    @patch("datasets.load_dataset")
    def test_limits_num_prompts(self, mock_load):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = [f"caption {i}" for i in range(20)]
        mock_load.return_value = self._make_mock_dataset(captions)

        result = load_xlrs_bench_prompts(num_prompts=5, seed=42)
        assert len(result) == 5

    @patch("datasets.load_dataset")
    def test_deterministic_with_seed(self, mock_load):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        captions = [f"caption {i}" for i in range(50)]
        mock_load.return_value = self._make_mock_dataset(captions)

        result1 = load_xlrs_bench_prompts(num_prompts=10, seed=42)
        result2 = load_xlrs_bench_prompts(num_prompts=10, seed=42)
        assert result1 == result2

    @patch("datasets.load_dataset")
    def test_fallback_column_name(self, mock_load):
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        ds = MagicMock()
        ds.column_names = ["text"]
        ds.__getitem__ = lambda self_ds, key: ["prompt A", "prompt B"] if key == "text" else None

        mock_load.return_value = ds
        result = load_xlrs_bench_prompts()
        assert len(result) == 2

    def test_import_error_without_datasets(self):
        """Verify a clear error when the datasets library is missing."""
        from rsgen8k.data.xlrs_bench import load_xlrs_bench_prompts

        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                load_xlrs_bench_prompts()
