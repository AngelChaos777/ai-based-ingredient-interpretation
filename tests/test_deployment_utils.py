"""
Tests for deployment_utils.py — ONNX export, model validation, latency benchmarking.

Validates function signatures and return types. Heavy operations (ONNX export,
model inference) are tested with mock objects or graceful-failure paths since
they require either a trained model checkpoint or a real ONNX Runtime session.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import pytest
import numpy as np
from unittest.mock import MagicMock


# ──────────────────────────────────────────────
# Module availability
# ──────────────────────────────────────────────

class TestModuleImports:
    """Verify deployment_utils imports cleanly and exposes expected functions."""

    def test_module_imports(self):
        from utils import deployment_utils
        assert hasattr(deployment_utils, "pytorch_to_onnx")
        assert hasattr(deployment_utils, "validate_onnx_output")
        assert hasattr(deployment_utils, "benchmark_latency")
        assert hasattr(deployment_utils, "benchmark_onnx_latency")
        assert hasattr(deployment_utils, "measure_model_size")
        assert hasattr(deployment_utils, "check_deployment_requirements")

    def test_expected_exports(self):
        from utils.deployment_utils import (
            pytorch_to_onnx,
            validate_onnx_output,
            benchmark_latency,
            benchmark_onnx_latency,
            measure_model_size,
            check_deployment_requirements,
        )
        assert callable(pytorch_to_onnx)
        assert callable(validate_onnx_output)
        assert callable(benchmark_latency)
        assert callable(benchmark_onnx_latency)
        assert callable(measure_model_size)
        assert callable(check_deployment_requirements)


# ──────────────────────────────────────────────
# measure_model_size
# ──────────────────────────────────────────────

class TestMeasureModelSize:
    """Model size measurement."""

    def test_measure_size(self, temporary_dir):
        """Should return a dict with size_bytes, size_kb, size_mb."""
        from utils.deployment_utils import measure_model_size
        dummy_path = os.path.join(str(temporary_dir), "dummy.onnx")
        with open(dummy_path, "wb") as f:
            f.write(b"\x00" * 1024)  # 1 KiB

        result = measure_model_size(dummy_path)
        assert isinstance(result, dict)
        assert "size_bytes" in result, f"Expected 'size_bytes' key, got {list(result.keys())}"
        assert "size_kb" in result
        assert "size_mb" in result
        assert result["size_bytes"] == 1024

    def test_file_not_found(self):
        """Missing file returns dict with error key (does not raise)."""
        from utils.deployment_utils import measure_model_size
        result = measure_model_size("/nonexistent/model.onnx")
        assert isinstance(result, dict)
        assert result.get("error") == "File not found", \
            f"Expected error message, got {result}"
        assert result.get("size_bytes") == 0


# ──────────────────────────────────────────────
# check_deployment_requirements
# ──────────────────────────────────────────────

class TestDeploymentRequirements:
    """Deployment requirements check."""

    def test_check_returns_dict(self):
        """Should return a dict with dep status."""
        from utils.deployment_utils import check_deployment_requirements
        result = check_deployment_requirements()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_lists_required_packages(self):
        """Should list required packages and their availability."""
        from utils.deployment_utils import check_deployment_requirements
        result = check_deployment_requirements()
        # At minimum should have onnxruntime or onnx
        package_names = [k.lower() for k in result.keys()]
        assert any("onnx" in name for name in package_names), \
            f"Should check ONNX packages, got: {list(result.keys())}"


# ──────────────────────────────────────────────
# pytorch_to_onnx
# ──────────────────────────────────────────────

class TestPytorchToOnnx:
    """Validate ONNX export function handles various inputs correctly."""

    def test_exports_using_dummy_input_arrays(self, tmp_path, monkeypatch):
        from utils.deployment_utils import pytorch_to_onnx
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        captured = {}

        def fake_export(model, args, f, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            Path(f).write_bytes(b"onnx")

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        input_ids = np.ones((1, 4), dtype=np.int64)
        attention_mask = np.ones((1, 4), dtype=np.int64)
        output_path = tmp_path / "export"

        result = pytorch_to_onnx(
            TinyModel(),
            type("Tokenizer", (), {"name_or_path": "test-tokenizer"})(),
            dummy_input_ids=input_ids,
            dummy_attention_mask=attention_mask,
            output_path=str(output_path),
            model_name="test_model",
        )

        assert result == str(output_path / "test_model.onnx")
        assert Path(result).read_bytes() == b"onnx"
        assert len(captured["args"]) == 2
        assert tuple(captured["args"][0].shape) == (1, 4)
        assert tuple(captured["args"][1].shape) == (1, 4)

    def test_export_failure_removes_partial_output(self, tmp_path, monkeypatch):
        from utils.deployment_utils import pytorch_to_onnx
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        def failing_export(model, args, f, **kwargs):
            Path(f).write_bytes(b"partial")
            raise RuntimeError("export failed")

        monkeypatch.setattr(torch.onnx, "export", failing_export)
        output_path = tmp_path / "export"
        with pytest.raises(RuntimeError, match="export failed"):
            pytorch_to_onnx(
                TinyModel(),
                type("Tokenizer", (), {"name_or_path": "test-tokenizer"})(),
                dummy_input_ids=np.ones((1, 4), dtype=np.int64),
                dummy_attention_mask=np.ones((1, 4), dtype=np.int64),
                output_path=str(output_path),
                model_name="test_model",
            )

        assert not (output_path / "test_model.onnx").exists()


# ──────────────────────────────────────────────
# validate_onnx_output
# ──────────────────────────────────────────────

class TestValidateOnnxOutput:
    """Validate ONNX output function handles missing file gracefully."""

    def test_needs_real_onnx_file(self):
        """validate_onnx_output requires a real ONNX file — should raise w/ nonexistent path."""
        from utils.deployment_utils import validate_onnx_output
        mock_model = MagicMock()
        # Need a mock that has .parameters() (returns iterator) and .eval()
        mock_model.eval.return_value = None
        mock_model.parameters.return_value = iter([])  # empty iterator
        mock_tokenizer = MagicMock()
        with pytest.raises(Exception):
            validate_onnx_output(
                onnx_path="/nonexistent/model.onnx",
                pytorch_model=mock_model,
                tokenizer=mock_tokenizer,
                test_text="milk, sugar"
            )


# ──────────────────────────────────────────────
# benchmark_latency
# ──────────────────────────────────────────────

class TestBenchmarkLatency:
    """Latency benchmarking for model inference."""

    def test_benchmark_requires_valid_model(self):
        """benchmark_latency should fail gracefully with wrong model type."""
        from utils.deployment_utils import benchmark_latency
        with pytest.raises((TypeError, AttributeError, Exception)):
            benchmark_latency(
                model=None,
                tokenizer=None,
                test_texts=["test"],
                max_length=64,
                n_runs=1,
                warmup_runs=0
            )

    def test_benchmark_with_mock(self):
        """Should return timing dict for a mock model + tokenizer."""
        from utils.deployment_utils import benchmark_latency
        import torch

        # Mock model that takes input_ids + attention_mask
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = None

        def fake_forward(input_ids, attention_mask=None):
            batch = input_ids.shape[0]
            return torch.randn(batch, 8)

        mock_model.side_effect = fake_forward

        # Mock tokenizer that returns torch tensors
        mock_tokenizer = MagicMock()

        def fake_tokenize(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            batch = len(texts)
            seq_len = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.ones((batch, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch, seq_len), dtype=torch.long),
            }

        mock_tokenizer.side_effect = fake_tokenize

        result = benchmark_latency(
            model=mock_model,
            tokenizer=mock_tokenizer,
            test_texts=["test input"],
            max_length=64,
            n_runs=3,
            warmup_runs=1,
            device="cpu",
        )
        assert isinstance(result, dict)
        # The function returns mean_ms, median_ms, std_ms, min_ms, max_ms, p95_ms, p99_ms, runs
        assert "mean_ms" in result, f"Missing 'mean_ms', got keys: {list(result.keys())}"
        assert "median_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "runs" in result
        assert result["runs"] == 3

    def test_benchmark_with_large_input(self):
        """benchmark_latency with more texts should complete."""
        from utils.deployment_utils import benchmark_latency
        import torch

        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = None

        def fake_forward(input_ids, attention_mask=None):
            batch = input_ids.shape[0]
            return torch.randn(batch, 8)

        mock_model.side_effect = fake_forward
        mock_tokenizer = MagicMock()

        def fake_tokenize(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            batch = len(texts)
            seq_len = kwargs.get("max_length", 64)
            return {
                "input_ids": torch.ones((batch, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch, seq_len), dtype=torch.long),
            }

        mock_tokenizer.side_effect = fake_tokenize

        result = benchmark_latency(
            model=mock_model,
            tokenizer=mock_tokenizer,
            test_texts=["test"] * 16,
            max_length=64,
            n_runs=2,
            warmup_runs=0,
            device="cpu",
        )
        assert result["mean_ms"] >= 0
        assert result["runs"] == 2


# ──────────────────────────────────────────────
# benchmark_onnx_latency
# ──────────────────────────────────────────────

class TestBenchmarkOnnxLatency:
    """ONNX-specific latency benchmarking."""

    def test_requires_valid_onnx_path(self):
        """benchmark_onnx_latency should fail with nonexistent ONNX file."""
        from utils.deployment_utils import benchmark_onnx_latency
        with pytest.raises(Exception):
            benchmark_onnx_latency(
                onnx_path="/nonexistent/model.onnx",
                input_ids=np.zeros((1, 64), dtype=np.int64),
                attention_mask=np.ones((1, 64), dtype=np.int64),
                n_runs=1,
                warmup_runs=0
            )
