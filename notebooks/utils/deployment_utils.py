"""
Deployment and model export utilities (Thesis Plan Deliverable 4)

This module provides functions for converting trained PyTorch models through
the deployment pipeline:

    PyTorch → ONNX → ONNX Runtime Mobile (Android deployment)

Key Features:
- PyTorch to ONNX export with dynamic axes
- ONNX output validation (compare against PyTorch baseline)
- ONNX inference latency benchmarking
- Model size measurement and deployment report generation

Android Deployment Target:
    ONNX Runtime Mobile — no TFLite conversion required.
    See: https://onnxruntime.ai/docs/install/#android-on-device

Usage:
    >>> from utils.deployment_utils import pytorch_to_onnx, validate_onnx_output
    >>> onnx_path = pytorch_to_onnx(model, tokenizer, dummy_text, "output_dir")
    >>> result = validate_onnx_output(onnx_path, model, tokenizer, "test text")
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ──────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────


def pytorch_to_onnx(
    model,
    tokenizer,
    dummy_input_ids: np.ndarray,
    dummy_attention_mask: np.ndarray,
    output_path: str,
    model_name: str = "model",
    dynamic_batch: bool = True,
    opset_version: int = 14,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model (in eval mode).
        tokenizer: Tokenizer for reference metadata.
        dummy_input_ids: Dummy input_ids tensor for tracing (shape: 1xseq_len).
        dummy_attention_mask: Dummy attention_mask tensor.
        output_path: Directory to write the ONNX file.
        model_name: Base filename (without extension).
        dynamic_batch: If True, sets batch dimension to dynamic.
        opset_version: ONNX opset version (default 14).

    Returns:
        Path to the exported ONNX file.
    """

    os.makedirs(output_path, exist_ok=True)
    onnx_path = os.path.join(output_path, f"{model_name}.onnx")

    # Save model config for reference
    config = {
        "model_name": model_name,
        "opset_version": opset_version,
        "dynamic_batch": dynamic_batch,
        "tokenizer": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else str(type(tokenizer).__name__),
        "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_path, f"{model_name}_export_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[deployment_utils] ONNX export config saved to {output_path}")
    print(f"[deployment_utils] Exporting model to ONNX...")
    print(f"[deployment_utils] Target: {onnx_path}")

    import torch

    # Move dummy inputs to the same device as the model
    device = next(model.parameters()).device
    dummy_input_ids_t = torch.tensor(dummy_input_ids, device=device)
    dummy_attention_mask_t = torch.tensor(dummy_attention_mask, device=device)

    model.eval()
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(dummy_input_ids_t, dummy_attention_mask_t),
                f=onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"},
                    "logits": {0: "batch_size"},
                } if dynamic_batch else None,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,  # Use legacy exporter for MobileBert compatibility
            )
            print(f"[deployment_utils] ONNX model exported to {onnx_path}")
    except Exception as e:
        print(f"[deployment_utils] ❌ ONNX export FAILED: {e}")
        print(f"[deployment_utils]    See traceback above for details.")
        # Remove partial output if file was created with error
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        raise

    return onnx_path


# ──────────────────────────────────────────────
# ONNX Validation & Android Deployment
# ──────────────────────────────────────────────


def validate_onnx_output(
    onnx_path: str,
    pytorch_model,
    tokenizer,
    test_text: str,
    max_length: int = 221,
    tolerance: float = 5e-3,
) -> Dict:
    """Run inference through both PyTorch and ONNX Runtime and compare outputs.

    Args:
        onnx_path: Path to the .onnx model file.
        pytorch_model: Original PyTorch model (eval mode).
        tokenizer: HuggingFace tokenizer.
        test_text: Input text for validation.
        max_length: Max tokenization length.
        tolerance: Max acceptable per-element absolute difference.

    Returns:
        Dictionary with max_diff, mean_diff, within_tolerance, and
        both logit arrays for inspection.
    """
    import onnxruntime as ort
    import torch

    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    # Tokenize
    encoded = tokenizer([test_text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # PyTorch inference
    with torch.no_grad():
        pt_out = pytorch_model(encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))
    pt_logits = pt_out.logits.cpu().numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy(),
    }
    ort_logits = ort_session.run(["logits"], ort_inputs)[0]

    # Compare
    diff = np.abs(pt_logits - ort_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"[deployment_utils] ONNX output validation")
    print(f"[deployment_utils]   Max diff vs PyTorch:  {max_diff:.6e}")
    print(f"[deployment_utils]   Mean diff vs PyTorch: {mean_diff:.6e}")
    print(f"[deployment_utils]   Within tolerance ({tolerance}): {'✅ YES' if max_diff < tolerance else '❌ NO'}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "within_tolerance": max_diff < tolerance,
        "tolerance": tolerance,
        "pt_logits": pt_logits,
        "ort_logits": ort_logits,
    }


# ──────────────────────────────────────────────
# Latency Benchmarking
# ──────────────────────────────────────────────


def measure_model_size(path: str) -> Dict[str, float]:
    """Measure model file size in various units.

    Args:
        path: Path to model file (onnx, tflite, .pt, etc.).

    Returns:
        Dictionary with size_bytes, size_kb, size_mb.
    """
    if not os.path.exists(path):
        return {"size_bytes": 0, "size_kb": 0, "size_mb": 0, "error": "File not found"}

    size_bytes = os.path.getsize(path)
    return {
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def benchmark_latency(
    model,
    tokenizer,
    test_texts: List[str],
    max_length: int = 221,
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark PyTorch model inference latency.

    NOTE: Only works with PyTorch models. For ONNX Runtime models,
    use benchmark_onnx_latency() instead.

    Args:
        model: PyTorch model in eval mode.
        tokenizer: HuggingFace tokenizer.
        test_texts: List of input texts for inference.
        max_length: Max tokenization length.
        n_runs: Number of benchmark runs.
        warmup_runs: Number of warm-up runs (not counted).
        device: Device for inference ("cpu" or "cuda").

    Returns:
        Dictionary with mean_ms, median_ms, std_ms, min_ms, max_ms, runs.
    """
    import torch

    model.eval()
    model.to(device)

    # Tokenize test inputs
    encoded = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    latencies = []

    with torch.no_grad():
        for i in range(warmup_runs + n_runs):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - start

            if i >= warmup_runs:
                latencies.append(elapsed * 1000)  # convert to ms

    return _summarize_latencies(latencies)


def benchmark_onnx_latency(
    onnx_path: str,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    n_runs: int = 100,
    warmup_runs: int = 10,
    providers: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Benchmark ONNX model inference latency using ONNX Runtime.

    Args:
        onnx_path: Path to .onnx model file.
        input_ids: Input token IDs (numpy array, shape: 1xseq_len).
        attention_mask: Attention mask (numpy array, shape: 1xseq_len).
        n_runs: Number of benchmark runs.
        warmup_runs: Number of warm-up runs.
        providers: ONNX Runtime providers (e.g., ['CPUExecutionProvider']).

    Returns:
        Dictionary with mean_ms, median_ms, std_ms, min_ms, max_ms.
    """
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    latencies: List[float] = []

    for i in range(warmup_runs + n_runs):
        start = time.perf_counter()
        session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
        elapsed = time.perf_counter() - start

        if i >= warmup_runs:
            latencies.append(elapsed * 1000)

    return _summarize_latencies(latencies)


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────


def _summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    """Compute summary statistics from a list of latency measurements.

    Args:
        latencies: List of latency values in milliseconds.

    Returns:
        Dictionary with aggregated statistics.
    """
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "runs": len(latencies),
    }


def get_model_size_report(model_path: str) -> str:
    """Get a human-readable model size report.

    Args:
        model_path: Path to the model file.

    Returns:
        Formatted string with size information.
    """
    sizes = measure_model_size(model_path)
    if "error" in sizes:
        return f"❌ {sizes['error']}: {model_path}"

    return (
        f"📦 Model: {os.path.basename(model_path)}\n"
        f"   Size: {sizes['size_mb']:.2f} MB ({sizes['size_kb']:.1f} KB, {sizes['size_bytes']:,} bytes)"
    )


def check_deployment_requirements() -> Dict[str, bool]:
    """Check which deployment packages are available in the environment.

    Note: The standalone ``tflite`` pip package is NOT required — TensorFlow
    2.x includes TFLite support via ``tf.lite``. The check below marks it as
    available whenever tensorflow is present.

    Returns:
        Dictionary mapping package names to availability (bool).
    """
    import_name_map = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("tensorflow", "tensorflow"),
        ("tflite", None),  # covered by tensorflow — see note above
    ]
    deps: Dict[str, bool] = {}
    for pkg_name, import_name in import_name_map:
        if import_name is None:
            # tflite is covered by tensorflow
            deps[pkg_name] = deps.get("tensorflow", False)
        else:
            try:
                __import__(import_name)
                deps[pkg_name] = True
            except ImportError:
                deps[pkg_name] = False
    return deps


def print_deployment_status() -> None:
    """Print a human-readable deployment readiness report."""
    deps = check_deployment_requirements()
    print("=" * 50)
    print("📦 Deployment Dependencies Status")
    print("=" * 50)
    for pkg, available in deps.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {pkg}")

    missing = [pkg for pkg, avail in deps.items() if not avail]
    if missing:
        print()
        print("Missing packages that would complete the pipeline:")
        for pkg in missing:
            if pkg == "onnx":
                print(f"  pip install onnx")
            elif pkg == "onnxruntime":
                print(f"  pip install onnxruntime")
            elif pkg == "tensorflow":
                print(f"  pip install tensorflow")
    print()
    print("Deployment target: ONNX Runtime Mobile (Android)")
    print("  No TFLite conversion required — onnx2tf removed.")
    print("=" * 50)
