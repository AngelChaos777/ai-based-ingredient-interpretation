#!/usr/bin/env python3
"""
End-to-end pipeline for the food label transparency project.

Orchestrates notebook execution in dependency order via
``jupyter nbconvert --to notebook --execute``.  Each notebook is run
from the notebooks/ directory so relative paths (``../data/``,
``../models/``) resolve correctly.  Executed copies are written to an
output directory and the original notebooks are left untouched.

Usage
-----
Run the full pipeline::

    python scripts/run_pipeline.py

Resume from a specific notebook (skipping prior ones)::

    python scripts/run_pipeline.py --skip-until 04_semantic_labeling

Validate that expected output files exist (no re-execution)::

    python scripts/run_pipeline.py --validate

Show which notebooks would run without executing::

    python scripts/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output" / "executed_notebooks"

# Ordered notebooks — this is the canonical execution order.
# Notebooks earlier in the list must complete before later ones start.
# Only exclude notebooks that are purely exploratory or not part of the
# production pipeline.
NOTEBOOK_ORDER: List[str] = [
    "01_extraction",
    "02_cleaning",
    "03_ingredient_parsing",
    "04_semantic_labeling",
    "05_semantic_model",
    "06_allergen_labeling",
    "07_allergen_training",
    "08_hybrid_evaluation",
    "09_model_export",
    "10_mobile_benchmark",
    "11_dataset_augmentation",
]

# Key outputs to validate after each notebook (relative to PROJECT_ROOT).
# Each entry maps "notebook_name" -> list of expected output paths (globs
# or exact paths).  Only the most important outputs are checked.
EXPECTED_OUTPUTS: dict = {
    "02_cleaning": [
        "data/cleaned_dataset.csv",
    ],
    "04_semantic_labeling": [
        "data/labeled_dataset.csv",
    ],
    "05_semantic_model": [
        "models/mobilebert_semantic_final/",
    ],
    "06_allergen_labeling": [
        "data/labeled_dataset_enhanced.csv",
    ],
    "07_allergen_training": [
        "models/mobilebert_allergen_final/",
        "models/hybrid_config.json",
    ],
    "09_model_export": [
        "models/mobilebert_allergen_final/model.onnx",
    ],
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def notebook_name_to_path(name: str) -> Path:
    """Convert ``"04_semantic_labeling"`` to an absolute ``.ipynb`` path."""
    return NOTEBOOKS_DIR / f"{name}.ipynb"


def parse_skip_until(value: Optional[str]) -> int:
    """Resolve ``--skip-until`` to an index into NOTEBOOK_ORDER.

    The value can be an integer index (0-based), a notebook short name
    (``"04_semantic_labeling"``), or ``None``.
    """
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        pass
    for i, name in enumerate(NOTEBOOK_ORDER):
        if name == value:
            return i
    print(f"error: --skip-until '{value}' not found in notebook order")
    print(f"  Available notebooks: {', '.join(NOTEBOOK_ORDER)}")
    sys.exit(1)


def run_notebook(name: str, output_dir: Path, timeout: int = -1) -> Tuple[bool, float]:
    """Execute a single notebook with ``nbconvert`` and return (success, elapsed_seconds)."""
    src = notebook_name_to_path(name)
    dst = output_dir / f"{name}.ipynb"

    if not src.exists():
        print(f"  [SKIP] {name}: source notebook not found at {src}")
        return False, 0.0

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--output", str(dst),
        str(src),
    ]

    start = time.monotonic()
    result = subprocess.run(cmd, cwd=str(NOTEBOOKS_DIR), capture_output=True, text=True)
    elapsed = time.monotonic() - start

    if result.returncode == 0:
        print(f"  [OK]   {name} ({elapsed:.1f}s)")
        return True, elapsed
    else:
        print(f"  [FAIL] {name} ({elapsed:.1f}s)")
        if result.stderr:
            lines = result.stderr.strip().splitlines()
            stderr_snippet = "\n".join(lines[-15:])
            for line in stderr_snippet.splitlines():
                print(f"         | {line}")
        return False, elapsed


def validate_outputs(name: str) -> bool:
    """Check that key output files exist for a given notebook.

    Returns True if all expected outputs for this notebook exist.
    """
    expected = EXPECTED_OUTPUTS.get(name, [])
    if not expected:
        return True  # nothing to validate

    all_ok = True
    for rel_path in expected:
        abs_path = PROJECT_ROOT / rel_path
        exists = abs_path.exists()
        if not exists:
            print(f"  [MISS] {name}: expected output not found at {rel_path}")
            all_ok = False
    return all_ok


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end notebook pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-until",
        default=None,
        help="Notebook name or index to resume from (skips earlier notebooks).",
        metavar="NAME_OR_IDX",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate that expected output files exist (skip execution).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    parser.add_argument(
        "--timeout",
        default=-1,
        type=int,
        help="Per-notebook timeout in seconds (-1 = no limit, default).",
        metavar="SECONDS",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory for executed notebooks (default: output/executed_notebooks/).",
        metavar="DIR",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    skip_index = parse_skip_until(args.skip_until)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # ── Dry run ──
    if args.dry_run:
        print("Dry run — notebooks that would execute:\n")
        for i, name in enumerate(NOTEBOOK_ORDER):
            if i < skip_index:
                print(f"  [{i}] {name}       (skipped)")
            else:
                src = notebook_name_to_path(name)
                status = "not found" if not src.exists() else "will execute"
                print(f"  [{i}] {name}       ({status})")
        print()
        return 0

    # ── Validate only ──
    if args.validate:
        print("Validating expected outputs...\n")
        all_good = True
        for name in NOTEBOOK_ORDER:
            ok = validate_outputs(name)
            tag = "OK" if ok else "MISSING"
            print(f"  [{tag}] {name}")
            if not ok:
                all_good = False
        print()
        if all_good:
            print("All expected outputs present.")
            return 0
        else:
            print("Some expected outputs are missing (run the pipeline to generate them).")
            return 1

    # ── Execute pipeline ──
    output_dir.mkdir(parents=True, exist_ok=True)

    notebooks_to_run = NOTEBOOK_ORDER[skip_index:]
    if skip_index > 0:
        skipped = NOTEBOOK_ORDER[:skip_index]
        print(f"Pipeline — skipping {len(skipped)} notebook(s): {', '.join(skipped)}")

    print(f"Pipeline — {len(notebooks_to_run)} notebook(s) to execute")
    print(f"Output   — {output_dir}\n")

    results: List[Tuple[str, bool, float]] = []
    start_global = time.monotonic()

    for name in notebooks_to_run:
        success, elapsed = run_notebook(name, output_dir, timeout=args.timeout)
        results.append((name, success, elapsed))

        if not success:
            print(f"\nPipeline aborted at {name} due to execution failure.\n")
            break

        # Validate key outputs after successful execution
        if success:
            validate_outputs(name)

    duration = time.monotonic() - start_global

    # ── Summary ──
    print(f"\n{'─' * 50}")
    print(f"  Pipeline Summary")
    print(f"{'─' * 50}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    print(f"  Total:  {total} notebook(s)")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Time:   {duration:.1f}s total")
    print()

    if failed > 0:
        print("Failed notebooks:")
        for name, ok, _ in results:
            if not ok:
                print(f"  - {name}")
        print()
        return 1

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
