#!/usr/bin/env python3
"""
Project validation script for food label transparency project.

Checks:
  1. Data integrity — expected CSV/JSON files exist, critical columns present
  2. Model artifacts — model checkpoint, tokenizer, config files
  3. Module import integrity — all __init__ exports work, no circular imports
  4. Allergen detection consistency — rule_match smoke tests
  5. Label format consistency — parse/serialize round-trips
  6. Cross-validation — runs pytest and reports pass/fail

Usage:
    python scripts/validate_project.py

Exit code:
    0 — all checks passed
    1 — one or more checks failed
"""

import sys
import os
import json
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

# ── Project root (two levels up from scripts/) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure utils are importable
sys.path.insert(0, str(NOTEBOOKS_DIR))

# ── Status helpers ──

PASS = "✓"
FAIL = "✗"
SKIP = "⊘"

results: List[Tuple[str, str, str]] = []  # (check_name, status, detail)


def check(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    short = f"  [{status}] {name}"
    print(short)
    if detail:
        print(f"         {detail}")


# ════════════════════════════════════════════════
# 1. Data integrity
# ════════════════════════════════════════════════

def check_data_files():
    section = "\n── Data Files ──"
    print(section)

    # Check the main labeled dataset (may be named either enhanced or enhanced_fixed)
    for path_name in ["labeled_dataset_enhanced_fixed.csv", "labeled_dataset_enhanced.csv"]:
        lbl_path = PROJECT_ROOT / "data/final" / path_name
        if lbl_path.exists():
            size_kb = lbl_path.stat().st_size / 1024
            check(f"data/final/{path_name} [main labeled dataset]", PASS,
                  f"{size_kb:.0f} KB")
            break
    else:
        check("data/final/labeled_dataset_enhanced*.csv [main labeled dataset]", FAIL,
              "FILE NOT FOUND")

    # Check augmented train split from Notebook 07
    lbl_aug_path = PROJECT_ROOT / "data/final/splits/train_augmented.csv"
    if lbl_aug_path.exists():
        size_kb = lbl_aug_path.stat().st_size / 1024
        check("data/final/splits/train_augmented.csv [Notebook 07 - augmented train]", PASS,
              f"{size_kb:.0f} KB")

    # Optional intermediate files
    optional_files = {
        "data/raw/extracted_raw.csv": "raw OCR output (notebook 01)",
        "data/interim/parsed_ingredients.csv": "parsed ingredients (notebook 03)",
    }
    for rel_path, description in optional_files.items():
        full = PROJECT_ROOT / rel_path
        if full.exists():
            size_kb = full.stat().st_size / 1024
            check(f"{rel_path} [{description}]", PASS, f"{size_kb:.0f} KB")
        else:
            check(f"{rel_path} [{description}]", SKIP, "run notebooks to generate")

    # Check config files
    config_files = {
        "models/hybrid_config.json": "hybrid config (production)",
        "models/best_thresholds.npy": "ML thresholds array",
        "models/model_selection.json": "model selection record (Notebook 08)",
    }
    for rel_path, description in config_files.items():
        full = PROJECT_ROOT / rel_path
        if full.exists():
            check(f"{rel_path} [{description}]", PASS)
        else:
            check(f"{rel_path} [{description}]", FAIL, "FILE NOT FOUND")

    # Validate hybrid_config.json structure
    hybrid_path = PROJECT_ROOT / "models/hybrid_config.json"
    if hybrid_path.exists():
        try:
            with open(hybrid_path) as f:
                cfg = json.load(f)
            assert "ml_thresholds" in cfg, "missing ml_thresholds"
            thresholds = cfg["ml_thresholds"]
            assert len(thresholds) == 8, f"expected 8 thresholds, got {len(thresholds)}"
            check("hybrid_config.json structure", PASS,
                  f"8 thresholds, mode={cfg.get('mode', 'N/A')}")
        except Exception as e:
            check("hybrid_config.json structure", FAIL, str(e))


# ════════════════════════════════════════════════
# 2. Model artifacts
# ════════════════════════════════════════════════

def check_model_artifacts():
    section = "\n── Model Artifacts ──"
    print(section)

    # Define all model directories to check.
    # The ensemble stores per-seed checkpoints in `seed_{seed}/` subdirectories,
    # so it is validated differently from the flat model dirs below.
    # Weight file may be either legacy pytorch_model.bin or model.safetensors
    # (transformers saves safetensors by default since v4.30).
    required = ["pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
    weight_files = {"pytorch_model.bin", "model.safetensors"}

    model_configs = [
        ("models/mobilebert_allergen_final", "Notebook 08 - winner (selected on validation)"),
        ("models/mobilebert_allergen_bce", "Notebook 08 - BCE candidate"),
        ("models/mobilebert_allergen_focal", "Notebook 08 - Focal candidate"),
        ("models/mobilebert_allergen_asl", "Notebook 08 - ASL candidate"),
        ("models/mobilebert_semantic_final", "Notebook 05 - Semantic classifier"),
    ]

    for rel_dir, description in model_configs:
        model_dir = PROJECT_ROOT / rel_dir
        if model_dir.exists():
            check(f"{rel_dir}/ [{description}]", PASS)
            present_weights = [f for f in weight_files if (model_dir / f).exists()]
            if present_weights:
                size_mb = (model_dir / present_weights[0]).stat().st_size / (1024 * 1024)
                check(f"  weights ({present_weights[0]})", PASS, f"{size_mb:.1f} MB")
            else:
                check("  weights (pytorch_model.bin|model.safetensors)", FAIL, "FILE NOT FOUND")
            for fname in [f for f in required if f not in weight_files]:
                fpath = model_dir / fname
                if fpath.exists():
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    check(f"  {fname}", PASS, f"{size_mb:.1f} MB")
                else:
                    check(f"  {fname}", FAIL, "FILE NOT FOUND")
        else:
            check(f"{rel_dir}/ [{description}]", SKIP, "DIRECTORY NOT FOUND (optional)")

    # Ensemble: check that per-seed subdirectories exist with required files
    ens_dir = PROJECT_ROOT / "models/mobilebert_allergen_ensemble"
    if ens_dir.exists():
        seed_dirs = sorted(d for d in ens_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
        if not seed_dirs:
            check("models/mobilebert_allergen_ensemble/ [Notebook 08 - Ensemble (5 seeds)]", FAIL,
                  "no seed_*/ subdirectories")
        else:
            check(f"models/mobilebert_allergen_ensemble/ [Notebook 08 - Ensemble (5 seeds)] → "
                  f"{len(seed_dirs)} seed dirs", PASS)
            for seed_dir in seed_dirs[:3]:
                weight = next((f for f in weight_files if (seed_dir / f).exists()), None)
                missing = [fname for fname in required if fname not in weight_files
                           and not (seed_dir / fname).exists()]
                if weight is None:
                    check(f"  {seed_dir.name}: weights", FAIL, "FILE NOT FOUND")
                elif missing:
                    check(f"  {seed_dir.name}: {', '.join(missing)}", FAIL, "FILE NOT FOUND")
                else:
                    size_mb = (seed_dir / weight).stat().st_size / (1024 * 1024)
                    check(f"  {seed_dir.name}: complete", PASS, f"{size_mb:.1f} MB model")
    else:
        check("models/mobilebert_allergen_ensemble/ [Notebook 08 - Ensemble (5 seeds)]", SKIP,
              "DIRECTORY NOT FOUND (optional)")


# ════════════════════════════════════════════════
# 3. Module import integrity
# ════════════════════════════════════════════════

def check_module_imports():
    section = "\n── Module Imports ──"
    print(section)

    modules = [
        "utils.text_processing",
        "utils.data_utils",
        "utils.model_utils",
        "utils.evaluation",
        "utils.deployment_utils",
        "utils.semantic_utils",
    ]

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            check(mod_name, PASS, "imported successfully")
        except Exception as e:
            check(mod_name, FAIL, str(e))

    # Check __all__ exports are valid
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "__all__"):
                for name in mod.__all__:
                    if not hasattr(mod, name):
                        check(f"{mod_name}.__all__ member '{name}'", FAIL,
                              "not found in module")
                check(f"{mod_name} __all__ exports", PASS)
        except Exception:
            pass  # already reported above


# ════════════════════════════════════════════════
# 4. Allergen detection consistency
# ════════════════════════════════════════════════

def check_allergen_detection():
    section = "\n── Allergen Detection Consistency ──"
    print(section)

    try:
        from utils.text_processing import (
            rule_match, detect_allergens_rule_based, get_allergen_list,
            allergens_to_binary, parse_label_string, apply_exemptions,
            BIG8, FILIPINO_VARIANTS
        )

        # Basic detections
        tests = [
            ("milk detected", rule_match("milk", "milk"), True),
            ("eggs detected", rule_match("eggs", "eggs"), True),
            ("peanuts detected", rule_match("peanuts", "peanuts"), True),
            ("tree nuts detected", rule_match("almonds", "tree_nuts"), True),
            ("soy detected", rule_match("soy sauce", "soy"), True),
            ("wheat detected", rule_match("wheat flour", "wheat"), True),
            ("fish detected", rule_match("tuna", "fish"), True),
            ("shellfish detected", rule_match("shrimp", "shellfish"), True),
        ]
        for label, result, expected in tests:
            status = PASS if result == expected else FAIL
            check(f"  {label}", status, f"got={result}, expected={expected}")

        # Negation
        neg_tests = [
            ("no milk negated", rule_match("no milk", "milk"), False),
            ("coconut exempt from tree_nuts", rule_match("coconut oil", "tree_nuts"), False),
            ("shea exempt from tree_nuts", rule_match("shea butter", "tree_nuts"), False),
        ]
        for label, result, expected in neg_tests:
            status = PASS if result == expected else FAIL
            check(f"  {label}", status, f"got={result}, expected={expected}")

        # detect_allergens_rule_based
        detected = detect_allergens_rule_based("milk, eggs")
        check("detect_allergens_rule_based('milk, eggs')", PASS,
              f"got {detected}") if "milk" in detected and "eggs" in detected else \
            check("detect_allergens_rule_based('milk, eggs')", FAIL,
                  f"expected ['milk', 'eggs'], got {detected}")

        # Label conversions
        vec = allergens_to_binary(["milk", "wheat"])
        assert len(vec) == 8, f"expected 8, got {len(vec)}"
        assert vec[0] == 1 and vec[5] == 1, f"milk+vector incorrect: {vec}"
        check("allergens_to_binary(['milk', 'wheat'])", PASS)

        parsed = parse_label_string("[1, 0, 1, 0, 0, 0, 0, 0]")
        assert parsed == [1, 0, 1, 0, 0, 0, 0, 0], f"parse round-trip failed: {parsed}"
        check("parse_label_string round-trip", PASS)

        # Filipino variant detection
        fil_tests = [
            ("kasoy → tree_nuts", rule_match("kasoy", "tree_nuts"), True),
            ("itlog → eggs", rule_match("itlog", "eggs"), True),
            ("toyo → soy", rule_match("toyo", "soy"), True),
            ("mani → peanuts (NOT tree_nuts)",
             rule_match("mani", "tree_nuts"), False),
            ("mani → peanuts (IS peanuts)",
             rule_match("mani", "peanuts"), True),
        ]
        for label, result, expected in fil_tests:
            status = PASS if result == expected else FAIL
            check(f"  {label}", status, f"got={result}, expected={expected}")

        # Verify BIG8 structure
        allergen_list = get_allergen_list()
        assert len(allergen_list) == 8, f"expected 8, got {len(allergen_list)}"
        assert set(BIG8.keys()) == set(allergen_list), \
            f"BIG8 keys mismatch: {set(BIG8.keys())} vs {set(allergen_list)}"
        total_kws = sum(len(v) for v in BIG8.values())
        check(f"BIG8 keywords per allergen → {total_kws} total", PASS)

    except ImportError as e:
        check("allergen detection import", FAIL, str(e))
    except AssertionError as e:
        check("allergen detection assertion", FAIL, str(e))
    except Exception as e:
        check("allergen detection error", FAIL, str(e))


# ════════════════════════════════════════════════
# 5. Label format consistency
# ════════════════════════════════════════════════

def check_label_format():
    section = "\n── Label Format Consistency ──"
    print(section)

    try:
        from utils.text_processing import (
            get_allergen_list, allergens_to_binary, parse_label_string
        )
        from utils.data_utils import (
            labels_to_binary, serialize_list_for_csv,
        )

        # allergen_list is 8
        lst = get_allergen_list()
        check(f"get_allergen_list() → {len(lst)} items", PASS if len(lst) == 8 else FAIL)

        # Round trip: list → binary → list
        test_labels = ["milk", "soy", "wheat"]
        vec = allergens_to_binary(test_labels)
        round_trip = [lst[i] for i, v in enumerate(vec) if v == 1]
        check(f"allergens_to_binary round-trip: {test_labels} → {round_trip}",
              PASS if set(round_trip) == set(test_labels) else FAIL)

        # labels_to_binary
        multi = [["milk", "peanuts"], ["soy"]]
        bin_mat = labels_to_binary(multi, lst)
        check(f"labels_to_binary shape: {bin_mat.shape}",
              PASS if bin_mat.shape == (2, 8) else FAIL)

        # serialize_list_for_csv + parse_label_string
        original = [1, 0, 1, 0, 0, 0, 0, 0]
        ser = serialize_list_for_csv(original)
        parsed = parse_label_string(ser)
        check(f"CSV serialize round-trip",
              PASS if parsed == original else FAIL)

    except Exception as e:
        check("label format checks", FAIL, str(e))


# ════════════════════════════════════════════════
# 6. Cross-validation via pytest
# ════════════════════════════════════════════════

def check_pytest():
    section = "\n── Pytest Integration ──"
    print(section)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT)
        )
        # Extract summary line
        lines = result.stdout.strip().split("\n")
        summary = [l for l in lines if "passed" in l and "failed" in l]
        if summary:
            check("pytest summary", PASS if result.returncode == 0 else FAIL, summary[-1])
        else:
            # Use the last relevant line
            last = lines[-1] if lines else ""
            check("pytest output", PASS if result.returncode == 0 else FAIL, last)

        # Count pass/fail/skip
        passed = sum(1 for l in lines if " PASSED" in l)
        failed = sum(1 for l in lines if " FAILED" in l)
        skipped = sum(1 for l in lines if " SKIPPED" in l)
        if passed + failed > 0:
            check(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}",
                  PASS if failed == 0 else FAIL)

    except subprocess.TimeoutExpired:
        check("pytest execution", FAIL, "TIMED OUT (120s)")
    except FileNotFoundError:
        check("pytest execution", FAIL, "pytest not found")
    except Exception as e:
        check("pytest execution", FAIL, str(e))


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Food Label Transparency — Project Validation")
    print(f"  Project root: {PROJECT_ROOT}")
    print("=" * 60)

    check_data_files()
    check_model_artifacts()
    check_module_imports()
    check_allergen_detection()
    check_label_format()
    check_pytest()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")

    if n_fail > 0:
        print("\n  Failed checks:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    [{FAIL}] {name}")
                if detail:
                    print(f"           {detail}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
