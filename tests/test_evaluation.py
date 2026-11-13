"""
Unit tests for evaluation.py — metrics, thresholds, error analysis, and OFF comparison.

Run with: python -m pytest tests/test_evaluation.py -v
"""
import sys
from pathlib import Path

# Add notebooks/ to the import path so the utils package is importable
# with its full package name, enabling relative imports (e.g. from .model_utils).
_notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(_notebooks_dir))

import numpy as np
import pandas as pd
import io
import pytest


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def import_module():
    """Import the module once per session, skip if unavailable."""
    global ev
    try:
        import utils.evaluation as ev
    except ImportError as e:
        pytest.skip(f"evaluation module not importable: {e}")


@pytest.fixture
def sample_binary():
    """Small binary arrays for metric tests: 4 samples, 3 classes."""
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=int)
    y_pred = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
    ], dtype=int)
    return y_true, y_pred


@pytest.fixture
def class_names():
    return ["milk", "eggs", "peanuts"]


# ──────────────────────────────────────────────
# jaccard
# ──────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        assert ev.jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert ev.jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        result = ev.jaccard({"a", "b", "c"}, {"b", "c", "d"})
        # intersection = {b,c} size 2, union = {a,b,c,d} size 4 => 0.5
        assert result == 0.5

    def test_one_empty(self):
        assert ev.jaccard(set(), {"a", "b"}) == 0.0

    def test_both_empty(self):
        assert ev.jaccard(set(), set()) == 1.0

    def test_superset(self):
        result = ev.jaccard({"a", "b", "c"}, {"a", "b"})
        assert result == 2 / 3

    def test_subset(self):
        result = ev.jaccard({"a", "b"}, {"a", "b", "c"})
        assert result == 2 / 3

    def test_single_identical(self):
        assert ev.jaccard({"milk"}, {"milk"}) == 1.0

    def test_single_different(self):
        assert ev.jaccard({"milk"}, {"eggs"}) == 0.0


# ──────────────────────────────────────────────
# apply_thresholds (test model_utils directly — evaluation.py delegates
# via relative import, which fails outside the notebooks/utils/ package)
# ──────────────────────────────────────────────

class TestApplyThresholds:
    def test_default_threshold(self):
        import utils.model_utils as mu
        probs = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.3]])
        thresholds = np.array([0.5, 0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_high_thresholds(self):
        import utils.model_utils as mu
        probs = np.array([[0.8, 0.2], [0.4, 0.6]])
        thresholds = np.array([0.9, 0.9])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[0, 0], [0, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_low_thresholds(self):
        import utils.model_utils as mu
        probs = np.array([[0.8, 0.2], [0.4, 0.6]])
        thresholds = np.array([0.1, 0.1])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 1], [1, 1]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_thresholds(self):
        import utils.model_utils as mu
        probs = np.array([[0.7, 0.3, 0.9], [0.2, 0.8, 0.1]])
        thresholds = np.array([0.5, 0.6, 0.8])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_all_below(self):
        import utils.model_utils as mu
        probs = np.array([[0.1, 0.2], [0.3, 0.4]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[0, 0], [0, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_all_above(self):
        import utils.model_utils as mu
        probs = np.array([[0.9, 0.8], [0.7, 0.6]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 1], [1, 1]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_single_sample(self):
        import utils.model_utils as mu
        probs = np.array([[0.95, 0.05]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_boundary_equal(self):
        """Exactly at threshold should be 1 (>=)."""
        import utils.model_utils as mu
        probs = np.array([[0.5, 0.3]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        expected = np.array([[1, 0]], dtype=int)
        np.testing.assert_array_equal(result, expected)


# ──────────────────────────────────────────────
# find_best_thresholds (test model_utils directly — same relative-import
# reason as apply_thresholds above)
# ──────────────────────────────────────────────

class TestFindBestThresholds:
    def test_perfect_separation(self):
        import utils.model_utils as mu
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([[1, 0], [0, 1]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.1)
        # Both classes: best threshold is just above the lower value (0.1)
        # t=0.01 gives preds=[1,1] (f1=2/3), t=0.11 gives perfect separation (f1=1)
        assert thresholds[0] == pytest.approx(0.11, abs=0.001)
        assert thresholds[1] == pytest.approx(0.11, abs=0.001)

    def test_partial_separation(self):
        import utils.model_utils as mu
        probs = np.array([[0.8, 0.1], [0.7, 0.2], [0.9, 0.3]])
        labels = np.array([[1, 0], [1, 0], [1, 0]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.2)
        assert thresholds[0] < 0.5
        assert thresholds[1] == 0.5

    def test_step_size_respected(self):
        import utils.model_utils as mu
        probs = np.array([[0.8, 0.2], [0.1, 0.9]])
        labels = np.array([[1, 0], [0, 1]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.5)
        assert isinstance(thresholds, np.ndarray)

    def test_single_class(self):
        import utils.model_utils as mu
        probs = np.array([[0.9], [0.1], [0.8]])
        labels = np.array([[1], [0], [1]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.2)
        assert len(thresholds) == 1

    def test_all_positive(self):
        import utils.model_utils as mu
        probs = np.array([[0.9], [0.8], [0.95]])
        labels = np.array([[1], [1], [1]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.2)
        assert thresholds[0] == 0.01

    def test_all_negative(self):
        import utils.model_utils as mu
        probs = np.array([[0.1], [0.2], [0.3]])
        labels = np.array([[0], [0], [0]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.2)
        assert thresholds[0] == 0.5


# ──────────────────────────────────────────────
# binary_to_label_list
# ──────────────────────────────────────────────

class TestBinaryToLabelList:
    def test_basic(self):
        binary = np.array([[1, 0, 1], [0, 1, 0]])
        names = ["milk", "eggs", "peanuts"]
        result = ev.binary_to_label_list(binary, names)
        assert result == [["milk", "peanuts"], ["eggs"]]

    def test_all_zero(self):
        binary = np.array([[0, 0, 0]])
        names = ["milk", "eggs", "peanuts"]
        result = ev.binary_to_label_list(binary, names)
        assert result == [[]]

    def test_all_one(self):
        binary = np.array([[1, 1]])
        names = ["a", "b"]
        result = ev.binary_to_label_list(binary, names)
        assert result == [["a", "b"]]

    def test_multiple_samples(self):
        binary = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        names = ["x", "y"]
        result = ev.binary_to_label_list(binary, names)
        assert result == [["x"], ["y"], ["x", "y"], []]

    def test_single_class(self):
        binary = np.array([[0], [1]])
        names = ["only"]
        result = ev.binary_to_label_list(binary, names)
        assert result == [[], ["only"]]


# ──────────────────────────────────────────────
# compute_per_class_metrics
# ──────────────────────────────────────────────

class TestComputePerClassMetrics:
    def test_perfect_prediction(self, sample_binary, class_names):
        y_true, _ = sample_binary
        # Use ground truth as predictions -> perfect
        metrics = ev.compute_per_class_metrics(y_true, y_true, class_names)
        for c in class_names:
            assert metrics[c]["precision"] == 1.0
            assert metrics[c]["recall"] == 1.0
            assert metrics[c]["f1"] == 1.0

    def test_all_wrong(self, sample_binary, class_names):
        y_true, _ = sample_binary
        # Predict opposite
        y_wrong = 1 - y_true
        metrics = ev.compute_per_class_metrics(y_true, y_wrong, class_names)
        # All metrics should be 0 (no correct predictions)
        for c in class_names:
            assert metrics[c]["precision"] == 0.0
            assert metrics[c]["recall"] == 0.0
            assert metrics[c]["f1"] == 0.0

    def test_partial(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        metrics = ev.compute_per_class_metrics(y_true, y_pred, class_names)
        # Class 0 (milk): TP=2, FP=0, FN=0 -> 1/1/1
        # Class 1 (eggs): TP=1, FP=0, FN=1 -> 1/0.5/0.666...
        # Class 2 (peanuts): TP=2, FP=1, FN=0 -> 0.666.../1/0.8
        assert metrics["milk"]["f1"] == 1.0
        assert metrics["eggs"]["recall"] == 0.5
        assert metrics["peanuts"]["precision"] == 2 / 3

    def test_return_type(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        metrics = ev.compute_per_class_metrics(y_true, y_pred, class_names)
        assert isinstance(metrics, dict)
        assert set(metrics.keys()) == set(class_names)
        for c in class_names:
            assert set(metrics[c].keys()) == {"precision", "recall", "f1"}
            for v in metrics[c].values():
                assert isinstance(v, float)


# ──────────────────────────────────────────────
# print_classification_report (smoke test)
# ──────────────────────────────────────────────

class TestPrintClassificationReport:
    def test_runs_with_prefix(self, sample_binary, class_names):
        """Should print without error, with a prefix."""
        y_true, y_pred = sample_binary
        # Capture stdout
        out = io.StringIO()
        sys.stdout = out
        ev.print_classification_report(y_true, y_pred, class_names, prefix="Test Set")
        sys.stdout = sys.__stdout__
        output = out.getvalue()
        assert "Test Set" in output
        assert "Macro F1" in output
        assert "Micro F1" in output

    def test_runs_without_prefix(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        out = io.StringIO()
        sys.stdout = out
        ev.print_classification_report(y_true, y_pred, class_names)
        sys.stdout = sys.__stdout__
        output = out.getvalue()
        assert "Macro F1" in output

    def test_custom_digits(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        out = io.StringIO()
        sys.stdout = out
        ev.print_classification_report(y_true, y_pred, class_names, digits=6)
        sys.stdout = sys.__stdout__
        output = out.getvalue()
        # With 6 digits, we should see more decimal places (e.g., "1.000000")
        assert "Macro F1" in output


# ──────────────────────────────────────────────
# print_per_class_metrics (smoke test)
# ──────────────────────────────────────────────

class TestPrintPerClassMetrics:
    def test_runs(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        out = io.StringIO()
        sys.stdout = out
        ev.print_per_class_metrics(y_true, y_pred, class_names)
        sys.stdout = sys.__stdout__
        output = out.getvalue()
        for c in class_names:
            assert c in output
        assert "Prec:" in output
        assert "F1:" in output

    def test_custom_prefix(self, sample_binary, class_names):
        y_true, y_pred = sample_binary
        out = io.StringIO()
        sys.stdout = out
        ev.print_per_class_metrics(y_true, y_pred, class_names, prefix="Custom Header")
        sys.stdout = sys.__stdout__
        assert "Custom Header" in out.getvalue()


# ──────────────────────────────────────────────
# compute_multilabel_confusion_matrix
# ──────────────────────────────────────────────

class TestComputeMultilabelConfusionMatrix:
    def test_shape(self, sample_binary):
        y_true, y_pred = sample_binary
        cm = ev.compute_multilabel_confusion_matrix(y_true, y_pred)
        assert cm.shape[0] == 3  # 3 classes
        assert cm.shape[1] == 2  # 2x2 per class
        assert cm.shape[2] == 2

    def test_values(self, sample_binary):
        y_true, y_pred = sample_binary
        cm = ev.compute_multilabel_confusion_matrix(y_true, y_pred)
        # Class 0 (milk): TP=2, FN=0, FP=0, TN=2
        assert cm[0, 0, 0] == 2  # TN
        assert cm[0, 0, 1] == 0  # FP
        assert cm[0, 1, 0] == 0  # FN
        assert cm[0, 1, 1] == 2  # TP


# ──────────────────────────────────────────────
# error_analysis
# ──────────────────────────────────────────────

class TestErrorAnalysis:
    def test_no_errors(self):
        y = np.array([[1, 0], [0, 1]])
        texts = ["text a", "text b"]
        names = ["milk", "eggs"]
        errors = ev.error_analysis(texts, y, y, names, max_errors=5)
        assert errors == []

    def test_some_errors(self):
        texts = ["has milk", "has eggs", "has peanuts"]
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # first and last wrong
        names = ["milk", "eggs", "peanuts"]
        errors = ev.error_analysis(texts, y_true, y_pred, names, max_errors=2)
        assert len(errors) == 2
        assert errors == [0, 2]

    def test_max_errors_cap(self):
        texts = ["a", "b", "c", "d"]
        y_true = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        y_pred = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])  # all wrong
        names = ["milk", "eggs"]
        errors = ev.error_analysis(texts, y_true, y_pred, names, max_errors=2)
        assert len(errors) == 2

    def test_empty_dataset(self):
        errors = ev.error_analysis([], np.empty((0, 2)), np.empty((0, 2)), ["a", "b"])
        assert errors == []


# ──────────────────────────────────────────────
# evaluate_rule_vs_official (integration-style)
# ──────────────────────────────────────────────

class TestEvaluateRuleVsOfficial:
    def test_perfect_agreement(self):
        df = pd.DataFrame({
            "detected_allergens": [["milk"], ["eggs"], []],
            "official_allergens_mapped": [["milk"], ["eggs"], []],
        })
        result = ev.evaluate_rule_vs_official(df, allergen_list=["milk", "eggs"])
        assert result["exact_agreement"] == 1.0
        assert result["avg_jaccard"] == 1.0
        assert result["n_samples"] == 3

    def test_no_agreement(self):
        df = pd.DataFrame({
            "detected_allergens": [["milk", "eggs"]],
            "official_allergens_mapped": [["peanuts"]],
        })
        result = ev.evaluate_rule_vs_official(df, allergen_list=["milk", "eggs", "peanuts"])
        assert result["exact_agreement"] == 0.0
        assert result["avg_jaccard"] == 0.0

    def test_subset_agreement(self):
        df = pd.DataFrame({
            "detected_allergens": [["milk", "eggs"]],
            "official_allergens_mapped": [["milk", "eggs", "peanuts"]],
        })
        result = ev.evaluate_rule_vs_official(df, allergen_list=["milk", "eggs", "peanuts"])
        # detected ⊆ official
        assert result["subset_agreement"] == 1.0
        # detected ⊉ official
        assert result["superset_agreement"] == 0.0

    def test_allergen_list_inference(self):
        df = pd.DataFrame({
            "detected_allergens": [["milk"], ["eggs"]],
            "official_allergens_mapped": [["milk"], ["milk"]],
        })
        result = ev.evaluate_rule_vs_official(df)
        # Should infer allergen_list = ["eggs", "milk"] (sorted union)
        assert set(result["per_class"].keys()) == {"eggs", "milk"}
        assert result["n_samples"] == 2

    def test_mixed_partial(self):
        df = pd.DataFrame({
            "detected_allergens": [["milk"], ["milk"], []],
            "official_allergens_mapped": [["milk"], ["eggs"], []],
        })
        result = ev.evaluate_rule_vs_official(df, allergen_list=["milk", "eggs"])
        # Samples: exact match, no match, exact match (empty)
        assert result["exact_agreement"] == 2 / 3
        assert result["avg_jaccard"] > 0


# ──────────────────────────────────────────────
# Edge cases and robustness
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_jaccard_non_set_inputs(self):
        """jaccard should handle list inputs (sets cast)."""
        assert ev.jaccard(set(["a"]), set(["a"])) == 1.0

    def test_apply_thresholds_empty(self):
        import utils.model_utils as mu
        probs = np.empty((0, 3))
        thresholds = np.array([0.5, 0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        assert result.shape == (0, 3)

    def test_binary_to_label_list_empty_classes(self):
        binary = np.array([[0, 0, 0]])
        names = []
        result = ev.binary_to_label_list(binary, names)
        assert result == [[]]

    def test_find_best_thresholds_identical_probs(self):
        import utils.model_utils as mu
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([[1, 0], [1, 0]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.1)
        assert len(thresholds) == 2
        assert np.all(thresholds >= 0.01)
        assert np.all(thresholds <= 0.99)

    def test_error_analysis_all_wrong(self):
        texts = ["a", "b"]
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0, 1], [1, 0]])  # both wrong
        names = ["x", "y"]
        # Capture stdout to avoid print noise
        out = io.StringIO()
        sys.stdout = out
        errors = ev.error_analysis(texts, y_true, y_pred, names)
        sys.stdout = sys.__stdout__
        assert len(errors) == 2
        output = out.getvalue()
        assert "Total errors: 2" in output
