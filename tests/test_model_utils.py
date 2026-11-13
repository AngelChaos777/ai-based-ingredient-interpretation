"""
Unit tests for model_utils.py — class weights, thresholds, and predictions.

Run with: python -m pytest tests/test_model_utils.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_utils_dir = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(_utils_dir))

from utils import model_utils as mu


# ──────────────────────────────────────────────
# compute_class_weights
# ──────────────────────────────────────────────

class TestComputeClassWeights:
    def test_balanced_classes(self):
        """Balanced classes should produce weights close to 1."""
        labels = np.array([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ])
        weights = mu.compute_class_weights(labels)
        assert len(weights) == 2
        # Both classes have same frequency, weights should be equal
        assert abs(float(weights[0]) - float(weights[1])) < 0.01

    def test_imbalanced_classes(self):
        """Imbalanced classes should give higher weight to rare class."""
        labels = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],  # rare
        ])
        weights = mu.compute_class_weights(labels)
        # Rare class (index 1) should have higher weight
        assert float(weights[1]) > float(weights[0])

    def test_no_positive_samples(self):
        """Classes with no positives should receive more weight than the common class."""
        labels = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ])
        weights = mu.compute_class_weights(labels, max_weight=10.0)
        assert torch.isfinite(weights).all()
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]

    def test_all_positive_samples(self):
        """Class where every sample is positive should also be handled."""
        labels = np.array([
            [1, 0],
            [1, 1],  # all ones for class 1
            [1, 1],
            [1, 1],
        ])
        weights = mu.compute_class_weights(labels)
        assert len(weights) == 2
        assert torch.isfinite(weights).all()
        assert weights[0] > weights[1]

    def test_single_sample(self):
        """Single sample should not crash."""
        labels = np.array([[1, 0, 1]])
        weights = mu.compute_class_weights(labels)
        assert len(weights) == 3

    def test_empty_labels_raises(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError):
            mu.compute_class_weights(np.array([]))

    def test_nan_labels_raises(self):
        """NaN in labels should raise ValueError."""
        labels = np.array([[1, np.nan], [0, 1]])
        with pytest.raises(ValueError):
            mu.compute_class_weights(labels)

    def test_inf_labels_raises(self):
        """Inf in labels should raise ValueError."""
        labels = np.array([[1, np.inf], [0, 1]])
        with pytest.raises(ValueError):
            mu.compute_class_weights(labels)

    def test_max_weight_caps_extreme(self):
        """max_weight parameter should cap output."""
        labels = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        weights = mu.compute_class_weights(labels, max_weight=5.0)
        # With max_weight=5, all weights should be <= 5
        assert all(float(w) <= 5.01 for w in weights)

    def test_torch_tensor_input(self):
        """Should accept torch tensors."""
        labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        weights = mu.compute_class_weights(labels)
        assert len(weights) == 2
        assert isinstance(weights, torch.Tensor)

    def test_output_is_torch_tensor(self):
        """Return type should be torch.Tensor."""
        labels = np.array([[1, 0], [0, 1]])
        result = mu.compute_class_weights(labels)
        assert isinstance(result, torch.Tensor)

    def test_multilabel_eight_classes(self):
        """Realistic: 8 allergen classes with typical imbalance."""
        labels = np.zeros((100, 8))
        # Simulate real distribution
        freqs = [0.47, 0.09, 0.08, 0.015, 0.26, 0.34, 0.07, 0.03]
        for i, freq in enumerate(freqs):
            labels[:int(freq * 100), i] = 1
        weights = mu.compute_class_weights(labels)
        # tree_nuts (idx 3) should have highest weight due to rarity
        tree_nuts_weight = float(weights[3])
        milk_weight = float(weights[0])
        assert tree_nuts_weight > milk_weight


# ──────────────────────────────────────────────
# apply_thresholds
# ──────────────────────────────────────────────

class TestApplyThresholds:
    def test_basic_threshold(self):
        probs = np.array([[0.8, 0.3], [0.2, 0.9]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1  # 0.8 >= 0.5
        assert result[0, 1] == 0  # 0.3 < 0.5
        assert result[1, 0] == 0  # 0.2 < 0.5
        assert result[1, 1] == 1  # 0.9 >= 0.5

    def test_variable_thresholds(self):
        probs = np.array([[0.5, 0.5, 0.5]])
        thresholds = np.array([0.3, 0.5, 0.7])
        result = mu.apply_thresholds(probs, thresholds)
        assert result[0, 0] == 1  # 0.5 >= 0.3
        assert result[0, 1] == 1  # 0.5 >= 0.5
        assert result[0, 2] == 0  # 0.5 < 0.7

    def test_all_zero_threshold(self):
        probs = np.array([[0.1, 0.2]])
        thresholds = np.array([0.0, 0.0])
        result = mu.apply_thresholds(probs, thresholds)
        assert result[0, 0] == 1
        assert result[0, 1] == 1

    def test_all_one_threshold(self):
        probs = np.array([[0.1, 0.2]])
        thresholds = np.array([1.0, 1.0])
        result = mu.apply_thresholds(probs, thresholds)
        assert result[0, 0] == 0
        assert result[0, 1] == 0

    def test_edge_values(self):
        probs = np.array([[0.0, 1.0]])
        thresholds = np.array([0.5, 0.5])
        result = mu.apply_thresholds(probs, thresholds)
        assert result[0, 0] == 0
        assert result[0, 1] == 1


# ──────────────────────────────────────────────
# find_best_thresholds
# ──────────────────────────────────────────────

class TestFindBestThresholds:
    def test_perfect_separation(self):
        """Selected thresholds should preserve clearly separated classes."""
        labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        probs = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
        thresholds = mu.find_best_thresholds(probs, labels, step=0.1)
        assert thresholds.shape == (2,)
        assert np.array_equal(mu.apply_thresholds(probs, thresholds), labels)

    def test_single_class(self):
        """Single class should not crash."""
        labels = np.array([[1], [0], [1]])
        probs = np.array([[0.9], [0.1], [0.8]])
        thresholds = mu.find_best_thresholds(probs, labels)
        assert len(thresholds) == 1


# ──────────────────────────────────────────────
# predict_ml — basic validation
# ──────────────────────────────────────────────

class TestPredictMLShape:
    def test_importable_functions(self):
        """Verify that core functions are importable."""
        assert hasattr(mu, 'predict_ml')
        assert hasattr(mu, 'apply_thresholds')
        assert hasattr(mu, 'find_best_thresholds')
        assert hasattr(mu, 'compute_class_weights')


class TestHybridPredict:
    @pytest.mark.parametrize("mode", ["hard_override", "rule_priority"])
    def test_rule_positive_overrides_low_ml_probability(self, mode):
        from types import SimpleNamespace

        class LowConfidenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask=None):
                logits = torch.full((input_ids.shape[0], 8), -8.0)
                return SimpleNamespace(logits=logits)

        def tokenize(texts, **kwargs):
            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

        model = LowConfidenceModel()
        milk_index = mu.get_allergen_list().index("milk")
        preds, probs = mu.hybrid_predict(
            ["milk powder"],
            model,
            tokenize,
            torch.device("cpu"),
            thresholds=np.full(8, 0.5),
            rule_conf_threshold=1.0,
            mode=mode,
        )

        assert probs[0, milk_index] < 1.0
        assert preds[0, milk_index] == 1
