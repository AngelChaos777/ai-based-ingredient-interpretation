"""
Integration tests for the food label transparency project.

Validates end-to-end workflows across modules:
- Data loading → label parsing → split creation → augmentation
- Rule-based allergen detection → label format consistency
- Cross-module function coordination (text_processing ↔ data_utils ↔ model_utils)
- Pipeline stage contracts (each stage produces expected output format)

These tests are lightweight — they verify module integration with small
synthetic data rather than loading the full 1057-sample dataset or a
trained model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import pytest
import numpy as np
import pandas as pd
from typing import List


# ──────────────────────────────────────────────
# Integration test categories
# ──────────────────────────────────────────────

ALLERGEN_LIST = ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]

SAMPLE_DATA = [
    {
        "text": "milk, sugar, wheat flour",
        "allergens": ["milk", "wheat"],
        "label_vector": [1, 0, 0, 0, 0, 1, 0, 0],
    },
    {
        "text": "almonds, cashews, salt",
        "allergens": ["tree_nuts"],
        "label_vector": [0, 0, 0, 1, 0, 0, 0, 0],
    },
    {
        "text": "sugar, salt, water, citric acid",
        "allergens": [],
        "label_vector": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        "text": "eggs, milk, peanut butter, soy sauce",
        "allergens": ["eggs", "milk", "peanuts", "soy"],
        "label_vector": [1, 1, 1, 0, 1, 0, 0, 0],
    },
    {
        "text": "no milk, contains soy, wheat free",
        "allergens": ["soy"],
        "label_vector": [0, 0, 0, 0, 1, 0, 0, 0],
    },
]


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small synthetic DataFrame for integration tests."""
    rows = []
    for item in SAMPLE_DATA:
        rows.append({
            "ingredients_cleaned": item["text"],
            "detected_allergens": item["allergens"],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════
# Class 1: Data loading → label format consistency
# ════════════════════════════════════════════════

class TestDataLoadingAndLabelFormat:
    """Labels flow correctly from text to binary format."""

    def test_allergens_to_binary(self):
        """Converting allergen lists to binary vectors must produce correct length."""
        from utils.text_processing import allergens_to_binary, get_allergen_list

        for item in SAMPLE_DATA:
            vector = allergens_to_binary(item["allergens"])
            assert len(vector) == len(ALLERGEN_LIST), \
                f"Expected {len(ALLERGEN_LIST)} elements, got {len(vector)}"
            assert vector == item["label_vector"], \
                f"Expected {item['label_vector']}, got {vector} for {item['allergens']}"

    def test_parse_label_string_consistency(self):
        """parse_label_string produces consistent output."""
        from utils.text_processing import parse_label_string, get_allergen_list

        label_str = "[1, 0, 1, 0, 0, 0, 0, 0]"
        result = parse_label_string(label_str)
        assert len(result) == len(ALLERGEN_LIST)
        assert result == [1, 0, 1, 0, 0, 0, 0, 0]

    def test_get_allergen_list_stable(self):
        """get_allergen_list must return stable ordering."""
        from utils.text_processing import get_allergen_list
        lst = get_allergen_list()
        assert lst == ALLERGEN_LIST, \
            f"Expected {ALLERGEN_LIST}, got {lst}"

    def test_serialize_list_for_csv(self):
        """List serialization must round-trip through parsing."""
        from utils.data_utils import serialize_list_for_csv
        from utils.text_processing import parse_label_string

        vector = [1, 0, 1, 0, 0, 0, 0, 0]
        serialized = serialize_list_for_csv(vector)
        assert isinstance(serialized, str)
        # The serialized string must parse back correctly
        reparsed = parse_label_string(serialized)
        assert reparsed == vector


# ════════════════════════════════════════════════
# Class 2: Rule-based detection → label vector pipeline
# ════════════════════════════════════════════════

class TestRuleDetectionPipeline:
    """Text → detect_allergens_rule_based → allergens_to_binary."""

    def test_full_pipeline(self):
        """End-to-end: text → rule detection → binary vector."""
        from utils.text_processing import detect_allergens_rule_based, allergens_to_binary

        for item in SAMPLE_DATA:
            detected = detect_allergens_rule_based(item["text"])
            vector = allergens_to_binary(detected)

            assert len(vector) == len(ALLERGEN_LIST)
            assert vector == item["label_vector"], \
                f"'{item['text']}': expected {item['label_vector']}, got {vector}"

    def test_empty_text_returns_no_allergens(self):
        """Empty or whitespace text must produce empty label vector."""
        from utils.text_processing import detect_allergens_rule_based, allergens_to_binary

        for text in ["", "   ", "\n\t"]:
            detected = detect_allergens_rule_based(text)
            assert detected == [], \
                f"Empty text '{repr(text)}' should give [], got {detected}"
            vector = allergens_to_binary(detected)
            assert sum(vector) == 0, \
                f"Empty text should produce zero vector, got {vector}"

    def test_all_allergens_detected(self):
        """Verify all 8 allergens can be detected in a single mixed text."""
        from utils.text_processing import detect_allergens_rule_based

        # Each allergen keyword clearly present
        text = "milk, eggs, peanuts, almonds, soy, wheat, tuna, shrimp"
        detected = detect_allergens_rule_based(text)
        expected = {"milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"}
        detected_set = set(detected)
        missing = expected - detected_set
        assert not missing, f"Missing allergens: {missing}"


# ════════════════════════════════════════════════
# Class 3: augmentation → label consistency
# ════════════════════════════════════════════════

class TestAugmentationPipeline:
    """Augmentation must preserve label consistency."""

    def test_augment_dataframe_maintains_structure(self, sample_df):
        """augment_dataframe produces rows with correct columns."""
        from utils.data_utils import augment_dataframe
        from utils.text_processing import get_allergen_list, allergens_to_binary

        df_copy = sample_df.copy()
        df_copy["labels"] = df_copy["detected_allergens"].apply(allergens_to_binary)
        df_copy["text"] = df_copy["ingredients_cleaned"]

        augmented = augment_dataframe(df_copy, num_augmented=1)
        assert len(augmented) > 0, "Augmentation should produce at least one row"
        expected_cols = {"text", "labels", "ingredients_cleaned", "detected_allergens"}
        assert expected_cols.issubset(augmented.columns), \
            f"Missing columns: {expected_cols - set(augmented.columns)}"

    def test_augment_targeted_preserves_dtype(self, sample_df):
        """augment_targeted returns DataFrame with consistent dtypes."""
        from utils.data_utils import augment_targeted
        from utils.text_processing import get_allergen_list, allergens_to_binary

        df_copy = sample_df.copy()
        df_copy["labels"] = df_copy["detected_allergens"].apply(allergens_to_binary)
        df_copy["text"] = df_copy["ingredients_cleaned"]

        result = augment_targeted(
            df_copy,
            target_allergens={"tree_nuts": 2},
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(df_copy), \
            "Result should include at least the original rows"


# ════════════════════════════════════════════════
# Class 4: Stratified split consistency
# ════════════════════════════════════════════════

class TestSplitPipeline:
    """Stratified splitting preserves label distributions."""

    def test_create_stratified_splits(self, sample_df):
        """Splits must preserve total count and have correct dimensions."""
        from utils.data_utils import create_stratified_splits
        from utils.text_processing import allergens_to_binary

        X = sample_df["ingredients_cleaned"].values
        df_copy = sample_df.copy()
        df_copy["labels"] = df_copy["detected_allergens"].apply(allergens_to_binary)
        y = np.array(df_copy["labels"].tolist())

        train_x, val_x, test_x, train_y, val_y, test_y = create_stratified_splits(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )
        assert len(train_x) + len(val_x) + len(test_x) == len(X)
        assert train_y.shape[1] == len(ALLERGEN_LIST)
        assert val_y.shape[1] == len(ALLERGEN_LIST)
        assert test_y.shape[1] == len(ALLERGEN_LIST)

    def test_split_no_data(self):
        """Empty input to create_stratified_splits should raise."""
        from utils.data_utils import create_stratified_splits
        with pytest.raises((ValueError, Exception)):
            create_stratified_splits(
                np.array([]), np.array([]).reshape(0, 8),
                train_size=0.7, val_size=0.15, test_size=0.15
            )


# ════════════════════════════════════════════════
# Class 5: Cross-module label consistency
# ════════════════════════════════════════════════

class TestCrossModuleConsistency:
    """Functions across different modules must agree on label format."""

    def test_detect_and_convert_match(self):
        """detect_allergens_rule_based → allergens_to_binary must be
        consistent with direct rule_match per allergen."""
        from utils.text_processing import (
            detect_allergens_rule_based, allergens_to_binary, rule_match, get_allergen_list
        )

        text = "milk, wheat, and soy"
        detected = detect_allergens_rule_based(text)
        vector = allergens_to_binary(detected)

        allergen_list = get_allergen_list()
        for i, allergen in enumerate(allergen_list):
            expected = 1 if allergen in detected else 0
            assert vector[i] == expected, \
                f"allergen_to_binary mismatch for '{allergen}': vector[{i}]={vector[i]} != {expected}"
            # Also confirm rule_match agrees
            rule_result = 1 if rule_match(text, allergen) else 0
            assert vector[i] == rule_result, \
                f"rule_match mismatch for '{allergen}': vector[{i}]={vector[i]} != rule_match={rule_result}"

    def test_label_vector_length_stable(self):
        """All modules must consistently use 8-label format."""
        from utils.text_processing import get_allergen_list, allergens_to_binary
        from utils.data_utils import labels_to_binary

        allergen_list = get_allergen_list()
        assert len(allergen_list) == 8

        # labels_to_binary agrees
        test_labels = [["milk", "peanuts"], ["soy"]]
        binary = labels_to_binary(test_labels, allergen_list)
        assert binary.shape == (2, 8)

    def test_validate_label_consistency(self, sample_df):
        """validate_label_consistency must pass on clean data."""
        from utils.data_utils import validate_label_consistency
        from utils.text_processing import allergens_to_binary

        df_copy = sample_df.copy()
        df_copy["labels"] = df_copy["detected_allergens"].apply(allergens_to_binary)

        # Should run without raising
        result = validate_label_consistency(df_copy, "labels")
        assert isinstance(result, list)


# ════════════════════════════════════════════════
# Class 6: Data directory structure
# ════════════════════════════════════════════════

class TestDataDirectories:
    """get_data_directories must return valid paths."""

    def test_get_data_directories_returns_paths(self):
        """Should return a dict with all expected keys."""
        from utils.data_utils import get_data_directories
        dirs = get_data_directories()
        expected = {"base", "raw", "interim", "processed", "final", "models", "configs", "outputs", "notebooks"}
        assert expected.issubset(dirs.keys()), \
            f"Missing keys: {expected - set(dirs.keys())}"

    def test_directories_are_absolute(self):
        """Returned paths should be absolute."""
        from utils.data_utils import get_data_directories
        dirs = get_data_directories()
        for key, path in dirs.items():
            assert os.path.isabs(path), f"'{key}' path '{path}' is not absolute"
