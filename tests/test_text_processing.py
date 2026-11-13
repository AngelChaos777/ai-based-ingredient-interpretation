"""
Unit tests for text_processing.py — rule matching, negation, exemptions, and context awareness.

Run with: python -m pytest tests/test_text_processing.py -v
"""
import re
import sys
import os
from pathlib import Path

# Add notebooks/utils to the import path
_utils_dir = Path(__file__).resolve().parents[1] / "notebooks" / "utils"
sys.path.insert(0, str(_utils_dir))

import pytest


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def import_module():
    """Import the module once per session, skip if unavailable."""
    global tp
    try:
        import text_processing as tp
    except ImportError as e:
        pytest.skip(f"text_processing module not importable: {e}")


# ──────────────────────────────────────────────
# rule_match — basic detection
# ──────────────────────────────────────────────

class TestRuleMatchBasic:
    def test_detects_milk(self):
        assert tp.rule_match("contains milk powder", "milk") is True

    def test_detects_eggs(self):
        assert tp.rule_match("contains egg whites", "eggs") is True

    def test_detects_peanuts(self):
        assert tp.rule_match("peanut butter", "peanuts") is True

    def test_detects_tree_nuts(self):
        assert tp.rule_match("almond flour", "tree_nuts") is True

    def test_detects_soy(self):
        assert tp.rule_match("soy sauce", "soy") is True

    def test_detects_wheat(self):
        assert tp.rule_match("whole wheat flour", "wheat") is True

    def test_detects_fish(self):
        assert tp.rule_match("tuna fish", "fish") is True

    def test_detects_shellfish(self):
        assert tp.rule_match("shrimp paste", "shellfish") is True

    def test_no_false_positive_on_empty(self):
        assert tp.rule_match("", "milk") is False

    def test_no_false_positive_on_unrelated(self):
        assert tp.rule_match("sugar, salt, water", "milk") is False

    def test_case_insensitive(self):
        assert tp.rule_match("MILK POWDER", "milk") is True
        assert tp.rule_match("Whole Wheat Flour", "wheat") is True

    def test_unknown_allergen_raises(self):
        with pytest.raises(ValueError):
            tp.rule_match("test", "fake_allergen")


# ──────────────────────────────────────────────
# rule_match — negation handling
# ──────────────────────────────────────────────

class TestRuleMatchNegation:
    def test_no_prefix(self):
        assert tp.rule_match("no milk", "milk") is False

    def test_free_suffix_hyphen(self):
        assert tp.rule_match("milk-free chocolate", "milk") is False

    def test_free_after_space(self):
        assert tp.rule_match("wheat free bread", "wheat") is False

    def test_does_not_contain(self):
        assert tp.rule_match("does not contain eggs", "eggs") is False

    def test_free_from(self):
        assert tp.rule_match("free from gluten", "wheat") is False

    def test_without_prefix(self):
        assert tp.rule_match("without peanuts", "peanuts") is False

    def test_hyphenated_compound_not_negated(self):
        """'milk-chocolate' should NOT be treated as negation (just a compound word)."""
        assert tp.rule_match("milk-chocolate", "milk") is True

    def test_peanut_butter_hyphen(self):
        """'peanut-butter' should NOT be negated."""
        assert tp.rule_match("peanut-butter", "peanuts") is True

    def test_low_in_prefix(self):
        assert tp.rule_match("low in sodium", "milk") is False  # unrelated term, but pattern triggers
        # More relevant:
        assert tp.rule_match("low in milk", "milk") is True  # Wait — "low in" IS a negation pattern

    def test_contains_no(self):
        assert tp.rule_match("contains no soy", "soy") is False

    def test_positive_mention_after_negated_mention(self):
        assert tp.rule_match("no milk, milk powder", "milk") is True


# ──────────────────────────────────────────────
# rule_match — context-aware disambiguation
# ──────────────────────────────────────────────

class TestRuleMatchContextAware:
    def test_coconut_cream_not_milk(self):
        """'coconut cream' should NOT trigger milk detection."""
        assert tp.rule_match("coconut cream", "milk") is False

    def test_almond_cream_not_milk(self):
        """'almond cream' should NOT trigger milk detection."""
        assert tp.rule_match("almond cream", "milk") is False

    def test_whipping_cream_is_milk(self):
        """'whipping cream' is dairy — should it be? Actually, whipping cream IS dairy."""
        assert tp.rule_match("whipping cream", "milk") is True

    def test_non_dairy_creamer_not_milk(self):
        """'non-dairy creamer' should NOT trigger milk."""
        assert tp.rule_match("non-dairy creamer", "milk") is False

    def test_rice_flour_not_wheat(self):
        """'rice flour' should NOT trigger wheat."""
        assert tp.rule_match("rice flour", "wheat") is False

    def test_wheat_flour_is_wheat(self):
        """'wheat flour' SHOULD trigger wheat."""
        assert tp.rule_match("wheat flour", "wheat") is True

    def test_corn_flour_not_wheat(self):
        """'corn flour' should NOT trigger wheat."""
        assert tp.rule_match("corn flour", "wheat") is False

    def test_soy_lecithin_is_soy(self):
        """'soy lecithin' SHOULD trigger soy."""
        assert tp.rule_match("soy lecithin", "soy") is True

    def test_standalone_lecithin_not_soy(self):
        """Standalone 'lecithin' (without 'soy' prefix) should NOT trigger soy.
        Since 'lecithin' is NOT in BIG8 directly — it only matches via context rules
        that require 'soy' prefix, so standalone lecithin should not match."""
        # 'lecithin' is not in the core BIG8 keywords for soy
        # The AMBIGUOUS_KEYWORDS rule requires 'soy', 'soya', 'soybean' prefix
        assert tp.rule_match("lecithin", "soy") is False

    def test_sunflower_lecithin_not_soy(self):
        """'sunflower lecithin' should NOT trigger soy."""
        assert tp.rule_match("sunflower lecithin", "soy") is False

    def test_coconut_cream_in_milk_context(self):
        """'milk, coconut cream' — only 'milk' matches, not 'cream' as milk."""
        result = tp.detect_allergens_rule_based("milk, coconut cream")
        assert "milk" in result

    def test_valid_cream_after_disallowed_cream(self):
        assert tp.rule_match("coconut cream, dairy cream", "milk") is True

    def test_mixed_ingredients_wheat(self):
        """'tapioca flour, potato starch' should NOT trigger wheat."""
        assert tp.rule_match("tapioca flour, potato starch", "wheat") is False


# ──────────────────────────────────────────────
# detect_allergens_rule_based
# ──────────────────────────────────────────────

class TestDetectAllergensRuleBased:
    def test_detects_multiple(self):
        result = tp.detect_allergens_rule_based("milk, eggs, wheat flour")
        assert "milk" in result
        assert "eggs" in result
        assert "wheat" in result

    def test_returns_empty_list(self):
        assert tp.detect_allergens_rule_based("sugar, salt, water") == []

    def test_handles_negation(self):
        result = tp.detect_allergens_rule_based("no milk, wheat free")
        assert "milk" not in result
        assert "wheat" not in result

    def test_handles_context(self):
        """'coconut cream' should not trigger any allergen."""
        result = tp.detect_allergens_rule_based("coconut cream, sugar")
        assert "milk" not in result


# ──────────────────────────────────────────────
# apply_exemptions
# ──────────────────────────────────────────────

class TestApplyExemptions:
    def test_soy_lecithin_exempted(self):
        """Standalone 'lecithin' or 'soy lecithin' in detected list should
        be removed if no other soy protein evidence is present."""
        tokens = ["soy", "lecithin"]
        # If soy is detected, and only 'soy lecithin' pattern matches
        result = tp.apply_exemptions(["soy"], tokens)
        assert "soy" not in result

    def test_soy_sauce_not_exempted(self):
        """'soy sauce' — the keep_if_also pattern should match, preserving soy."""
        tokens = ["soy", "sauce"]
        result = tp.apply_exemptions(["soy"], tokens)
        assert "soy" in result

    def test_tofu_not_exempted(self):
        """'tofu' in keep_if_also should preserve soy detection."""
        tokens = ["tofu"]
        result = tp.apply_exemptions(["soy"], tokens)
        assert "soy" in result

    def test_shea_butter_exempted_from_tree_nuts(self):
        tokens = ["shea", "butter"]
        result = tp.apply_exemptions(["tree_nuts"], tokens)
        assert "tree_nuts" not in result

    def test_coconut_oil_exempted_from_tree_nuts(self):
        tokens = ["coconut", "oil"]
        result = tp.apply_exemptions(["tree_nuts"], tokens)
        assert "tree_nuts" not in result

    def test_almond_kept_as_tree_nut(self):
        tokens = ["almond", "flour"]
        result = tp.apply_exemptions(["tree_nuts"], tokens)
        assert "tree_nuts" in result

    def test_lecithin_exempted_from_eggs(self):
        tokens = ["lecithin"]
        result = tp.apply_exemptions(["eggs"], tokens)
        assert "eggs" not in result

    def test_egg_yolk_kept_as_egg(self):
        tokens = ["egg", "yolk"]
        result = tp.apply_exemptions(["eggs"], tokens)
        assert "eggs" in result

    def test_multiple_allergens(self):
        tokens = ["soy", "lecithin", "almond", "flour"]
        result = tp.apply_exemptions(["soy", "tree_nuts"], tokens)
        assert "soy" not in result  # exempted (only lecithin)
        assert "tree_nuts" in result  # kept (almond)

    def test_coconut_oil_not_exempted_from_coconut(self):
        """When coconut is in the detected list (not as tree_nuts), it stays."""
        tokens = ["coconut", "oil"]
        result = tp.apply_exemptions(["coconut"], tokens)
        assert "coconut" in result


# ──────────────────────────────────────────────
# parse_label_string
# ──────────────────────────────────────────────

class TestParseLabelString:
    def test_none_returns_zeros(self):
        assert tp.parse_label_string(None) == [0] * 8

    def test_nan_returns_zeros(self):
        import numpy as np
        assert tp.parse_label_string(float('nan')) == [0] * 8

    def test_empty_list_returns_zeros(self):
        assert tp.parse_label_string("[]") == [0] * 8

    def test_binary_list(self):
        result = tp.parse_label_string("[1, 0, 1, 0, 0, 0, 0, 0]")
        assert result == [1, 0, 1, 0, 0, 0, 0, 0]

    def test_name_list(self):
        result = tp.parse_label_string("['milk', 'wheat']")
        assert len(result) == 8
        assert result[0] == 1  # milk
        assert result[5] == 1  # wheat
        assert sum(result) == 2


# ──────────────────────────────────────────────
# get_allergen_list
# ──────────────────────────────────────────────

class TestGetAllergenList:
    def test_returns_eight_allergens(self):
        result = tp.get_allergen_list()
        assert len(result) == 8

    def test_contains_expected(self):
        expected = ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]
        assert tp.get_allergen_list() == expected


# ──────────────────────────────────────────────
# clean_html and text preprocessing
# ──────────────────────────────────────────────

class TestTextCleaning:
    def test_clean_html_removes_tags(self):
        assert tp.clean_html("<div>milk</div>") == "milk"

    def test_clean_html_multiple_tags(self):
        assert tp.clean_html("<p>contains <b>milk</b></p>") == "contains milk"

    def test_preprocess_text(self):
        assert tp.preprocess_text("  MILK  ") == "milk"


# ──────────────────────────────────────────────
# test for Filipino variants
# ──────────────────────────────────────────────

class TestFilipinoVariants:
    def test_gatas_detects_milk(self):
        assert tp.rule_match("gatas na pulbos", "milk") is True

    def test_itlog_detects_eggs(self):
        assert tp.rule_match("itlog na pula", "eggs") is True

    def test_mani_detects_peanuts(self):
        assert tp.rule_match("mantika ng mani", "peanuts") is True

    def test_toyo_detects_soy(self):
        assert tp.rule_match("toyo", "soy") is True

    def test_trigo_detects_wheat(self):
        assert tp.rule_match("trigo", "wheat") is True

    def test_isda_detects_fish(self):
        assert tp.rule_match("isda", "fish") is True

    def test_hipon_detects_shellfish(self):
        assert tp.rule_match("hipon", "shellfish") is True
