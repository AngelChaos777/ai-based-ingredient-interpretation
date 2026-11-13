"""Tests for semantic utility functions — ingredient category mappings.

Tests cover:
1. New direct mappings (spices, emulsifiers, chemicals, vitamins, etc.)
2. Plural normalization (emulsifiers -> emulsifier, NO normalization for peanuts/eggs)
3. E-number pattern matching (E330 -> acidulant, E200 -> preservative)
4. FDC color matching (FD&C Red No. 40 -> colorant)
5. INS code matching (INS 330 -> acidulant)
6. Substring fallback for known patterns
7. Unknown ingredients return empty list
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from notebooks.utils.semantic_utils import (
    ingredient_to_categories,
    _normalize_plural,
    _match_enumber,
    _match_fdc_color,
    _match_ins_code,
    INGREDIENT_SEMANTIC_MAP,
    SEMANTIC_CATEGORIES,
)


# ──────────────────────────────────────────────
# Direct mapping tests
# ──────────────────────────────────────────────

class TestDirectMappings:
    """Verify new Phase 3 INGREDIENT_SEMANTIC_MAP entries resolve correctly."""

    def test_spices_functional_term(self):
        cats = ingredient_to_categories("spices")
        assert "spice" in cats, f"Expected 'spice' in {cats}"

    def test_emulsifier_functional_term(self):
        cats = ingredient_to_categories("emulsifiers")
        # Should resolve via plural normalization -> INGREDIENT_SEMANTIC_MAP
        assert "emulsifier" in cats, f"Expected 'emulsifier' in {cats}"
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"

    def test_stabilizer_functional_term(self):
        cats = ingredient_to_categories("stabilizers")
        assert "stabilizer" in cats, f"Expected 'stabilizer' in {cats}"

    def test_preservative_functional_term(self):
        cats = ingredient_to_categories("preservatives")
        assert "preservative" in cats, f"Expected 'preservative' in {cats}"

    def test_antioxidant_functional_term(self):
        cats = ingredient_to_categories("antioxidants")
        assert "antioxidant" in cats, f"Expected 'antioxidant' in {cats}"

    def test_flavor_enhancer_term(self):
        cats = ingredient_to_categories("flavor enhancers")
        # Should be in INGREDIENT_SEMANTIC_MAP directly (plural normalized)
        assert "flavor_enhancer" in cats, f"Expected 'flavor_enhancer' in {cats}"

    def test_silicon_dioxide(self):
        cats = ingredient_to_categories("silicon dioxide")
        assert "mineral" in cats, f"Expected 'mineral' in {cats}"

    def test_sucralose(self):
        cats = ingredient_to_categories("sucralose")
        assert "sweetener" in cats, f"Expected 'sweetener' in {cats}"
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"

    def test_tartrazine_resolves(self):
        cats = ingredient_to_categories("tartrazine")
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_sunset_yellow_resolves(self):
        cats = ingredient_to_categories("sunset yellow")
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_riboflavin(self):
        cats = ingredient_to_categories("riboflavin")
        assert "vitamin" in cats, f"Expected 'vitamin' in {cats}"

    def test_shrimp(self):
        cats = ingredient_to_categories("shrimp")
        assert "animal_derived" in cats, f"Expected 'animal_derived' in {cats}"

    def test_vanillin(self):
        cats = ingredient_to_categories("vanillin")
        assert "flavoring" in cats, f"Expected 'flavoring' in {cats}"

    def test_nata_de_coco(self):
        cats = ingredient_to_categories("nata de coco")
        assert "fermented" in cats, f"Expected 'fermented' in {cats}"

    def test_sodium_citrate(self):
        cats = ingredient_to_categories("sodium citrate")
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"
        assert "acidulant" in cats, f"Expected 'acidulant' in {cats}"


# ──────────────────────────────────────────────
# Plural normalization tests
# ──────────────────────────────────────────────

class TestPluralNormalization:
    """Verify _normalize_plural() behavior."""

    def test_emulsifiers_to_emulsifier(self):
        assert _normalize_plural("emulsifiers") == "emulsifier"

    def test_stabilizers_to_stabilizer(self):
        assert _normalize_plural("stabilizers") == "stabilizer"

    def test_thickeners_to_thickener(self):
        assert _normalize_plural("thickeners") == "thickener"

    def test_preservatives_to_preservative(self):
        assert _normalize_plural("preservatives") == "preservative"

    def test_acidulants_to_acidulant(self):
        assert _normalize_plural("acidulants") == "acidulant"

    def test_sweeteners_to_sweetener(self):
        assert _normalize_plural("sweeteners") == "sweetener"

    def test_enzymes_to_enzyme(self):
        assert _normalize_plural("enzymes") == "enzyme"

    def test_no_normalize_peanuts(self):
        assert _normalize_plural("peanuts") == "peanuts"

    def test_no_normalize_eggs(self):
        assert _normalize_plural("eggs") == "eggs"

    def test_no_normalize_spices(self):
        assert _normalize_plural("spices") == "spices"

    def test_no_normalize_oats(self):
        assert _normalize_plural("oats") == "oats"

    def test_no_normalize_almonds(self):
        assert _normalize_plural("almonds") == "almonds"

    def test_no_normalize_cultures(self):
        assert _normalize_plural("cultures") == "cultures"

    def test_unknown_word_passes_through(self):
        assert _normalize_plural("random_word") == "random_word"

    def test_singular_passes_through(self):
        assert _normalize_plural("emulsifier") == "emulsifier"

    def test_leavening_agents(self):
        assert _normalize_plural("leavening agents") == "leavening agent"

    def test_gelling_agents(self):
        assert _normalize_plural("gelling agents") == "gelling agent"

    def test_colors_to_colorant(self):
        result = _normalize_plural("colors")
        assert result == "colorant", f"Expected 'colorant', got '{result}'"

    def test_colours_to_colorant(self):
        result = _normalize_plural("colours")
        assert result == "colorant", f"Expected 'colorant', got '{result}'"


# ──────────────────────────────────────────────
# E-number matching tests
# ──────────────────────────────────────────────

class TestEnumberMatching:
    """Verify _match_enumber() classifications."""

    def test_e330_antioxidant(self):
        cats = _match_enumber("e330")
        assert cats is not None
        assert "antioxidant" in cats, f"Expected 'antioxidant' in {cats}"

    def test_e200_preservative(self):
        cats = _match_enumber("e200")
        assert cats is not None
        assert "preservative" in cats, f"Expected 'preservative' in {cats}"

    def test_e100_colorant(self):
        cats = _match_enumber("e100")
        assert cats is not None
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_e400_thickener(self):
        cats = _match_enumber("e400")
        assert cats is not None
        assert "thickener" in cats, f"Expected 'thickener' in {cats}"

    def test_e500_acidulant(self):
        cats = _match_enumber("e500")
        assert cats is not None
        assert "acidulant" in cats, f"Expected 'acidulant' in {cats}"

    def test_e600_flavor_enhancer(self):
        cats = _match_enumber("e600")
        assert cats is not None
        assert "flavor_enhancer" in cats, f"Expected 'flavor_enhancer' in {cats}"

    def test_e900_sweetener(self):
        cats = _match_enumber("e900")
        assert cats is not None
        assert "sweetener" in cats, f"Expected 'sweetener' in {cats}"

    def test_e1000_plus_food_additive(self):
        cats = _match_enumber("e1000")
        assert cats is not None
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"

    def test_invalid_enumber_returns_none(self):
        assert _match_enumber("e9") is None
        assert _match_enumber("abc") is None
        assert _match_enumber("e12345") is None

    def test_e330_via_ingredient_to_categories(self):
        cats = ingredient_to_categories("e330")
        assert "antioxidant" in cats, f"Expected 'antioxidant' in {cats}"

    def test_e200_via_ingredient_to_categories(self):
        cats = ingredient_to_categories("e200")
        assert "preservative" in cats, f"Expected 'preservative' in {cats}"


# ──────────────────────────────────────────────
# FDC color matching tests
# ──────────────────────────────────────────────

class TestFDCColorMatching:
    """Verify _match_fdc_color() classifications."""

    def test_fdc_red_no_40(self):
        cats = _match_fdc_color("fd&c red no. 40")
        assert cats is not None
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_fdc_yellow_no_5(self):
        cats = _match_fdc_color("fd&c yellow no. 5")
        assert cats is not None
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_fdc_blue_no_1(self):
        cats = _match_fdc_color("fd&c blue no. 1")
        assert cats is not None
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_fdc_green_no_3(self):
        cats = _match_fdc_color("fd&c green no. 3")
        assert cats is not None
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_via_ingredient_to_categories(self):
        cats = ingredient_to_categories("FD&C Red No. 40")
        assert len(cats) > 0, "Should resolve via FDC matching"
        assert "colorant" in cats, f"Expected 'colorant' in {cats}"

    def test_no_match(self):
        assert _match_fdc_color("sugar") is None
        assert _match_fdc_color("e330") is None


# ──────────────────────────────────────────────
# INS code matching tests
# ──────────────────────────────────────────────

class TestINSCodeMatching:
    """Verify _match_ins_code() classifications."""

    def test_ins_330(self):
        cats = _match_ins_code("ins 330")
        assert cats is not None
        # E300-399 range → antioxidant
        assert "antioxidant" in cats, f"Expected 'antioxidant' in {cats}"
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"

    def test_ins_200(self):
        cats = _match_ins_code("ins 200")
        assert cats is not None
        # E200-299 range → preservative
        assert "preservative" in cats, f"Expected 'preservative' in {cats}"
        assert "food_additive" in cats, f"Expected 'food_additive' in {cats}"

    def test_via_ingredient_to_categories(self):
        cats = ingredient_to_categories("INS 330")
        assert len(cats) > 0, "Should resolve via INS matching"


# ──────────────────────────────────────────────
# Substring fallback tests
# ──────────────────────────────────────────────

class TestSubstringFallback:
    """Verify substring fallback for ingredients without exact matches."""

    def test_contains_known_keyword(self):
        cats = ingredient_to_categories("natural vanilla flavor")
        assert "flavoring" in cats, f"Expected 'flavoring' in {cats}"

    def test_contains_oil(self):
        cats = ingredient_to_categories("palm olein")
        assert "oil_source" in cats or "fat_source" in cats, \
            f"Expected oil/fat source in {cats}"

    def test_contains_vitamin(self):
        cats = ingredient_to_categories("ascorbic acid")
        assert "vitamin" in cats or "antioxidant" in cats, \
            f"Expected 'vitamin' or 'antioxidant' in {cats}"

    def test_contains_egg(self):
        cats = ingredient_to_categories("egg protein")
        assert "egg_derivative" in cats, f"Expected 'egg_derivative' in {cats}"


# ──────────────────────────────────────────────
# Unknown ingredient tests
# ──────────────────────────────────────────────

class TestUnknownIngredients:
    """Ingredients with no mapping should return empty list."""

    def test_bogus_ingredient_returns_empty(self):
        cats = ingredient_to_categories("xyzzynon_existent_12345")
        assert cats == [], f"Expected [], got {cats}"

    def test_garbage_numbers(self):
        cats = ingredient_to_categories("12345")
        assert cats == [], f"Expected [], got {cats}"

    def test_empty_string(self):
        cats = ingredient_to_categories("")
        assert cats == [], f"Expected [], got {cats}"


# ──────────────────────────────────────────────
# Edge case tests
# ──────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases for ingredient mapping."""

    def test_case_insensitivity(self):
        """Exact match should be case-insensitive."""
        cats_upper = ingredient_to_categories("MSG")
        cats_lower = ingredient_to_categories("msg")
        assert cats_upper == cats_lower, \
            f"Upper {cats_upper} != lower {cats_lower}"

    def test_semantic_categories_are_valid(self):
        """Every category returned should be in SEMANTIC_CATEGORIES."""
        test_ingredients = [
            "spices", "emulsifiers", "e330", "sucralose",
            "salt", "sugar", "wheat flour", "shrimp"
        ]
        for ing in test_ingredients:
            cats = ingredient_to_categories(ing)
            for c in cats:
                assert c in SEMANTIC_CATEGORIES, \
                    f"'{c}' from '{ing}' not in SEMANTIC_CATEGORIES"

    def test_spices_gets_spice_flavoring_plant(self):
        cats = ingredient_to_categories("spices")
        assert "spice" in cats
        assert "plant_derived" in cats, f"Expected 'plant_derived' in {cats}"

    def test_emulsifiers_gets_food_additive_and_emulsifier(self):
        cats = ingredient_to_categories("emulsifiers")
        assert "emulsifier" in cats
        assert "food_additive" in cats
        # Should also have a functional category
        assert any(c in cats for c in ["food_additive", "emulsifier"]), \
            f"Expected functional category in {cats}"


# ──────────────────────────────────────────────
# Integration / data coverage tests
# ──────────────────────────────────────────────

class TestCoverage:
    """Verify that key high-frequency ingredients are mapped."""

    SAMPLE_INGREDIENTS = [
        # Functional terms
        "spices", "emulsifier", "stabilizer", "thickener",
        "preservative", "antioxidant", "flavor enhancer",
        "sweetener", "humectant", "colorant",
        # Specific chemicals
        "sucralose", "vanillin", "silicon dioxide",
        "sodium citrate", "tartrazine", "steviol glycosides",
        # Allergen-relevant
        "shrimp", "fish", "beef", "chicken", "pork",
        # Vitamins
        "riboflavin", "niacin", "folic acid",
        # Oils and fats
        "palm oil", "corn oil", "vegetable fat",
        # Filipino-relevant
        "nata de coco", "fish sauce",
    ]

    def test_all_sample_ingredients_resolve(self):
        """Every high-frequency ingredient should resolve to at least one category."""
        unresolved = []
        for ing in self.SAMPLE_INGREDIENTS:
            cats = ingredient_to_categories(ing)
            if not cats:
                unresolved.append(ing)
        assert len(unresolved) == 0, \
            f"Unresolved ingredients: {unresolved}"

    def test_no_ingredient_returns_too_many_categories(self):
        """Sanity check: no ingredient should return more than 6 categories."""
        for ing in self.SAMPLE_INGREDIENTS:
            cats = ingredient_to_categories(ing)
            assert len(cats) <= 6, \
                f"'{ing}' returned {len(cats)} categories (>6): {cats}"
