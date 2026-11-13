"""
Shared test fixtures and constants for the food label transparency project.

All test files in tests/ can import from conftest.py automatically — pytest
discovers conftest.py fixtures without explicit imports.

Usage:
    def test_something(sample_ingredient_texts, sample_label_vectors):
        text = sample_ingredient_texts["tree_nuts_basic"]
        ...
"""

import pytest

# ──────────────────────────────────────────────
# Allergen constants (single source of truth for tests)
# ──────────────────────────────────────────────

ALLERGEN_LIST = ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]

# Index positions matching ALLERGEN_LIST
MILK_IDX = 0
EGGS_IDX = 1
PEANUTS_IDX = 2
TREE_NUTS_IDX = 3
SOY_IDX = 4
WHEAT_IDX = 5
FISH_IDX = 6
SHELLFISH_IDX = 7


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_ingredient_texts():
    """Dictionary of sample ingredient texts for various allergen scenarios.

    Each key names the scenario; the value is ingredient text. Useful for
    parametrizing tests across different allergen combinations, negation
    patterns, exemptions, and edge cases.
    """
    return {
        # Basic single-allergen texts
        "milk_basic": "milk, sugar, wheat flour",
        "eggs_basic": "eggs, salt, pepper",
        "peanuts_basic": "peanut butter, sugar",
        "tree_nuts_basic": "almonds, cashews, salt",
        "soy_basic": "soy sauce, sugar, vinegar",
        "wheat_basic": "wheat flour, water, yeast",
        "fish_basic": "tuna, oil, salt",
        "shellfish_basic": "shrimp, garlic, butter",

        # Filipino variant texts
        "tree_nuts_filipino": "kasoy, almendras, asin",
        "milk_filipino": "gatas, asukal",
        "eggs_filipino": "itlog, asin",
        "peanuts_filipino": "mani, asukal",
        "soy_filipino": "toyo, asukal",
        "wheat_filipino": "harina, tubig",
        "fish_filipino": "isda, asin",
        "shellfish_filipino": "hipon, bawang",

        # Mixed allergens
        "mixed_allergens": "milk, eggs, peanut butter, soy sauce, wheat",
        "triple_nuts": "almonds, peanuts, cashews, soy",

        # Negation patterns
        "negated_milk": "no milk, contains soy",
        "negated_peanuts": "peanut-free, contains wheat",
        "negated_eggs": "does not contain eggs, milk present",
        "negated_wheat": "free from wheat, contains soy",
        "negated_allergen_statement": "contains no peanuts, may contain tree nuts",

        # Exemption patterns
        "coconut_exempt": "coconut oil, shea butter, salt",
        "soy_lecithin_exempt": "sunflower lecithin, sugar, salt",
        "standalone_lecithin": "lecithin, sugar",  # should NOT trigger eggs

        # Tree nut specific — uncommon/exotic nuts
        "tree_nuts_uncommon": "pili nuts, ginkgo nuts, kola nuts",

        # Edge cases
        "empty": "",
        "no_allergens": "sugar, salt, water, citric acid",
        "single_word": "milk",
        "whitespace_only": "   ",

        # May-contain statements
        "may_contain_peanuts": "may contain peanuts, processed in facility",
        "may_contain_tree_nuts": "may contain tree nuts, sugar",

        # Complex scenarios
        "multiple_negations": "no milk, no eggs, wheat free, contains soy",
        "nested_allergens": "milk chocolate, peanut butter cookies",
        "derivative_forms": "whey protein, caseinate, albumin",
    }


@pytest.fixture
def sample_label_vectors():
    """Dictionary of binary label vectors for common allergen combinations.

    Each vector is length 8 in the order:
    [milk, eggs, peanuts, tree_nuts, soy, wheat, fish, shellfish]
    """
    return {
        "empty": [0, 0, 0, 0, 0, 0, 0, 0],
        "milk_only": [1, 0, 0, 0, 0, 0, 0, 0],
        "eggs_only": [0, 1, 0, 0, 0, 0, 0, 0],
        "peanuts_only": [0, 0, 1, 0, 0, 0, 0, 0],
        "tree_nuts_only": [0, 0, 0, 1, 0, 0, 0, 0],
        "soy_only": [0, 0, 0, 0, 1, 0, 0, 0],
        "wheat_only": [0, 0, 0, 0, 0, 1, 0, 0],
        "fish_only": [0, 0, 0, 0, 0, 0, 1, 0],
        "shellfish_only": [0, 0, 0, 0, 0, 0, 0, 1],
        "milk_and_wheat": [1, 0, 0, 0, 0, 1, 0, 0],
        "all_allergens": [1, 1, 1, 1, 1, 1, 1, 1],
        "common_pair": [1, 0, 0, 0, 1, 0, 0, 0],  # milk + soy
    }


@pytest.fixture
def temporary_dir(tmp_path):
    """Provide a temporary directory for file I/O tests.

    Uses pytest's built-in tmp_path fixture. Files written here are
    automatically cleaned up after each test.
    """
    return tmp_path
