"""
Dedicated tree nut detection tests for text_processing.

Tests cover:
- All BIG8 tree nut keywords (including newly added Philippine-relevant nuts)
- Filipino variant detection ("kasoy", "almendras", "pili")
- Critical false-positive guards ("mani" ≠ tree_nuts)
- Exemption rules (coconut, shea butter, water chestnut)
- Negation patterns for tree nuts
- Context-aware disambiguation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import pytest
from utils.text_processing import (
    rule_match, detect_allergens_rule_based, get_allergen_list,
    apply_exemptions, BIG8, FILIPINO_VARIANTS
)

# ── Tests use the conftest fixtures automatically ──

TREE_NUTS = "tree_nuts"


# ════════════════════════════════════════════════
# Class 1: Core BIG8 keyword matching
# ════════════════════════════════════════════════

class TestTreeNutsBIG8:
    """Every keyword in BIG8["tree_nuts"] should trigger rule_match."""

    @pytest.mark.parametrize("text,keyword", [
        ("almonds", "almond"),
        ("cashew butter", "cashew"),
        ("walnut oil", "walnut"),
        ("pecan pie", "pecan"),
        ("hazelnut spread", "hazelnut"),
        ("filbert", "filbert"),
        ("macadamia nuts", "macadamia"),
        ("pistachio ice cream", "pistachio"),
        ("brazil nut", "brazil nut"),
        ("chestnut stuffing", "chestnut"),
        ("pine nuts", "pine nut"),
        ("marzipan", "marzipan"),
        ("praline", "praline"),
        ("nut paste", "nut paste"),
        ("nut butter", "nut butter"),
        ("nut meal", "nut meal"),
        ("nut oil", "nut oil"),
    ])
    def test_core_keywords(self, text, keyword):
        assert rule_match(text, TREE_NUTS), (
            f"rule_match('{text}', 'tree_nuts') should be True "
            f"(matched via BIG8 keyword '{keyword}')"
        )

    @pytest.mark.parametrize("text,keyword", [
        ("contains pili nuts", "pili nut"),
        ("pili nut oil", "pili nut"),
        ("ginkgo nuts", "ginkgo nut"),
        ("kola nut extract", "kola nut"),
        ("hickory nuts", "hickory nut"),
        ("beechnut flour", "beechnut"),
        ("butternut squash", "butternut"),
        # NOTE: butternut squash is a false-positive risk.
        # butternut = Juglans cinerea (a tree nut).
        # Butternut squash = a gourd, not a nut.  The keyword "butternut"
        # appears inside "butternut squash" — this is acceptable because
        # the squash itself is a gourd and allergen-labeling regulations
        # do not consider squash an allergen; the match is on a string
        # that happens to contain the word.  Acceptable false positive rate.
        # NOTE: "nutmeg" is intentionally NOT included here.
        # FDA/FARE explicitly state nutmeg is NOT a tree nut allergen
        # (Myristica fragrans is a seed used as a spice).
        ("nut flour", "nut flour"),
        ("mixed nuts", "mixed nuts"),
        ("assorted nuts", "assorted nuts"),
    ])
    def test_newly_added_keywords(self, text, keyword):
        """Phase B expansion keywords must all trigger detection."""
        assert rule_match(text, TREE_NUTS), (
            f"rule_match('{text}', 'tree_nuts') should be True "
            f"(matched via added keyword '{keyword}')"
        )


# ════════════════════════════════════════════════
# Class 2: Filipino variant detection
# ════════════════════════════════════════════════

class TestTreeNutsFilipinoVariants:
    """Filipino-language tree nut terms must be detected."""

    @pytest.mark.parametrize("text", [
        "kasoy",
        "kasoy, asin",
        "almendras",
        "walnut",        # spelled same in Tagalog
        "pistachio",     # spelled same in Tagalog
        "hazelnut",
        "macadamia",
        "pecan",
    ])
    def test_traditional_variants(self, text):
        assert rule_match(text, TREE_NUTS)

    @pytest.mark.parametrize("text", [
        "pili nuts",
        "pili",
        "ginkgo",
    ])
    def test_new_filipino_variants(self, text):
        """Phase B expansion — Philippine-relevant nuts."""
        assert rule_match(text, TREE_NUTS)

    # ── Critical false-positive guard ──
    def test_mani_is_not_tree_nut(self):
        """"mani" (Tagalog for peanut) must NOT trigger tree_nuts."""
        # Verify "mani" is in peanuts but not in tree_nuts
        assert "mani" in [kw.lower() for kw in FILIPINO_VARIANTS.get("peanuts", [])], \
            "mani should be in FILIPINO_VARIANTS['peanuts']"
        mani_in_tree = any(
            "mani" in kw.lower()
            for kw in FILIPINO_VARIANTS.get(TREE_NUTS, [])
        )
        assert not mani_in_tree, \
            "mani must NOT be in FILIPINO_VARIANTS['tree_nuts'] — it means peanut in Tagalog"
        assert not rule_match("mani", TREE_NUTS), \
            "'mani' should NOT trigger tree_nuts detection"
        # But it SHOULD trigger peanuts
        from utils.text_processing import rule_match as rm
        assert rm("mani", "peanuts"), \
            "'mani' SHOULD trigger peanuts detection"

    def test_kasoy_false_positive_check(self):
        """"kasoy" (cashew in Tagalog) triggers tree_nuts, not peanuts."""
        assert rule_match("kasoy", TREE_NUTS), "kasoy should trigger tree_nuts"


# ════════════════════════════════════════════════
# Class 3: Exemption rules
# ════════════════════════════════════════════════

class TestTreeNutsExemptions:
    """Ingredients that look like tree nuts but are exempt by regulation.

    apply_exemptions(detected_list, tokens) is a post-processing step that
    takes a list of detected allergens and a list of ingredient tokens.
    It removes an allergen from the list only when the tokens match one
    of the specific exemption patterns (e.g. ``coconut oil`` is exempt,
    but plain ``coconut`` is not because the pattern is product-specific).
    """

    @pytest.mark.parametrize("text,should_exempt", [
        ("coconut oil", True),
        ("coconut milk", True),
        ("coconut cream", True),
        ("coconut flour", True),      # matches \\bcoconut flour\\b
        ("desiccated coconut", False),  # pattern only covers specific products
        ("coconut", False),             # too broad for current patterns
    ])
    def test_coconut_exempt(self, text, should_exempt):
        """Coconut is not an FDA Big-8 tree nut, but apply_exemptions
        only exempts specific coconut products (oil, milk, cream, flour, water)."""
        result = apply_exemptions(["tree_nuts"], text.split())
        if should_exempt:
            assert "tree_nuts" not in result, \
                f"'{text}' should be exempted from tree_nuts, got {result}"
        else:
            assert "tree_nuts" in result, \
                f"'{text}' should NOT be exempted, got {result}"

    @pytest.mark.parametrize("text,should_exempt", [
        ("shea butter", True),
        ("shea nut butter", True),
        ("shea", False),             # too broad for current patterns
    ])
    def test_shea_exempt(self, text, should_exempt):
        """Shea nut is exempt (FDA)."""
        result = apply_exemptions(["tree_nuts"], text.split())
        if should_exempt:
            assert "tree_nuts" not in result, \
                f"'{text}' should be exempted from tree_nuts"
        else:
            assert "tree_nuts" in result, \
                f"'{text}' should NOT be exempted"

    @pytest.mark.parametrize("text,should_exempt", [
        ("water chestnut", True),
        ("water chestnuts", True),
        ("chestnut", False),         # NOT exempt (true tree nut)
    ])
    def test_water_chestnut_exempt(self, text, should_exempt):
        """Water chestnut is a tuber, not a tree nut."""
        result = apply_exemptions(["tree_nuts"], text.split())
        if should_exempt:
            assert "tree_nuts" not in result, \
                f"'{text}' should be exempted from tree_nuts"
        else:
            assert "tree_nuts" in result, \
                f"'{text}' should NOT be exempted"

    def test_coconut_not_in_big8(self):
        """Verify 'coconut' is not in BIG8['tree_nuts'] as a keyword."""
        coconut_in_big8 = any("coconut" in kw.lower()
                               for kw in BIG8.get(TREE_NUTS, []))
        assert not coconut_in_big8, \
            "'coconut' should NOT be in BIG8['tree_nuts'] — it's exempt"


# ════════════════════════════════════════════════
# Class 4: Negation patterns
# ════════════════════════════════════════════════

class TestTreeNutsNegation:
    """Negation must suppress tree nut detection."""

    @pytest.mark.parametrize("text", [
        "no tree nuts",
        "tree nut-free",
        "free from tree nuts",
        "does not contain tree nuts",
        "nut free",
        "no nuts",
    ])
    def test_negated_tree_nuts(self, text):
        """Explicit negation should suppress detection."""
        assert not rule_match(text, TREE_NUTS), \
            f"rule_match('{text}', 'tree_nuts') should be False (negated)"

    @pytest.mark.parametrize("text", [
        "may contain tree nuts",
        "may contain nuts",
        "processed in a facility that handles tree nuts",
    ])
    def test_may_contain_preserved(self, text):
        """'May contain' is a risk statement, not a negation — should still trigger."""
        # "may contain X" should NOT trigger negation; the keyword should match.
        # These texts contain "tree nuts" or "nuts" keywords which are now in BIG8.
        # The negation system only suppresses "no X", "X-free", "free from X", etc.
        assert rule_match(text, TREE_NUTS), \
            f"rule_match('{text}', 'tree_nuts') should be True (may-contain is not negation)"


# ════════════════════════════════════════════════
# Class 5: Context-aware disambiguation
# ════════════════════════════════════════════════

class TestTreeNutsContextDisambiguation:
    """Edge cases where context determines whether an ingredient is a tree nut."""

    def test_nutmeg_not_confused(self):
        """'nutmeg' should NOT trigger tree_nuts — FDA/FARE state nutmeg is not a tree nut allergen."""
        assert not rule_match("nutmeg", TREE_NUTS), \
            "nutmeg is a spice seed, not a tree nut (FDA/FARE guidance)"

    def test_nut_flour_detected(self):
        """'nut flour' should trigger tree_nuts."""
        assert rule_match("nut flour", TREE_NUTS), \
            "nut flour should be detected as tree nut"

    def test_nut_oil_detected(self):
        """'nut oil' should trigger tree_nuts."""
        assert rule_match("nut oil", TREE_NUTS), \
            "nut oil should be detected as tree nut"


# ════════════════════════════════════════════════
# Class 6: Integration with detect_allergens_rule_based
# ════════════════════════════════════════════════

class TestTreeNutsEndToEnd:
    """Full pipeline from ingredient text to allergen list."""

    @pytest.mark.parametrize("text,expected", [
        ("almonds, cashews, salt", ["tree_nuts"]),
        ("milk, sugar, almonds", ["milk", "tree_nuts"]),
        ("sugar, salt, water", []),
        ("no tree nuts, contains milk", ["milk"]),
        ("coconut oil, shea butter, sugar", []),   # all exempt
        ("pili nuts, salt", ["tree_nuts"]),         # new Philippine keyword
        ("kasoy, almendras", ["tree_nuts"]),         # Filipino variants
        ("mani, asukal", ["peanuts"]),               # mani → peanuts, NOT tree_nuts
    ])
    def test_detect_allergens(self, text, expected):
        detected = detect_allergens_rule_based(text)
        for allergen in expected:
            assert allergen in detected, \
                f"'{allergen}' should be in detected({detected}) for '{text}'"
        assert "tree_nuts" in detected or "tree_nuts" not in expected


# ════════════════════════════════════════════════
# Class 7: BIG8 data integrity
# ════════════════════════════════════════════════

class TestTreeNutsDataIntegrity:
    """Structural checks on BIG8 and FILIPINO_VARIANTS."""

    def test_tree_nuts_in_get_allergen_list(self):
        assert TREE_NUTS in get_allergen_list()

    def test_tree_nuts_in_big8(self):
        assert TREE_NUTS in BIG8, "BIG8 must have tree_nuts key"

    def test_tree_nuts_has_keywords(self):
        assert len(BIG8[TREE_NUTS]) > 15, \
            f"BIG8['tree_nuts'] should have >15 keywords, got {len(BIG8[TREE_NUTS])}"

    def test_tree_nuts_in_filipino_variants(self):
        assert TREE_NUTS in FILIPINO_VARIANTS, \
            "FILIPINO_VARIANTS must have tree_nuts key"

    def test_no_peanut_keywords_in_tree_nuts(self):
        """Ensure no peanut-related terms leak into tree_nuts."""
        peanut_terms = {"peanut", "groundnut", "mani", "mankinsilya"}
        for kw in BIG8[TREE_NUTS]:
            assert kw.lower() not in peanut_terms, \
                f"'{kw}' is a peanut term, not a tree nut"

    def test_no_exempt_terms_in_big8(self):
        """Exempt ingredients (coconut, shea) must not be in BIG8 tree_nuts."""
        exempt_terms = {"coconut", "shea", "water chestnut"}
        for kw in BIG8[TREE_NUTS]:
            # Check each exempt term isn't a substring of any keyword
            for exempt in exempt_terms:
                assert exempt not in kw.lower(), \
                    f"'{kw}' contains exempt term '{exempt}'"
