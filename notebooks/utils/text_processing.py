"""
Text processing for food label allergen detection.

Centralizes all text cleaning, BIG8 keyword matching, context disambiguation, negation
handling, exemption logic, and label parsing. Notebooks should import from here
rather than redefining this logic.

Usage:
    >>> from utils.text_processing import detect_allergens_rule_based, clean_html, get_allergen_list
    >>> allergens = detect_allergens_rule_based("milk, sugar, wheat flour")
    >>> cleaned = clean_html("<div>milk - free</div>")
"""

__all__ = [
    # Core allergen functions
    "detect_allergens_rule_based",
    "rule_match",
    "BIG8",
    "ALLERGEN_RULES",
    "OFFICIAL_MAP",
    "COMPILED_RULES",
    "NEGATION_PATTERN",

    # Text processing functions
    "clean_html",
    "preprocess_text",
    "clean_ingredient_text",
    "tokenize_ingredients",

    # Label parsing functions
    "parse_label_string",
    "get_allergen_list",
    "allergens_to_binary",

    # Special utilities
    "extract_may_contain",
    "has_explicit_allergen_statement",
    "parse_traces_tags",
    "parse_official_tags",
    "apply_exemptions",
    "combine_allergen_labels",
    "detect_coconut_improved",

    # Filipino variant data
    "FILIPINO_VARIANTS",
]

import re
import ast
from typing import List, Dict, Tuple, Union, Optional
import numpy as np

# ──────────────────────────────────────────────
# Allergen keyword dictionaries
# ──────────────────────────────────────────────

# Expanded allergen keywords (consolidated from notebooks 03's BIG8 + original ALLERGEN_RULES)
# DESIGN NOTE: Broad/ambiguous keywords (standalone "cream", "flour", "lecithin") have been
# removed from BIG8 to reduce false positives. They are handled by CONTEXTUAL_KEYWORDS below
# with disambiguation rules. The ML model can learn to catch cases the rule-based system misses.
BIG8 = {
    "milk": [
        "milk", "whey", "casein", "caseinate", "butter", "cheese",
        "lactose", "ghee", "buttermilk", "milk solids", "skim milk",
        "whole milk", "milk powder", "nonfat milk", "evaporated milk",
        "condensed milk", "powdered milk", "curds",
        "lactalbumin", "lactoglobulin", "butterfat",
        "whey protein",
        # "cream" is broad (can be non-dairy), but context-aware matching
        # filters false positives like "coconut cream", "almond cream"
        "cream",
    ],
    "eggs": [
        "egg", "eggs", "albumin", "egg white", "egg yolk",
        "ovalbumin", "lysozyme", "globulin", "ovomucoid",
        "mayonnaise", "meringue", "ovotransferrin", "eggshell"
    ],
    "peanuts": [
        "peanut", "peanuts", "groundnut",
        "peanut butter", "peanut oil", "arachis oil",
        "monkey nut", "peanut flour", "peanut protein"
    ],
    "tree_nuts": [
        "almond", "almonds", "cashew", "cashews",       # +plurals
        "walnut", "walnuts", "pecan", "pecans",
        "hazelnut", "hazelnuts", "filbert",
        "macadamia", "macadamias", "pistachio", "pistachios",
        "brazil nut", "brazil nuts",                    # +plural
        "chestnut", "chestnuts",                         # +plural
        "pine nut", "pine nuts", "marzipan", "praline",
        "nut paste", "nut butter", "nut meal", "nut oil",
        "nuts",                                          # generic (doughnuts/peanuts exempted by \b)
        "tree nuts", "tree nut",                         # explicit phrase in allergen warnings
        "pili nut", "pili nuts",                    # Canarium ovatum — native Philippine nut
        "ginkgo nut", "ginkgo nuts",                # Ginkgo biloba — Asian cuisine
        "kola nut", "kola nuts",                    # Cola acuminata — tree nut, not just flavor
        "hickory nut", "hickory nuts",              # Carya genus
        "beechnut", "beechnuts",                    # Fagus genus
        "butternut", "butternuts",                  # Juglans cinerea
        # NOTE: "nutmeg" is EXCLUDED — FDA/FARE state nutmeg is NOT a tree nut allergen
        "nut flour",                                # common processed form
        "mixed nuts", "assorted nuts",
    ],
    "soy": [
        "soy", "soya", "soybean", "soybeans", "edamame", "natto",
        "soy lecithin",
        "soy protein", "textured vegetable protein",
        "tvp", "tofu", "miso", "tempeh",
        "soy sauce", "tamari", "soy milk", "soybean oil", "soy flour"
    ],
    "wheat": [
        "wheat", "whole wheat", "wheat flour", "spelt", "kamut", "triticale", "durum",
        "gluten", "semolina",
        "farina", "bulgur",
        "wheat germ", "wheat starch", "einkorn"
    ],
    "fish": [
        "fish", "tuna", "salmon", "sardine",
        "anchovy", "mackerel", "cod",
        "fish sauce", "fish oil",
        "surimi", "pollock", "trout", "roe",
        "fish gelatin", "isinglass", "caviar", "fish meal",
        "omega-3 from fish"
    ],
    "shellfish": [
        "shrimp", "prawn", "crab", "lobster",
        "crayfish", "krill", "clam", "mussel", "oyster", "scallop", "crawfish", "langoustine",
        "squid", "crab paste", "abalone", "conch", "cockle", "whelk",
        "crab meat", "lobster paste"
    ],
}

# ──────────────────────────────────────────────
# Context-aware keyword disambiguation rules
# ──────────────────────────────────────────────
#
# Ambiguous terms that are NOT in BIG8 directly because they match many
# non-allergen ingredients. Instead, they are checked only when specific
# context rules pass.
#
# Structure: {allergen: {keyword: {"disallow_when_preceded_by": [...], ...}}}
# - disallow_when_preceded_by: if any of these tokens appear immediately before
#   the keyword, the match is rejected (e.g. "coconut cream" → not milk)
# - disallow_when_followed_by: same but after the keyword
# - require_preceded_by: MUST be preceded by one of these to match (AND logic)
# - require_followed_by: MUST be followed by one of these to match
AMBIGUOUS_KEYWORDS = {
    "milk": {
        "cream": {
            "disallow_when_preceded_by": [
                "coconut", "coconut milk", "non-dairy", "nondairy",
                "vegetable", "almond", "soy", "oat", "rice",
                "sunflower", "cashew",
            ],
            "disallow_when_followed_by": [
                "substitute", "alternative", "non-dairy", "nondairy",
            ],
        },
    },
    "soy": {
        "lecithin": {
            # "soy lecithin" is already in BIG8 — this catches standalone "lecithin"
            # but rejects it if preceded by "sunflower" or similar
            "disallow_when_preceded_by": [
                "sunflower", "canola", "sun flower",
            ],
            "require_preceded_by": [
                "soy", "soya", "soybean", "soya bean",
            ],
        },
    },
    "wheat": {
        "flour": {
            "disallow_when_preceded_by": [
                "rice", "corn", "maize", "tapioca", "cassava",
                "potato", "coconut", "almond", "oat", "barley",
                "rye", "chickpea", "gram", "bean", "soy",
                "buckwheat", "millet", "sorghum", "quinoa",
                "amaranth", "teff", "arrowroot", "chestnut",
                "sesame", "linseed", "hemp", "pea",
            ],
            "require_preceded_by": [
                "wheat", "whole wheat", "white wheat", "bread",
            ],
        },
    },
}

# Per-keyword specificity rules:
#   milk/cream — often plant-based; disallow when preceded by non-dairy modifiers
#   soy/lecithin — sunflower lecithin is common; require "soy" prefix
#   wheat/flour — rice flour, corn flour, etc.; require "wheat" prefix

# ──────────────────────────────────────────────
# Filipino ingredient variants (Thesis Plan Deliverable 3)
# ──────────────────────────────────────────────

# Common Tagalog/Filipino terms for allergens found in Philippine food labels
FILIPINO_VARIANTS = {
    "milk": [
        "gatas", "gatas na pulbos", "gatas na kondensada",
        "keso", "kesong puti", "mantikilya", "krema",
        "milk", "whey", "kasein", "lactose",
    ],
    "eggs": [
        "itlog", "itlog na pula", "puti ng itlog", "buro",
        "itlog na maalat", "itlog na asin",
    ],
    "peanuts": [
        "mani", "mantika ng mani", "butter ng mani",
        "peanut", "groundnut", "mankinsilya",
    ],
    "tree_nuts": [
        "kasoy", "almendras", "walnut", "pistachio",
        "hazelnut", "macadamia", "pecan",
        "pili", "pili nuts",                       # Canarium ovatum
        "ginkgo", "ginkgo nut",                     # Added local name
    ],
    "soy": [
        "toyo", "tokwa", "taho", "miso", "bitsin", "patis",
        "toyo beans", "soy sauce", "tausi", "tausi paste",
    ],
    "wheat": [
        "trigo", "harina", "harinang trigo", "tinapay",
        "pansit", "bihon", "miki", "canton", "sotanghon",
        "pasta", "spaghetti", "elbow macaroni",
        "grano", "semolina", "durum",
    ],
    "fish": [
        "isda", "bagoong", "patis", "dilis", "sardinas",
        "tuna", "galunggong", "bangus", "tilapia",
        "tuyo", "daing", "tinapa", "fish sauce",
    ],
    "shellfish": [
        "hipon", "sugpo", "alamang", "bagoong alamang",
        "alimasag", "talangka", "suso", "tahong",
        "talaba", "halaan", "kabibe", "lobster",
        "prawn", "crab", "shrimp", "crayfish",
    ],
}

# Merge Filipino variants into the main BIG8 dictionary so COMPILED_RULES
# and all downstream functions (rule_match, detect_allergens_rule_based)
# automatically include both English and Filipino terms.
for _allergen, _variants in FILIPINO_VARIANTS.items():
    if _allergen in BIG8:
        for _v in _variants:
            if _v not in BIG8[_allergen]:
                BIG8[_allergen].append(_v)

# Alias for backward compatibility
ALLERGEN_RULES = BIG8

# Mapping from official tag suffix to Big-8 key
OFFICIAL_MAP = {
    # Big-8 allergens
    "gluten": "wheat",
    "wheat": "wheat",
    "milk": "milk",
    "eggs": "eggs",
    "soybeans": "soy",
    "soya": "soy",
    "peanuts": "peanuts",
    "tree nuts": "tree_nuts",
    "nuts": "tree_nuts",
    "fish": "fish",
    # OFF taxonomy uses "crustaceans" and "molluscs" rather than "shellfish"
    "crustaceans": "shellfish",
    "molluscs": "shellfish",
    "shellfish": "shellfish"
}

# Pre-compile regex patterns for each allergen
COMPILED_RULES = {}
for allergen, keywords in ALLERGEN_RULES.items():
    pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
    COMPILED_RULES[allergen] = re.compile(pattern, re.IGNORECASE)

# Negation pattern (enhanced to catch missed constructions while reducing false positives)
NEGATION_PATTERN = re.compile(
    r'\b(no|not|free|without|minus|none)\s+(?:\w+\s+)?\w+\b|\b(does not contain|free from)\s+\w+\b|\b\w+-\w+free\b',
    re.IGNORECASE
)

# Extended negation phrases for pre-check before positive match
NEGATION_PHRASES_BEFORE = [
    "does not contain", "free from", "contains no", "not a source of",
    "not a significant source of", "no added", "no artificial",
    "manufactured in a facility that also processes",
    "produced in a facility that also processes",
    "packaged in a facility that also processes",
]
NEGATION_COMPILED_BEFORE = re.compile(
    r'\b(?:' + '|'.join(re.escape(p) for p in NEGATION_PHRASES_BEFORE) + r')\s+',
    re.IGNORECASE
)

def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', str(text))

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: lowercase and strip."""
    return text.lower().strip()

def _context_aware_match(text_lower: str, allergen: str, keyword: str) -> bool:
    """
    Check if a keyword match is valid given context disambiguation rules.

    Handles AMBIGUOUS_KEYWORDS where certain terms (like "cream", "flour")
    only count as allergen matches when not preceded/followed by disallowed tokens.

    Args:
        text_lower: Lowercased input text
        allergen: Allergen key
        keyword: The matched keyword from the BIG8 or AMBIGUOUS_KEYWORDS dict

    Returns:
        True if the match is valid (passes context rules), False if should be rejected
    """
    if allergen not in AMBIGUOUS_KEYWORDS:
        return True  # No disambiguation rules for this allergen
    if keyword not in AMBIGUOUS_KEYWORDS[allergen]:
        return True  # No disambiguation rules for this keyword

    rules = AMBIGUOUS_KEYWORDS[allergen][keyword]
    kw_escaped = re.escape(keyword)

    # Check disallow_when_preceded_by — if a disallowed token precedes keyword, reject
    for word in rules.get("disallow_when_preceded_by", []):
        # Matches "disallowed_word keyword" with optional space or hyphen
        pattern = r'\b' + re.escape(word) + r'[\s\-]+' + kw_escaped + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    # Check disallow_when_followed_by
    for word in rules.get("disallow_when_followed_by", []):
        pattern = r'\b' + kw_escaped + r'[\s\-]+' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    # Check require_preceded_by — MUST have one of these before keyword
    required_before = rules.get("require_preceded_by", [])
    if required_before:
        for word in required_before:
            pattern = r'\b' + re.escape(word) + r'[\s\-]+' + kw_escaped + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        # None of the required preceding words found — reject
        return False

    # Check require_followed_by — MUST have one of these after keyword
    required_after = rules.get("require_followed_by", [])
    if required_after:
        for word in required_after:
            pattern = r'\b' + kw_escaped + r'[\s\-]+' + re.escape(word) + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    return True


# ──────────────────────────────────────────────
# Allergen stem variations for negation checks
# ──────────────────────────────────────────────
ALLERGEN_STEMS = {
    "milk": ["milk"],
    "eggs": ["egg", "eggs"],
    "peanuts": ["peanut", "peanuts"],
    "tree_nuts": ["tree nut", "tree nuts", "nut", "nuts"],
    "soy": ["soy", "soya"],
    "wheat": ["wheat"],
    "fish": ["fish"],
    "shellfish": ["shellfish"],
}


def _check_negation(text_lower: str, allergen: str, matched_keyword: Optional[str] = None) -> bool:
    """
    Check if an allergen match in text is negated.

    Returns True if negated (i.e., the match should be skipped).

    Checks 4 patterns:
      1. Negation word BEFORE stem: "no milk", "free from wheat"
      2. Multi-word phrase BEFORE stem: "does not contain eggs"
      3. Stem followed by negation suffix: "milk-free", "peanut-free"
      4. Stem followed by negation word: "wheat free", "egg free"

    IMPORTANT: For hyphenated compounds like "milk-chocolate" or "egg-white",
    pattern 3 only matches suffixes like "-free", "-less", "-absent" etc.
    Regular hyphenated compounds are NOT treated as negation.

    Checks negation against both the allergen's canonical stems AND the
    matched keyword (which may be something like "gluten" for the "wheat"
    allergen). This ensures phrases like "free from gluten" are correctly
    recognized as negation.
    """
    stems = ALLERGEN_STEMS.get(allergen, [allergen])
    if matched_keyword and matched_keyword not in stems:
        stems = stems + [matched_keyword]

    for stem in stems:
        stem_escaped = re.escape(stem)

        # Pattern 1: negation words before stem (with optional intervening words)
        # e.g., "no milk", "free from wheat"
        # NOTE: "low in X" means reduced quantity, NOT absence — not treated as negation
        negation_before = re.compile(
            r'\b(no|not|free|without|minus|none)\s+(?:\w+\s+)?' + stem_escaped + r'\b',
            re.IGNORECASE
        )
        # Pattern 2: multi-word negation phrases before stem
        # e.g., "does not contain eggs", "free from gluten"
        negation_phrases = re.compile(
            r'\b(does not contain|free from|contains no)\s+' + stem_escaped + r'\b',
            re.IGNORECASE
        )
        # Pattern 3: stem followed by explicit negation suffix
        # Only matches known suffixes to avoid false negatives on
        # hyphenated compounds like "milk-chocolate", "peanut-butter", "egg-white"
        negation_after = re.compile(
            r'\b' + stem_escaped + r'(?:[a-z]+)?-(?:free|less|absent|avoid|removed|excluded|none)\b',
            re.IGNORECASE
        )
        # Pattern 4: stem followed by negation word
        # e.g., "wheat free", "egg free", "milk free"
        negation_after_space = re.compile(
            r'\b' + stem_escaped + r'\s+(no|not|free|without|minus|none)\b',
            re.IGNORECASE
        )

        if (negation_before.search(text_lower) or
            negation_phrases.search(text_lower) or
            negation_after.search(text_lower) or
            negation_after_space.search(text_lower)):
            return True

    return False


def rule_match(text: str, allergen: str) -> bool:
    """
    Check if allergen is present in text using rule-based matching.

    Checks the core BIG8 keywords first, then applies:
    - Context disambiguation (for ambiguous terms like "cream", "flour")
    - Negation detection (e.g., "no milk", "peanut-free", "contains no eggs")
    - Exemption checks (e.g., "may contain" statements are excluded)

    Args:
        text: Input text to search
        allergen: Allergen to check for (must be key in ALLERGEN_RULES)

    Returns:
        True if allergen found and not negated, False otherwise
    """
    if allergen not in COMPILED_RULES:
        raise ValueError(f"Unknown allergen: {allergen}. Must be one of {list(ALLERGEN_RULES.keys())}")

    if not isinstance(text, str) or not text.strip():
        return False

    text_lower = text.lower()

    # Strip underscores -- they break \b word boundary matching because
    # underscore (_) is a "word character" in regex. Ingredient lists
    # sometimes use underscores for emphasis (e.g., _almond_), which
    # prevents \b from matching across the word boundary.
    text_lower = text_lower.replace('_', ' ')

    # Step 1: Check core BIG8 keyword match
    # Find all matching keywords for this allergen
    match = COMPILED_RULES[allergen].search(text_lower)
    if not match:
        return False

    matched_keyword = match.group(0)

    # Step 2: Apply context disambiguation for ambiguous terms
    if not _context_aware_match(text_lower, allergen, matched_keyword.lower()):
        return False

    # Step 3: Check negation — if negated, return False
    # Pass the matched keyword so negation patterns like "free from gluten"
    # are detected even though "gluten" differs from the canonical stem "wheat"
    if _check_negation(text_lower, allergen, matched_keyword=matched_keyword):
        return False

    # Step 4: Check if this is a "may contain" statement (not a declaration of presence)
    # We only check this for the broader match (not per-stem)
    if re.search(r'\bmay contain\b.*' + re.escape(matched_keyword), text_lower, re.IGNORECASE):
        # "May contain" means traces, not intentional ingredient — still counts as present
        # per food labeling rules. Return True.
        pass

    return True

def detect_allergens_rule_based(text: str) -> List[str]:
    """
    Detect allergens using rule-based approach.

    Args:
        text: Input text to analyze

    Returns:
        List of detected allergens
    """
    detected = []
    for allergen in ALLERGEN_RULES.keys():
        if rule_match(text, allergen):
            detected.append(allergen)
    return detected

def clean_ingredient_text(text: str) -> str:
    """
    Clean ingredient text by removing allergen declarations and noise.

    Args:
        text: Raw ingredient text

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return text

    # Patterns to remove (case-insensitive)
    patterns = [
        r'allergen information[^.]*\.',
        r'may contain[^.]*\.',
        r'contains\s+(?:wheat|milk|eggs|soy|fish|shellfish|peanuts|tree nuts)[^.]*\.',
        r'free from[^.]*\.',
        r'does not contain[^.]*\.',
        r'contains no[^.]*\.'
    ]

    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    return text.strip()

def tokenize_ingredients(text: str) -> List[str]:
    """
    Tokenize ingredient text into individual ingredients.

    Args:
        text: Ingredient text (cleaned)

    Returns:
        List of ingredient tokens
    """
    if not isinstance(text, str):
        return []

    # Split by common delimiters and clean tokens
    tokens = re.split(r'[,;]', text)
    tokens = [t.strip() for t in tokens if t.strip()]

    # Remove tokens that are likely non-ingredients
    stop_tokens = {"span", "class", "allergen", "div", "id", "style", "\\", ""}
    tokens = [t for t in tokens if t not in stop_tokens and len(t) > 1]

    return tokens

def parse_label_string(label_str: str) -> List[int]:
    """
    Parse label string representation to binary indicator vector (length 8).

    Accepts both formats:
      - "[1, 0, 1]"  — binary notation (legacy)
      - "['milk', 'wheat']" — allergen name notation

    Delegates to allergens_to_binary for the name-based format.
    If the literal evaluates to integers, it returns them directly.

    Args:
        label_str: String representation of labels

    Returns:
        List of 8 integers (0 or 1) indicating allergen presence
    """
    # Handle NaN/None values and empty lists
    if (label_str is None or
        (isinstance(label_str, float) and np.isnan(label_str)) or
        label_str == "[]"):
        return [0] * 8  # Allergen list default — 8 BIG8 allergens

    try:
        parsed = ast.literal_eval(label_str)
        # If parsed as list of ints, return directly
        if parsed and all(isinstance(x, int) for x in parsed):
            return [int(x) for x in parsed]
        # Otherwise delegate to name-based conversion
        return allergens_to_binary(label_str)
    except Exception:
        return [0] * 8

def get_allergen_list() -> List[str]:
    """Get list of allergen classes."""
    return ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]


def allergens_to_binary(allergen_list_str) -> List[int]:
    """
    Convert a list or string representation of allergen names to binary vector.

    Accepts both a string representation (e.g., "['milk', 'wheat']") and a
    direct Python list (e.g., ['milk', 'wheat']) for convenience.

    Args:
        allergen_list_str: List of allergen names, or string representation

    Returns:
        Binary vector indicating presence of each allergen
    """
    # Accept both string representations and direct lists
    if isinstance(allergen_list_str, str):
        try:
            allergen_list = ast.literal_eval(allergen_list_str)
        except:
            allergen_list = []
    elif isinstance(allergen_list_str, (list, tuple)):
        allergen_list = list(allergen_list_str)
    else:
        allergen_list = []
    ordered_allergens = get_allergen_list()
    binary = [0] * len(ordered_allergens)
    for allergen in allergen_list:
        if allergen in ordered_allergens:
            idx = ordered_allergens.index(allergen)
            binary[idx] = 1
    return binary


def extract_may_contain(text):
    """
    Extract allergens mentioned in "may contain" statements.

    Args:
        text: Ingredient text to search

    Returns:
        List of allergens found in may contain statement
    """
    if not isinstance(text, str):
        return []
    pattern = r'may contain\s*:?\s*([^\.]+)'
    match = re.search(pattern, text.lower())
    if match:
        candidates = match.group(1)
        found = []
        for allergen, keywords in BIG8.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', candidates):
                    found.append(allergen)
                    break
        return list(set(found))
    return []


def has_explicit_allergen_statement(text):
    """
    Check if text contains explicit allergen statement.
    
    Args:
        text: Ingredient text to check
        
    Returns:
        True if explicit allergen statement found
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r'contains\s*:+\s*\w+', text, re.IGNORECASE) or
                re.search(r'allergen information\s*:+\s*\w+', text, re.IGNORECASE))


def parse_traces_tags(tag_str):
    """
    Parse traces tags string to list of allergens.
    
    Args:
        tag_str: String representation of traces tags list
        
    Returns:
        List of mapped allergen keys
    """
    try:
        tags = ast.literal_eval(tag_str)
    except:
        return []
    mapped = []
    for tag in tags:
        if not tag.startswith('en:'):
            continue
        suffix = tag.split(':', 1)[1]
        mapped_key = OFFICIAL_MAP.get(suffix)
        if mapped_key:
            mapped.append(mapped_key)
        elif suffix == "tree-nuts":
            mapped.append("tree_nuts")
    return list(set(mapped))


def parse_official_tags(tag_str):
    """
    Parse official tags string to list of allergens.

    Handles both standard Python list syntax (comma-separated) and
    numpy-style space-separated format (e.g. "['en:milk' 'en:eggs']").

    Args:
        tag_str: String representation of official tags list

    Returns:
        List of mapped allergen keys
    """
    if not isinstance(tag_str, str) or tag_str in ('[]', 'nan', ''):
        return []

    # Extract all 'en:xxx' tags via regex — handles both comma-separated
    # and numpy-style space-separated formats without relying on literal_eval
    tags = re.findall(r"'([^']*)'", tag_str)
    if not tags:
        return []

    mapped = []
    for tag in tags:
        if not tag.startswith('en:'):
            continue
        suffix = tag.split(':', 1)[1]
        mapped_key = OFFICIAL_MAP.get(suffix)
        if mapped_key:
            mapped.append(mapped_key)
        elif suffix == "tree-nuts":
            mapped.append("tree_nuts")
    return list(set(mapped))


def apply_exemptions(detected_list, tokens):
    """
    Apply exemptions to detected allergens based on context.

    Handles cases where:
    - "soy lecithin" ≠ soy allergen (highly refined, protein removed)
    - "soybean oil" ≠ soy allergen (highly refined)
    - "coconut" is NOT an FDA Big-8 tree nut
    - "shea butter" is exempt from tree_nuts
    - "coconut oil/milk" is NOT a dairy product
    - "flour" or "bran" alone ≠ wheat allergen (context removed in BIG8,
      but kept as safety net for edge cases)
    - "lecithin" with no egg context ≠ egg allergen

    Args:
        detected_list: List of detected allergens
        tokens: List of ingredient tokens

    Returns:
        Filtered list of allergens after applying exemptions
    """
    text = " ".join(tokens).lower()
    detected_set = set(detected_list)

    exempt_config = {
        "soy": {
            # Highly refined soy products are generally exempt per FDA
            # because the allergenic protein is removed during processing
            "patterns": [
                r'\bsoy lecithin\b', r'\bsoybean oil\b', r'\bsoya oil\b',
                r'\bhydrolyzed soy protein\b', r'\bsoy flour\b', r'\bsoy fiber\b',
                r'\bsoy polysaccharide\b',
            ],
            "keep_if_also": [
                r'\bsoy protein\b', r'\btofu\b', r'\bmiso\b', r'\btempeh\b',
                r'\bsoy sauce\b', r'\btamari\b', r'\bedamame\b', r'\bnatto\b',
                r'\bsoy milk\b', r'\bsoybean\b', r'\bsoy beans\b',
                r'\btextured vegetable protein\b', r'\btvp\b',
            ]
        },
        "wheat": {
            # "wheat starch" is highly processed and may have negligible gluten
            # "cereal" alone is vague (could be oats, rice, etc.)
            # "bran" alone could be oat bran, rice bran, etc.
            # "flour" alone is too broad (removed from BIG8, safety net here)
            "patterns": [
                r'\bflour\b', r'\bcereal\b', r'\bbran\b', r'\bwheat starch\b',
            ],
            "keep_if_also": [
                r'\bwheat\b', r'\bgluten\b', r'\bspelt\b', r'\bdurum\b',
                r'\bsemolina\b', r'\bbulgur\b', r'\bfarina\b',
            ]
        },
        "tree_nuts": {
            # FDA exemptions: shea nuts (shea butter) and coconut are NOT
            # considered major tree nut allergens
            "patterns": [
                r'\bshea butter\b', r'\bshea nut\b',
                r'\bcoconut oil\b', r'\bcoconut milk\b', r'\bcoconut flour\b',
                r'\bcoconut cream\b', r'\bcoconut water\b',
                # Water chestnut is a vegetable, not a tree nut
                r'\bwater chestnut\b', r'\bwater chestnuts\b',
            ],
            "keep_if_also": [
                r'\b(?:almond|cashew|walnut|pecan|hazelnut|pistachio|macadamia|'
                r'brazil nut|pine nut|chestnut|filbert|marzipan|praline)\b',
                r'\bnut paste\b', r'\bnut butter\b', r'\bnut meal\b', r'\bnut oil\b',
                r'\bnut flour\b',
            ]
        },
        "eggs": {
            # Standalone "lecithin" without egg context is unlikely to be egg-derived
            # (most lecithin is soy or sunflower-based)
            "patterns": [
                r'\blecithin\b',
            ],
            "keep_if_also": [
                r'\begg\b', r'\begg yolk\b', r'\begg white\b', r'\balbumin\b',
                r'\bovalbumin\b', r'\bovomucoid\b', r'\bovotransferrin\b',
                r'\blysozyme\b', r'\bmeringue\b', r'\bmayonnaise\b',
            ]
        }
    }

    for allergen, config in exempt_config.items():
        if allergen not in detected_set:
            continue

        # Collect all exemption match spans so we can detect over-matching
        # by keep_if_also patterns that happen to match inside exemption text
        # (e.g. \bchestnut\b inside "water chestnut", \bnut butter\b inside "shea nut butter")
        exempt_spans = set()
        exempt_found = False
        for pat in config["patterns"]:
            for m in re.finditer(pat, text):
                exempt_found = True
                for pos in range(m.start(), m.end()):
                    exempt_spans.add(pos)
        if not exempt_found:
            continue

        # A keep_if_also match only counts if it falls OUTSIDE all exemption spans.
        # This prevents "water chestnut" from being re-classified as tree_nuts
        # via the \bchestnut\b keep_if_also pattern matching within the exempt phrase.
        keep_match = False
        for pat in config.get("keep_if_also", []):
            for m in re.finditer(pat, text):
                # Check if this match overlaps with any exemption span
                overlap = any(pos in exempt_spans for pos in range(m.start(), m.end()))
                if not overlap:
                    keep_match = True
                    break
            if keep_match:
                break

        if exempt_found and not keep_match:
            detected_set.discard(allergen)

    return list(detected_set)

def combine_allergen_labels(detected, official, traces, may_contain):
    """
    Combine different allergen detection sources using standardized logic.

    Args:
        detected: List of detected allergens
        official: List of official allergen tags
        traces: List of traces allergens
        may_contain: List of may-contain allergens

    Returns:
        Dictionary with combined allergen sets:
            - detected_only: Detected allergens not in official tags
            - detected_or_official: Union of detected and official allergens
            - consensus: Intersection of detected and official allergens
            - detected_with_traces: Detected allergens with traces
            - detected_with_may_contain: Detected allergens with may-contain
            - traces_only: Allergens only in traces
            - may_contain_only: Allergens only in may-contain
    """
    detected_set = set(detected)
    official_set = set(official)
    traces_set = set(traces)
    may_contain_set = set(may_contain)

    return {
        "detected_only": sorted(detected_set - official_set - traces_set - may_contain_set),
        "detected_or_official": sorted(detected_set | official_set),
        "consensus": sorted(detected_set & official_set),
        "detected_with_traces": sorted(detected_set | traces_set),
        "detected_with_may_contain": sorted(detected_set | may_contain_set),
        "traces_only": sorted(traces_set - detected_set - official_set - may_contain_set),
        "may_contain_only": sorted(may_contain_set - detected_set - official_set - traces_set)
    }

def detect_coconut_improved(tokens):
    """
    Detect coconut allergen using improved detection logic.

    Args:
        tokens: List of ingredient tokens

    Returns:
        List containing "coconut" if detected, empty list otherwise
    """
    if not isinstance(tokens, list):
        return []

    text = " ".join(tokens).lower()
    coconut_keywords = ["coconut", "coconut oil", "coconut milk", "coconut flour"]
    for kw in coconut_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            return ["coconut"]
    return []
