"""
Text processing utilities for food label transparency project.

This module consolidates all text cleaning, allergen rule matching, and
label parsing logic used across notebooks. It serves as the single source
of truth — notebooks should import from here rather than redefining logic.

Key Features:
- Enhanced allergen detection with BIG8 comprehensive database
- HTML tag removal and text cleaning
- Tokenization and text splitting utilities
- Label parsing for multiple allergen types
- Exemption handling for refined products
- Rule-based allergen detection with negation handling
- Utility functions for food label processing

Usage Examples:
    >>> from utils.text_processing import detect_allergens_rule_based, clean_html, get_allergen_list
    >>> text = "milk, sugar, wheat flour"
    >>> allergens = detect_allergens_rule_based(text)
    >>> cleaned = clean_html("<div>milk - free</div>")
    >>> allergen_list = get_allergen_list()
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
BIG8 = {
    "milk": [
        "milk", "whey", "casein", "caseinate", "butter", "cheese",
        "lactose", "cream", "ghee", "buttermilk", "milk solids", "skim milk",
        "whole milk", "milk powder", "nonfat milk", "evaporated milk",
        "condensed milk", "powdered milk", "curds", "yogurt",
        "kefir", "lactalbumin", "lactoglobulin", "butterfat", "custard",
        "whey protein"
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
        "almond", "cashew", "walnut", "pecan", "hazelnut", "filbert",
        "macadamia", "pistachio", "brazil nut", "chestnut",
        "pine nut", "marzipan", "praline",
        "nut paste", "nut butter", "nut meal", "nut oil", "shea nut"
    ],
    "soy": [
        "soy", "soya", "soybean", "soybeans", "edamame", "natto",
        "soy lecithin", "lecithin",
        "soy protein", "textured vegetable protein",
        "tvp", "tofu", "miso", "tempeh",
        "soy sauce", "tamari", "soy milk", "soybean oil", "soy flour"
    ],
    "wheat": [
        "wheat", "whole wheat", "wheat flour", "spelt", "kamut", "triticale", "durum",
        "flour", "gluten", "semolina",
        "farina", "bran", "bulgur",
        "bread", "breadcrumbs", "pasta", "noodles",
        "wheat germ", "wheat starch", "couscous", "einkorn"
    ],
    "fish": [
        "fish", "tuna", "salmon", "sardine",
        "anchovy", "mackerel", "cod",
        "fish sauce", "fish oil",
        "surimi", "pollock", "trout", "roe",
        "fish gelatin", "isnglass", "caviar", "fish meal",
        "omega-3 from fish"
    ],
    "shellfish": [
        "shrimp", "prawn", "crab", "lobster",
        "crayfish", "krill", "clam", "mussel", "oyster", "scallop", "crawfish", "langoustine",
        "shellfish extract",
        "squid", "crab paste", "abalone", "conch", "cockle", "whelk",
        "crab meat", "lobster paste"
    ],
}

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
    "shellfish": "shellfish"
}

# Pre-compile regex patterns for each allergen
COMPILED_RULES = {}
for allergen, keywords in ALLERGEN_RULES.items():
    pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
    COMPILED_RULES[allergen] = re.compile(pattern, re.IGNORECASE)

# Negation pattern (enhanced to catch missed constructions while reducing false positives)
NEGATION_PATTERN = re.compile(
    r'\b(no|not|free|without|minus|low\s+in|none)\s+(?:\w+\s+)?\w+\b|\b(does not contain|free from)\s+\w+\b|\b\w+-\w+free\b',
    re.IGNORECASE
)

def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', str(text))

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: lowercase and strip."""
    return text.lower().strip()

def rule_match(text: str, allergen: str) -> bool:
    """
    Check if allergen is present in text using rule-based matching.

    Args:
        text: Input text to search
        allergen: Allergen to check for (must be key in ALLERGEN_RULES)

    Returns:
        True if allergen found and not negated, False otherwise
    """
    if allergen not in COMPILED_RULES:
        raise ValueError(f"Unknown allergen: {allergen}. Must be one of {list(ALLERGEN_RULES.keys())}")

    text_lower = text.lower()
    if COMPILED_RULES[allergen].search(text_lower):
        # Define common allergen stem variations to handle different word forms
        allergen_stems = {
            "milk": ["milk"],
            "eggs": ["egg", "eggs"],
            "peanuts": ["peanut", "peanuts"],
            "tree_nuts": ["tree nut", "tree nuts", "nut", "nuts"],
            "soy": ["soy", "soya"],
            "wheat": ["wheat"],
            "fish": ["fish"],
            "shellfish": ["shellfish", "shellfish"]
        }

        # Get possible stems for this allergen (default to just the allergen itself)
        stems = allergen_stems.get(allergen, [allergen])

        # Check for negation patterns that are specific to this allergen
        for stem in stems:
            # Pattern 1: negation words followed by allergen stem (e.g., "no milk", "free soy")
            # Allow for optional words between negation and allergen (like "not containing")
            negation_before = re.compile(
                r'\b(no|not|free|without|minus|low\s+in|none)\s+(?:\w+\s+)?' + re.escape(stem) + r'\b',
                re.IGNORECASE
            )
            # Pattern 2: "does not contain" or "free from" followed by allergen stem
            negation_phrases = re.compile(
                r'\b(does not contain|free from)\s+' + re.escape(stem) + r'\b',
                re.IGNORECASE
            )
            # Pattern 3: allergen stem followed by known negation suffix (e.g., "milk-free", "peanut-free")
            # WARNING: Only match specific negation suffixes to avoid false negatives
            # on hyphenated compounds like "milk-chocolate", "peanut-butter", "egg-white"
            negation_after = re.compile(
                r'\b' + re.escape(stem) + r'(?:[a-z]+)?-(?:free|less|absent|avoid|removed|excluded|none)\b',
                re.IGNORECASE
            )

            # Pattern 4: allergen stem followed by negation word (e.g., "wheat free", "egg free")
            negation_after_space = re.compile(
                r'\b' + re.escape(stem) + r'\s+(no|not|free|without|minus|low\s+in|none)\b',
                re.IGNORECASE
            )

            if (negation_before.search(text_lower) or
                negation_phrases.search(text_lower) or
                negation_after.search(text_lower) or
                negation_after_space.search(text_lower)):
                return False

        return True
    return False

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


def allergens_to_binary(allergen_list_str: str) -> List[int]:
    """
    Convert a string representation of a list of allergen names to binary vector.

    Args:
        allergen_list_str: String representation of a list of allergen names (e.g., "['milk', 'wheat']")

    Returns:
        Binary vector indicating presence of each allergen
    """
    # Parse the string to a Python list
    try:
        allergen_list = ast.literal_eval(allergen_list_str)
    except:
        allergen_list = []
    # Get the ordered allergen list
    ordered_allergens = get_allergen_list()
    # Initialize binary vector
    binary = [0] * len(ordered_allergens)
    # Set to 1 if allergen is present
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
    
    Args:
        tag_str: String representation of official tags list
        
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


def apply_exemptions(detected_list, tokens):
    """
    Apply exemptions to detected allergens based on context.

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
            "patterns": [r'\bsoy lecithin\b', r'\bsoybean oil\b', r'\bsoya oil\b', r'\bhydrolyzed soy protein\b'],
            "keep_if_also": [r'\bsoy protein\b', r'\btofu\b', r'\bmiso\b', r'\btempeh\b', r'\bsoy sauce\b']
        },
        "wheat": {
            "patterns": [r'\bflour\b', r'\bcereal\b', r'\bbran\b', r'\bwheat starch\b'],
            "keep_if_also": [r'\bwheat\b', r'\bgluten\b', r'\bspelt\b', r'\bdurum\b', r'\bsemolina\b']
        },
        "coconut": {
            "patterns": [r'\bcoconut oil\b', r'\bcoconut milk\b', r'\bcoconut flour\b'],
            "keep_if_also": [r'\bcoconut\b(?!\s+(?:oil|milk|flour))']
        },
        "tree_nuts": {
            "patterns": [r'\bshea butter\b', r'\bcoconut oil\b', r'\bcoconut milk\b'],
            "keep_if_also": [r'\b(?:almond|cashew|walnut|pecan|hazelnut|pistachio|macadamia|brazil nut|pine nut|chestnut)\b']
        },
        "eggs": {
            "patterns": [r'\blecithin\b'],  # egg lecithin is rare; if no other egg word, skip
            "keep_if_also": [r'\begg\b', r'\begg yolk\b', r'\begg white\b', r'\balbumin\b']
        }
    }

    for allergen, config in exempt_config.items():
        if allergen not in detected_set:
            continue
        exempt_match = any(re.search(pat, text) for pat in config["patterns"])
        if not exempt_match:
            continue
        keep_match = any(re.search(pat, text) for pat in config.get("keep_if_also", []))
        if exempt_match and not keep_match:
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
