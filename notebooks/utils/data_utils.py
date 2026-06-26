"""
Data utilities for food label transparency project.

This module provides data loading, preprocessing, and split utilities for the food label transparency project.
It includes functions for loading CSV datasets, creating stratified splits for multi-label classification,
and augmenting datasets to handle class imbalance.

Key Features:
- Load various dataset formats from standard data directories
- Create stratified train/val/test splits for multi-label data
- Augment datasets with synonym replacement and negative examples
- Convert label lists to binary indicator matrices
- Safe parsing of label strings with error handling
- Metadata management for models and configurations

Usage Examples:
    >>> from utils.data_utils import load_labeled_data, create_stratified_splits, augment_dataframe
    >>> df = load_labeled_data("../data/final/labeled_dataset_enhanced.csv")
    >>> train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = create_stratified_splits(
    ...     texts, labels, train_size=0.7, val_size=0.15, test_size=0.15
    ... )
    >>> augmented_df = augment_dataframe(train_df, num_augmented=2)
"""

# Re-export commonly used functions
__all__ = [
    'load_extracted_data',
    'load_cleaned_data',
    'load_labeled_data',
    'parse_label_column',
    'create_stratified_splits',
    'augment_dataframe',
    'labels_to_binary',
    'safe_parse_list',
    'save_metadata',
    'load_metadata',
    'get_data_directories',
    'serialize_list_for_csv',
    'normalize_list_column',
    'normalize_dict_column',
    'validate_label_consistency',
    'prepare_dataframe_for_save',
]

import pandas as pd
import numpy as np
import ast
import os
import re
from typing import List, Tuple, Dict, Any, Optional
import json
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def load_extracted_data(filepath: str = "../data/interim/extracted_raw.csv") -> pd.DataFrame:
    """
    Load extracted raw data from CSV.

    Args:
        filepath: Path to the extracted data CSV

    Returns:
        DataFrame containing extracted data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Extracted data not found at {filepath}")

    return pd.read_csv(filepath)


def load_cleaned_data(filepath: str = "../data/processed/cleaned_dataset.csv") -> pd.DataFrame:
    """
    Load cleaned data from CSV.

    Args:
        filepath: Path to the cleaned data CSV

    Returns:
        DataFrame containing cleaned data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data not found at {filepath}")

    return pd.read_csv(filepath)


def load_labeled_data(filepath: str = "../data/final/labeled_dataset.csv") -> pd.DataFrame:
    """
    Load labeled data from CSV.

    Args:
        filepath: Path to the labeled data CSV

    Returns:
        DataFrame containing labeled data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Labeled data not found at {filepath}")

    return pd.read_csv(filepath)


def parse_label_column(df: pd.DataFrame, column_name: str = "detected_allergens",
                      allergen_list: List[str] = None) -> pd.DataFrame:
    """
    Parse a column containing string representations of label lists into binary columns.

    Args:
        df: Input DataFrame
        column_name: Name of column containing label strings
        allergen_list: List of allergen names (in order)

    Returns:
        DataFrame with added binary columns for each allergen
    """
    if allergen_list is None:
        from .text_processing import get_allergen_list as _gal
        allergen_list = _gal()

    # Make a copy to avoid modifying original
    result_df = df.copy()

    # Parse the label column safely
    def _safe_parse_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x) or x == "[]":
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    # Pre-parse once to avoid repeated eval calls
    parsed = result_df[column_name].apply(_safe_parse_list)

    # Create binary columns
    for i, allergen in enumerate(allergen_list):
        result_df[allergen] = parsed.apply(lambda lst: 1 if allergen in lst else 0)

    return result_df


def create_stratified_splits(texts: List[str],
                           labels: np.ndarray,
                           train_size: float = 0.7,
                           val_size: float = 0.15,
                           test_size: float = 0.15,
                           random_state: int = 42) -> Tuple[List[str], List[str], List[str],
                                                             np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/validation/test splits for multi-label data.

    Args:
        texts: List of input texts
        labels: Numpy array of labels (n_samples, n_classes)
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # Validate sizes
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"

    # First split: train vs (val + test)
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size + test_size,
        random_state=random_state
    )

    train_idx, temp_idx = next(msss1.split(texts, labels))

    # Second split: val vs test
    relative_val_size = val_size / (val_size + test_size)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - relative_val_size,  # test size relative to remaining
        random_state=random_state
    )

    val_idx, test_idx = next(msss2.split(
        np.array(texts)[temp_idx],
        labels[temp_idx]
    ))

    # Convert indices
    train_texts = [texts[i] for i in train_idx]
    temp_texts = np.array(texts)[temp_idx]
    val_texts = [temp_texts[i] for i in val_idx]
    test_texts = [temp_texts[i] for i in test_idx]

    train_labels = labels[train_idx]
    val_labels = labels[temp_idx][val_idx]
    test_labels = labels[temp_idx][test_idx]

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def augment_dataframe(df: pd.DataFrame, num_augmented: int = 2,
                     ALLERGENS: List[str] = None,
                     SYNONYMS: Dict[str, List[str]] = None,
                     ALLERGEN_KEYWORDS: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Augment dataframe with synonym replacement and negative examples.

    Args:
        df: Input DataFrame with 'text' and 'labels' columns
        num_augmented: Number of augmented examples to create per original example
        ALLERGENS: List of allergen class names
        SYNONYMS: Dictionary mapping allergens to synonym lists
        ALLERGEN_KEYWORDS: Dictionary mapping allergens to keyword lists for removal

    Returns:
        DataFrame containing augmented examples (original not included)
    """
    if ALLERGENS is None:
        from .text_processing import get_allergen_list as _gal
        ALLERGENS = _gal()

    if SYNONYMS is None:
        SYNONYMS = {
            "milk": ["cow's milk", "dairy milk", "whole milk", "low-fat milk"],
            "eggs": ["hen's eggs", "whole eggs", "egg product",
                     "egg white powder", "dried egg", "egg solids", "pasteurized eggs"],
            "peanuts": ["groundnuts", "peanut kernels",
                        "monkey nuts", "peanut flour", "peanut protein"],
            "tree_nuts": ["almonds", "cashews", "walnuts", "hazelnuts", "pecans",
                          "pistachio", "macadamia", "pine nuts", "brazil nuts",
                          "chestnuts", "marzipan", "praline"],
            "soy": ["soya", "soybean", "edamame",
                    "soy protein isolate", "soy flour", "soy milk"],
            "wheat": ["whole wheat", "wheat grain",
                      "wheat germ", "wheat starch", "triticale"],
            "fish": ["fish meat", "white fish",
                     "tuna", "salmon", "sardine", "anchovy", "mackerel",
                     "cod", "pollock", "trout", "surimi", "roe", "tuna flakes"],
            "shellfish": ["shrimp", "prawn", "crab", "lobster",
                          "crayfish", "krill", "clam", "mussel", "oyster",
                          "scallop", "squid", "crawfish", "abalone", "conch", "cockle"]
        }

    if ALLERGEN_KEYWORDS is None:
        from .text_processing import BIG8 as _BIG8
        ALLERGEN_KEYWORDS = {k: v[:5] for k, v in _BIG8.items()}

    def synonym_replacement(text: str, p: float = 0.3) -> str:
        tokens = re.split(r'(\W+)', text)
        new_tokens = []
        for t in tokens:
            t_lower = t.strip().lower()
            if t_lower in SYNONYMS and np.random.random() < p:
                new_tokens.append(np.random.choice(SYNONYMS[t_lower]))
            else:
                new_tokens.append(t)
        return ''.join(new_tokens)

    def remove_keywords(text: str, keywords: List[str]) -> str:
        pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'^\s*,\s*', '', cleaned)
        cleaned = re.sub(r'\s*,\s*$', '', cleaned)
        return cleaned.strip()

    def create_negative_example(text: str, label_vector: List[int]) -> Optional[Tuple[str, List[int]]]:
        present = [ALLERGENS[i] for i, v in enumerate(label_vector) if v == 1]
        if not present:
            return None
        chosen_idx = np.random.choice([i for i, v in enumerate(label_vector) if v == 1])
        chosen = ALLERGENS[chosen_idx]
        new_text = remove_keywords(text, ALLERGEN_KEYWORDS[chosen])
        if not new_text:
            return None
        new_labels = label_vector.copy()
        new_labels[ALLERGENS.index(chosen)] = 0
        return (new_text, new_labels)

    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(num_augmented):
            method = np.random.choice(['synonym', 'negative'])
            if method == 'synonym':
                new_text = synonym_replacement(row['text'])
                new_labels = row['labels']
            else:  # negative
                result = create_negative_example(row['text'], row['labels'])
                if result is None:
                    continue
                new_text, new_labels = result
            augmented_rows.append({
                'text': new_text,
                'labels': new_labels,
                'ingredients_cleaned': new_text,  # keep same format
                'detected_allergens': [ALLERGENS[i] for i, v in enumerate(new_labels) if v == 1]
            })

    return pd.DataFrame(augmented_rows)


def augment_targeted(
    df: pd.DataFrame,
    target_allergens: Dict[str, float],
    ALLERGENS: List[str] = None,
    SYNONYMS: Dict[str, List[str]] = None,
    ALLERGEN_KEYWORDS: Dict[str, List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Augment a dataset by oversampling rows containing specific rare allergens.

    For each target allergen, finds rows where that allergen is present and
    creates multiple variant copies via synonym replacement. The augmentation
    factor determines how many extra copies per original row.

    Args:
        df: Input DataFrame with 'text' (or 'ingredients_cleaned') and 'labels' columns.
        target_allergens: Dict mapping allergen names to multipliers.
            E.g., {"tree_nuts": 5} means 5 copies per tree_nuts-positive row.
        ALLERGENS: List of allergen class names (loaded from text_processing if None).
        SYNONYMS: Dictionary mapping allergens to synonym lists.
        ALLERGEN_KEYWORDS: Dictionary mapping allergens to keyword lists.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with original rows plus augmented copies.
    """
    if ALLERGENS is None:
        from .text_processing import get_allergen_list as _gal
        ALLERGENS = _gal()

    if SYNONYMS is None:
        SYNONYMS = {
            "milk": ["cow's milk", "dairy milk", "whole milk", "low-fat milk"],
            "eggs": ["hen's eggs", "whole eggs", "egg product",
                     "egg white powder", "dried egg", "egg solids", "pasteurized eggs"],
            "peanuts": ["groundnuts", "peanut kernels",
                        "monkey nuts", "peanut flour", "peanut protein"],
            "tree_nuts": ["almonds", "cashews", "walnuts", "hazelnuts", "pecans",
                          "pistachio", "macadamia", "pine nuts", "brazil nuts",
                          "chestnuts", "marzipan", "praline"],
            "soy": ["soya", "soybean", "edamame",
                    "soy protein isolate", "soy flour", "soy milk"],
            "wheat": ["whole wheat", "wheat grain",
                      "wheat germ", "wheat starch", "triticale"],
            "fish": ["fish meat", "white fish",
                     "tuna", "salmon", "sardine", "anchovy", "mackerel",
                     "cod", "pollock", "trout", "surimi", "roe", "tuna flakes"],
            "shellfish": ["shrimp", "prawn", "crab", "lobster",
                          "crayfish", "krill", "clam", "mussel", "oyster",
                          "scallop", "squid", "crawfish", "abalone", "conch", "cockle"]
        }

    if ALLERGEN_KEYWORDS is None:
        from .text_processing import BIG8 as _BIG8
        ALLERGEN_KEYWORDS = {k: v[:5] for k, v in _BIG8.items()}

    np.random.seed(random_state)

    # Determine which column holds the text
    text_col = 'text' if 'text' in df.columns else 'ingredients_cleaned'

    def synonym_replacement(text: str, p: float = 0.3) -> str:
        tokens = re.split(r'(\W+)', text)
        new_tokens = []
        for t in tokens:
            t_lower = t.strip().lower()
            if t_lower in SYNONYMS and np.random.random() < p:
                new_tokens.append(np.random.choice(SYNONYMS[t_lower]))
            else:
                new_tokens.append(t)
        return ''.join(new_tokens)

    augmented_rows = []
    for allergen, multiplier in target_allergens.items():
        if allergen not in ALLERGENS:
            continue
        allergen_idx = ALLERGENS.index(allergen)
        # Find rows where this allergen is present
        mask = df['labels'].apply(lambda lbl: isinstance(lbl, (list, np.ndarray)) and len(lbl) > allergen_idx and lbl[allergen_idx] == 1)
        if not mask.any():
            continue
        target_rows = df[mask]
        num_copies = max(1, int(multiplier))
        for _, row in target_rows.iterrows():
            for _ in range(num_copies):
                new_text = synonym_replacement(row[text_col])
                augmented_rows.append({
                    text_col: new_text,
                    'ingredients_cleaned': new_text,
                    'labels': row['labels'],
                    'detected_allergens': [ALLERGENS[i] for i, v in enumerate(row['labels']) if v == 1],
                })

    augmented_df = pd.DataFrame(augmented_rows)
    if len(augmented_df) == 0:
        return df.copy()

    # Concatenate original + augmented
    result = pd.concat([df, augmented_df], ignore_index=True)
    return result


def labels_to_binary(label_list: List[List[str]], classes: List[str]) -> np.ndarray:
    """
    Convert list of label strings to binary indicator matrix.

    Args:
        label_list: List of lists containing string labels
        classes: List of class names in order

    Returns:
        Binary numpy array of shape (len(label_list), len(classes))
    """
    binary = np.zeros((len(label_list), len(classes)))
    for i, labels in enumerate(label_list):
        for label in labels:
            if label in classes:
                binary[i][classes.index(label)] = 1
    return binary


def safe_parse_list(x):
    """
    Safely parse string representation of list.

    Args:
        x: String to parse or already a list

    Returns:
        Parsed list or empty list if parsing fails
    """
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def save_metadata(metadata: Dict[str, Any], filepath: str):
    """
    Save metadata to JSON file.

    Args:
        metadata: Dictionary to save
        filepath: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary containing loaded metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def get_data_directories() -> Dict[str, str]:
    """
    Get standard data directory paths for the project.

    Returns:
        Dictionary mapping directory names to paths
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        "base": base_dir,
        "raw": os.path.join(base_dir, "data", "raw"),
        "interim": os.path.join(base_dir, "data", "interim"),
        "processed": os.path.join(base_dir, "data", "processed"),
        "final": os.path.join(base_dir, "data", "final"),
        "models": os.path.join(base_dir, "models"),
        "configs": os.path.join(base_dir, "configs"),
        "outputs": os.path.join(base_dir, "outputs"),
        "notebooks": os.path.join(base_dir, "notebooks")
    }


# ──────────────────────────────────────────────
# Label format normalization utilities
# ──────────────────────────────────────────────
# These functions handle the common task of converting between
# different label representation formats:
#   - Python list objects: ['milk', 'wheat']
#   - String literals:    "['milk', 'wheat']"  (from CSV save/load)
#   - Binary vectors:     [1, 0, 1, 0, 0, 1, 0, 0]
#   - JSON strings:       '["milk", "wheat"]'
#
# Using these consistently prevents the "mixed types" bugs where
# a column contains both string and list objects.

def serialize_list_for_csv(lst: list) -> str:
    """
    Serialize a list to a stable string format for CSV storage.
    Uses JSON format ('["milk", "wheat"]') rather than Python repr
    ("['milk', 'wheat']") because JSON is more portable and consistent.

    Args:
        lst: Python list to serialize (e.g., ['milk', 'wheat'])

    Returns:
        JSON-formatted string (e.g., '["milk", "wheat"]')
    """
    return json.dumps(lst)


def normalize_list_column(series: pd.Series) -> pd.Series:
    """
    Normalize a DataFrame column that may contain mixed list/string types.

    Ensures every entry is a Python list object, parsing string representations
    as needed. Handles:
    - Already a list: returned as-is
    - JSON string:     '["milk", "wheat"]' → ['milk', 'wheat']
    - Python repr:     "['milk', 'wheat']" → ['milk', 'wheat']
    - NaN/None:        → []

    Args:
        series: Pandas Series potentially containing mixed types

    Returns:
        Series where every entry is a Python list
    """
    def _normalize(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            # Try JSON first (cleaner format)
            if val.startswith('['):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Fall back to ast.literal_eval for Python repr
                try:
                    result = ast.literal_eval(val)
                    if isinstance(result, list):
                        return result
                except (SyntaxError, ValueError):
                    pass
            # Single word that looks like a raw value?
            return []
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        return []

    return series.apply(_normalize)


def normalize_dict_column(series: pd.Series) -> pd.Series:
    """
    Normalize a DataFrame column that may contain mixed dict/string types.

    Similar to normalize_list_column but for dictionary values.
    Handles serializations via both JSON and Python repr.

    Args:
        series: Pandas Series potentially containing mixed types

    Returns:
        Series where every entry is a Python dict
    """
    def _normalize(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                result = ast.literal_eval(val)
                if isinstance(result, dict):
                    return result
            except (SyntaxError, ValueError):
                pass
            return {}
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {}
        return {}

    return series.apply(_normalize)


def validate_label_consistency(df: pd.DataFrame, label_columns: List[str] = None,
                               expected_length: Optional[int] = None) -> List[str]:
    """
    Validate that label columns in a DataFrame are consistent.

    Checks:
    - No NaN values in label columns
    - Every entry has the expected length (only checked if expected_length is set)
    - Every entry is a list type

    Args:
        df: DataFrame to validate
        label_columns: List of column names to check (if None, checks all object columns)
        expected_length: Expected length of each label vector (None = skip length check)

    Returns:
        List of validation issues found (empty if all pass)
    """
    issues = []
    if label_columns is None:
        label_columns = [col for col in df.columns
                        if any(name in col.lower()
                              for name in ['allergen', 'label', 'detected', 'combined'])]

    for col in label_columns:
        if col not in df.columns:
            continue
        for idx, val in enumerate(df[col]):
            if isinstance(val, float) and np.isnan(val):
                issues.append(f"{col}[{idx}]: NaN value")
            elif isinstance(val, str):
                # String means it wasn't parsed — may be OK but note it
                if val.startswith('['):
                    issues.append(f"{col}[{idx}]: string repr (needs parsing): {val[:50]}")
            elif isinstance(val, (list, tuple)):
                if expected_length is not None and len(val) != expected_length:
                    issues.append(f"{col}[{idx}]: length {len(val)}, expected {expected_length}")
            # dict is OK for combined columns
            elif not isinstance(val, dict):
                issues.append(f"{col}[{idx}]: unexpected type {type(val).__name__}")

    return issues


# ──────────────────────────────────────────────
# DataFrame serialization helpers
# ──────────────────────────────────────────────

# Columns in labeled datasets that contain list/dict values
LIST_LIKE_COLUMNS = {
    'detected_allergens', 'official_allergens_mapped', 'traces_allergens_mapped',
    'may_contain_allergens', 'combined_allergens', 'all_allergens',
    'detected_only', 'detected_or_official', 'consensus',
    'detected_with_traces', 'detected_with_may_contain',
}
DICT_LIKE_COLUMNS = {'combined_allergens'}


def prepare_dataframe_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all list/dict columns in a DataFrame for reliable CSV serialization.

    Converts every cell in known list-like columns to JSON-formatted strings so
    that pandas to_csv produces consistent, parseable output.

    Args:
        df: Input DataFrame (may contain mixed list/string types)

    Returns:
        DataFrame with all list/dict columns serialized to JSON strings
    """
    result = df.copy()
    for col in result.columns:
        col_lower = col.lower()
        # Check if this column likely contains list values
        if col in DICT_LIKE_COLUMNS or col_lower in DICT_LIKE_COLUMNS:
            result[col] = result[col].apply(
                lambda x: json.dumps(x) if not isinstance(x, str) and x is not None
                else x if isinstance(x, str) else '{}'
            )
        elif col in LIST_LIKE_COLUMNS or col_lower in LIST_LIKE_COLUMNS:
            result[col] = result[col].apply(
                lambda x: json.dumps(x) if not isinstance(x, str) and x is not None
                else x if isinstance(x, str) else '[]'
            )
    return result