"""
Evaluation utilities for food label transparency project.

This module provides comprehensive evaluation metrics and reporting functions for the food label transparency project.
It includes functions for computing classification metrics, threshold optimization, and error analysis specifically
tailored for multi-label allergen classification tasks.

Key Features:
- Classification report generation with customizable formatting
- Per-class metric computation (precision, recall, F1)
- Optimal threshold finding for multi-label classifiers
- Error analysis with sample case extraction
- Confusion matrix plotting capabilities
- Binary label vector conversion

Usage Examples:
    >>> from utils.evaluation import (
    ...     print_classification_report,
    ...     find_best_thresholds,
    ...     error_analysis,
    ...     apply_thresholds
    ... )
    >>> # Generate classification report
    >>> print_classification_report(y_true, y_pred, target_names, prefix="Test Set")
    >>> # Find optimal thresholds
    >>> thresholds = find_best_thresholds(probabilities, labels)
    >>> # Analyze errors
    >>> error_indices = error_analysis(texts, y_true, y_pred, target_names)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, multilabel_confusion_matrix
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                              target_names: List[str],
                              prefix: str = "",
                              digits: int = 4) -> None:
    """
    Print formatted classification report.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: List of class names
        prefix: Prefix string to print before report
        digits: Number of decimal places for metrics
    """
    if prefix:
        print(f"=== {prefix} ===")
    report = classification_report(y_true, y_pred, target_names=target_names,
                                 zero_division=0, digits=digits)
    print(report)

    # Print key metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    print(f"Macro F1: {macro_f1:.{digits}f}")
    print(f"Micro F1: {micro_f1:.{digits}f}\n")


def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, step: float = 0.01) -> np.ndarray:
    """
    Find optimal thresholds for each class to maximize F1 score.
    Delegates to model_utils.find_best_thresholds (single source of truth).

    Args:
        probs: Predicted probabilities from model (n_samples, n_classes)
        labels: True binary labels (n_samples, n_classes)
        step: Step size for threshold search

    Returns:
        Array of optimal thresholds for each class
    """
    from .model_utils import find_best_thresholds as _find_best_thresholds
    return _find_best_thresholds(probs, labels, step)


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Apply threshold array to probability matrix to get binary predictions.
    Delegates to model_utils.apply_thresholds (single source of truth).

    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        thresholds: Thresholds for each class (n_classes,)

    Returns:
        Binary predictions array
    """
    from .model_utils import apply_thresholds as _apply_thresholds
    return _apply_thresholds(probs, thresholds)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1 for each class.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: List of class names

    Returns:
        Dictionary mapping class names to their metrics
    """
    # Compute per-class F1, precision, recall
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {}
    for i, class_name in enumerate(target_names):
        metrics[class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i])
        }

    return metrics


def print_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          target_names: List[str],
                          prefix: str = "Per-class metrics:") -> None:
    """
    Print per-class precision, recall, and F1 scores.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: List of class names
        prefix: Prefix string to print before metrics
    """
    print(f"\n{prefix}")
    metrics = compute_per_class_metrics(y_true, y_pred, target_names)
    for class_name, metric_vals in metrics.items():
        print(f"{class_name:12}  Prec: {metric_vals['precision']:.2%}  "
              f"Rec: {metric_vals['recall']:.2%}  F1: {metric_vals['f1']:.2%}")


def compute_multilabel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute multilabel confusion matrix.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Multilabel confusion matrix array
    """
    return multilabel_confusion_matrix(y_true, y_pred)


def plot_confusion_matrices(cm: np.ndarray, class_names: List[str],
                          figsize_per_plot: Tuple[int, int] = (3, 3)) -> None:
    """
    Plot confusion matrices for each class.

    Args:
        cm: Multilabel confusion matrix from sklearn.metrics.multilabel_confusion_matrix
        class_names: List of class names
        figsize_per_plot: Size of each subplot
    """
    n_classes = len(class_names)
    n_cols = min(n_classes, 5)
    n_rows = (n_classes + n_cols - 1) // n_cols  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0]*n_cols, figsize_per_plot[1]*n_rows))
    axes = axes.ravel()

    for i in range(n_classes):
        axes[i].imshow(cm[i], cmap='Blues')
        axes[i].set_title(class_names[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        plt.colorbar(axes[i].images[0], ax=axes[i])

    # Hide empty subplots if any
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def error_analysis(texts: List[str], y_true: np.ndarray, y_pred: np.ndarray,
                 target_names: List[str], max_errors: int = 5) -> List[int]:
    """
    Perform error analysis and return indices of misclassified samples.

    Args:
        texts: List of input texts
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: List of class names
        max_errors: Maximum number of errors to return

    Returns:
        List of indices of misclassified samples
    """
    errors = []
    for i in range(len(texts)):
        if not np.array_equal(y_true[i], y_pred[i]):
            errors.append(i)

    print(f"Total errors: {len(errors)}")

    if len(errors) > 0:
        print(f"\nFirst {min(max_errors, len(errors))} examples:")
        for i in range(min(max_errors, len(errors))):
            idx = errors[i]
            true_labels = [target_names[j] for j, v in enumerate(y_true[idx]) if v == 1]
            pred_labels = [target_names[j] for j, v in enumerate(y_pred[idx]) if v == 1]
            print(f"\nText: {texts[idx][:150]}...")
            print(f"True: {true_labels}")
            print(f"Pred: {pred_labels}")

    return errors[:max_errors]


def binary_to_label_list(binary_array: np.ndarray,
                        class_names: List[str]) -> List[List[str]]:
    """
    Convert binary predictions to list of label strings.

    Args:
        binary_array: Binary array of shape (n_samples, n_classes)
        class_names: List of class names

    Returns:
        List of lists containing predicted label strings
    """
    label_lists = []
    for row in binary_array:
        labels = [class_names[i] for i, v in enumerate(row) if v == 1]
        label_lists.append(labels)
    return label_lists

def jaccard(set1: set, set2: set) -> float:
    """
    Compute Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity (0-1)
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def evaluate_rule_vs_official(df: pd.DataFrame, allergen_list: List[str] = None,
                              prefix: str = "Rule vs Official Tags") -> Dict[str, Any]:
    """
    Evaluate rule-based allergen detection against official OFF allergen tags.

    Compares the ``detected_allergens`` column (rule-based) against the
    ``official_allergens_mapped`` column (parsed from Open Food Facts tags).
    Reports exact agreement, per-class precision/recall/F1, and aggregate metrics.

    Args:
        df: DataFrame with at least ``detected_allergens``,
            ``official_allergens_mapped``, and optionally ``ingredients_text_en``
            columns.  The two allergen columns should contain lists of string
            keys (e.g. ``["milk", "wheat"]``).
        allergen_list: Ordered list of all allergen keys.  If not provided,
            inferred from the union of detected and official values.
        prefix: Label printed before the evaluation report.

    Returns:
        Dictionary with keys:
        - ``exact_agreement``: fraction of rows with identical sets
        - ``subset_agreement``: fraction where detected ⊆ official
        - ``superset_agreement``: fraction where detected ⊇ official
        - ``avg_jaccard``: mean Jaccard similarity across all rows
        - ``per_class``: dict mapping allergen → {precision, recall, f1}
        - ``n_samples``: number of rows evaluated
    """
    # Deduplicate columns (rename can create duplicates if the target
    # column name already exists in the DataFrame)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Normalize string-representation columns to actual lists
    for col in ["detected_allergens", "official_allergens_mapped"]:
        if col in df.columns:
            from .data_utils import normalize_list_column
            df[col] = normalize_list_column(df[col])

    if allergen_list is None:
        all_vals = set()
        for row in df["detected_allergens"]:
            if isinstance(row, list):
                all_vals.update(row)
        for row in df["official_allergens_mapped"]:
            if isinstance(row, list):
                all_vals.update(row)
        allergen_list = sorted(all_vals)

    # Build binary matrices
    def labels_to_binary(label_lists, classes):
        n = len(label_lists)
        m = np.zeros((n, len(classes)), dtype=int)
        for i, lst in enumerate(label_lists):
            if isinstance(lst, list):
                for item in lst:
                    if item in classes:
                        m[i, classes.index(item)] = 1
        return m

    y_true = labels_to_binary(df["official_allergens_mapped"].tolist(), allergen_list)
    y_pred = labels_to_binary(df["detected_allergens"].tolist(), allergen_list)

    # Exact agreement metrics
    detected_sets = df["detected_allergens"].apply(lambda x: set(x) if isinstance(x, list) else set())
    official_sets = df["official_allergens_mapped"].apply(lambda x: set(x) if isinstance(x, list) else set())

    exact_agreement = (detected_sets == official_sets).mean()
    subset_agreement = (detected_sets <= official_sets).mean()
    superset_agreement = (detected_sets >= official_sets).mean()
    avg_jac = df.apply(
        lambda row: jaccard(
            set(row["detected_allergens"]) if isinstance(row["detected_allergens"], list) else set(),
            set(row["official_allergens_mapped"]) if isinstance(row["official_allergens_mapped"], list) else set()
        ), axis=1
    ).mean()

    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred, allergen_list)

    print(f"\n{'=' * 60}")
    print(f"  {prefix}")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated: {len(df)}")
    print(f"  Exact match (strict):        {exact_agreement:.2%}")
    print(f"  Detected ⊆ official (no FP): {subset_agreement:.2%}")
    print(f"  Detected ⊇ official (no FN): {superset_agreement:.2%}")
    print(f"  Avg Jaccard:                 {avg_jac:.2%}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Allergen':<12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'─' * 44}")
    for allergen in allergen_list:
        m = per_class[allergen]
        print(f"  {allergen:<12s} {m['precision']:>9.2%} {m['recall']:>9.2%} {m['f1']:>9.2%}")
    print()

    return {
        "exact_agreement": float(exact_agreement),
        "subset_agreement": float(subset_agreement),
        "superset_agreement": float(superset_agreement),
        "avg_jaccard": float(avg_jac),
        "per_class": per_class,
        "n_samples": len(df),
    }


__all__ = [
    'print_classification_report',
    'print_per_class_metrics',
    'find_best_thresholds',
    'apply_thresholds',
    'compute_per_class_metrics',
    'compute_multilabel_confusion_matrix',
    'plot_confusion_matrices',
    'error_analysis',
    'binary_to_label_list',
    'jaccard',
    'evaluate_rule_vs_official',
]

# Export commonly used evaluation functions
__all__ = [
    'print_classification_report',
    'print_per_class_metrics',
    'find_best_thresholds',
    'apply_thresholds',
    'compute_per_class_metrics',
    'compute_multilabel_confusion_matrix',
    'plot_confusion_matrices',
    'error_analysis',
    'binary_to_label_list',
    'jaccard',
]