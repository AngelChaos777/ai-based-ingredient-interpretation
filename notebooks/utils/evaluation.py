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