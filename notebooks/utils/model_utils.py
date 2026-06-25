"""
Model utilities for food label transparency project.

This module provides essential utilities for loading, training, and evaluating
MobileBERT models for multi-label allergen classification. It includes
functions for loading models and tokenizers, computing class weights for
imbalanced datasets, generating predictions, and supporting hybrid inference
methods that combine rule-based and ML approaches.

Key Features:
- Load pre-trained MobileBERT models and tokenizers with device management
- Compute class weights to handle class imbalance in multi-label tasks
- Generate predictions with configurable thresholds
- Support hybrid inference combining ML and rule-based methods
- Find optimal classification thresholds to maximize F1 score

Usage Examples:
    >>> from utils.model_utils import (
    ...     load_model_and_tokenizer,
    ...     compute_class_weights,
    ...     predict_ml,
    ...     hybrid_predict
    ... )
    >>> # Load model and tokenizer
    >>> model, tokenizer, device = load_model_and_tokenizer("../models/mobilebert_allergen_final/")
    >>> # Compute class weights for imbalanced data
    >>> weights = compute_class_weights(train_labels)
    >>> # Generate ML predictions
    >>> predictions, probabilities = predict_ml(texts, model, tokenizer, device)
    >>> # Generate hybrid predictions
    >>> hybrid_preds = hybrid_predict(texts, model, tokenizer, device)
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional, Dict, Any
import os
from sklearn.metrics import f1_score

from scipy.special import expit


def load_model_and_tokenizer(model_path: str = "../models/mobilebert_allergen_final/"):
    """
    Load the trained MobileBERT model and tokenizer.

    Args:
        model_path: Path to the saved model directory

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def load_hybrid_config(config_path: str = "../models/hybrid_config.json") -> Dict[str, Any]:
    """
    Load hybrid configuration including ML thresholds and rule parameters.

    Args:
        config_path: Path to the hybrid config JSON file

    Returns:
        Dictionary containing hybrid configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hybrid config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced multi-label classification.

    Args:
        labels: Numpy array of binary labels (n_samples, n_classes)

    Returns:
        Tensor of class weights normalized to mean=1
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    # Avoid division by zero
    weights = np.log1p(neg_counts / (pos_counts + 1e-6))
    # Normalize to mean=1 for stable training
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)


def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, step: float = 0.01) -> np.ndarray:
    """
    Find optimal thresholds for each class to maximize F1 score.

    Args:
        probs: Predicted probabilities from model (n_samples, n_classes)
        labels: True binary labels (n_samples, n_classes)
        step: Step size for threshold search

    Returns:
        Array of optimal thresholds for each class
    """
    num_classes = probs.shape[1]
    best_thresholds = []

    for i in range(num_classes):
        best_t = 0.5
        best_f1 = 0
        for t in np.arange(0.01, 0.99, step):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)

    return np.array(best_thresholds)


def predict_ml(texts: List[str],
               model: torch.nn.Module,
               tokenizer: AutoTokenizer,
               device: torch.device,
               thresholds: Optional[np.ndarray] = None,
               max_length: int = 209) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the MobileBERT model.

    Args:
        texts: List of input texts
        model: Trained MobileBERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        thresholds: Optional array of thresholds for each class (default: 0.5 for all)
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (predictions, probabilities)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = np.array([0.5] * 8)  # 8 allergens

    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (probs >= thresholds).astype(int)

    return preds, probs


def rule_match(text: str, allergen: str) -> bool:
    """
    Check if allergen is present in text using rule-based matching.
    Delegates to the comprehensive implementation in text_processing.py.

    Args:
        text: Input text to search
        allergen: Allergen to check for

    Returns:
        True if allergen found and not negated, False otherwise
    """
    from .text_processing import rule_match as _tp_rule_match
    return _tp_rule_match(text, allergen)


def hybrid_predict(texts: List[str],
                   model: torch.nn.Module,
                   tokenizer: AutoTokenizer,
                   device: torch.device,
                   thresholds: Optional[np.ndarray] = None,
                   rule_conf_threshold: float = 0.2,
                   mode: str = 'hard_override',
                   alpha: float = 0.3,
                   max_length: int = 209) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make hybrid predictions combining ML and rule-based approaches.

    Args:
        texts: List of input texts
        model: Trained MobileBERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        thresholds: Optional array of thresholds for each class (default: 0.5 for all)
        rule_conf_threshold: Confidence threshold for rule-based override
        mode: Hybrid mode ('hard_override', 'soft', 'high_confidence_bypass')
        alpha: Weight for rule-based contribution in 'soft' mode
        max_length: Maximum sequence length for tokenization

    Returns:
        Tuple of (hybrid_predictions, ml_probabilities)
    """
    # Get ML predictions and probabilities
    ml_preds, probs = predict_ml(texts, model, tokenizer, device, thresholds, max_length)

    # Initialize hybrid predictions as ML predictions
    hybrid_preds = ml_preds.copy()

    # Apply hybrid logic
    for i, text in enumerate(texts):
        for j, allergen in enumerate(["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]):
            rule_present = rule_match(text, allergen)

            if mode == 'hard_override':
                if rule_present and probs[i][j] > rule_conf_threshold:
                    hybrid_preds[i][j] = 1
            elif mode == 'soft':
                final_prob = probs[i][j] + alpha * (1 if rule_present else 0)
                hybrid_preds[i][j] = 1 if final_prob > 0.5 else 0
            elif mode == 'high_confidence_bypass':
                if rule_present and probs[i][j] > rule_conf_threshold and probs[i][j] < 0.9:
                    hybrid_preds[i][j] = 1

    return hybrid_preds, probs

def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Apply threshold array to probability matrix to get binary predictions.

    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        thresholds: Thresholds for each class (n_classes,)

    Returns:
        Binary predictions array
    """
    return (probs >= thresholds).astype(int)

# Re-export commonly used functions
__all__ = [
    'load_model_and_tokenizer',
    'load_hybrid_config',
    'compute_class_weights',
    'find_best_thresholds',
    'predict_ml',
    'hybrid_predict',
    'apply_thresholds',
]