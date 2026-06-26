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
from .text_processing import rule_match, get_allergen_list



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


def compute_class_weights(labels: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    """
    Compute class weights for imbalanced multi-label classification.

    Uses inverse-frequency weighting with log1p smoothing.
    Weights are normalized to mean=1 and capped to prevent extreme values
    that can cause training instability (loss explosion in epoch 1).

    Accepts both numpy arrays and torch tensors — converts to numpy
    internally for consistent computation.

    Args:
        labels: Array of binary labels (n_samples, n_classes)
        max_weight: Cap class weights at this value (default: 10.0).
                    Set to None to disable capping.

    Returns:
        Tensor of class weights normalized to mean=1

    Raises:
        ValueError: If labels contain NaN or Inf values, or if labels is empty
    """
    # Convert torch tensor to numpy for consistent handling
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if labels.size == 0:
        raise ValueError("labels array is empty — cannot compute class weights")

    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"labels must be numeric, got dtype={labels.dtype}")

    if np.any(np.isnan(labels)):
        raise ValueError("labels contains NaN values")
    if np.any(np.isinf(labels)):
        raise ValueError("labels contains Inf values")

    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    n_classes = labels.shape[1]

    # Handle edge case: all-positive or all-negative columns
    # For a column with 0 positives: give max_weight (rare class)
    # For a column with all positives: give max_weight (rare negative case)
    weights = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        if pos_counts[i] == 0:
            # No positive samples at all — maximum weight
            weights[i] = max_weight if max_weight is not None else 10.0
        elif neg_counts[i] == 0:
            # All positive — rare but possible
            weights[i] = max_weight if max_weight is not None else 10.0
        else:
            # Standard inverse frequency with log smoothing
            weights[i] = np.log1p(neg_counts[i] / pos_counts[i])

    # Normalize to mean=1 for stable training
    weights = weights / (weights.mean() + 1e-8)

    # Cap maximum weight to prevent loss explosion
    if max_weight is not None:
        weights = np.clip(weights, None, max_weight)

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
               max_length: int = 209,
               batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the MobileBERT model with batched inference.

    Processes texts in small batches to avoid GPU OOM on memory-constrained
    systems (e.g. consumer GPUs with 4-8 GiB VRAM).

    Args:
        texts: List of input texts
        model: Trained MobileBERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        thresholds: Optional array of thresholds for each class (default: 0.5 for all)
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for inference (default: 8)

    Returns:
        Tuple of (predictions, probabilities)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = np.array([0.5] * len(get_allergen_list()))

    all_probs = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)

        # Clear cached memory periodically to avoid fragmentation
        if i % 64 == 0 and i > 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= thresholds).astype(int)

    return preds, probs


def hybrid_predict(texts: List[str],
                   model: torch.nn.Module,
                   tokenizer: AutoTokenizer,
                   device: torch.device,
                   thresholds: Optional[np.ndarray] = None,
                   rule_conf_threshold: float = 0.2,
                   mode: str = 'hard_override',
                   alpha: float = 0.3,
                   max_length: int = 209,
                   batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
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
        batch_size: Batch size for ML inference (default: 8)

    Returns:
        Tuple of (hybrid_predictions, ml_probabilities)
    """
    # Get ML predictions and probabilities with batched inference
    ml_preds, probs = predict_ml(texts, model, tokenizer, device, thresholds, max_length, batch_size)

    # Initialize hybrid predictions as ML predictions
    hybrid_preds = ml_preds.copy()

    # Apply hybrid logic
    allergen_list = get_allergen_list()
    for i, text in enumerate(texts):
        for j, allergen in enumerate(allergen_list):
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