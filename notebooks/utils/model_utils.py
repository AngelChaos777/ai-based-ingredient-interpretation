"""
Model loading, predictions, and hybrid inference for MobileBERT allergen classifiers.

Provides model/tokenizer loading with automatic device placement, class weight
computation for imbalanced multi-label data, ML prediction with configurable
thresholds, hybrid (ML + rule-based) inference, and optimal threshold search.

Usage:
    >>> model, tokenizer, device = load_model_and_tokenizer("../models/mobilebert_allergen_final/")
    >>> weights = compute_class_weights(train_labels)
    >>> predictions, probabilities = predict_ml(texts, model, tokenizer, device)
    >>> hybrid = hybrid_predict(texts, model, tokenizer, device, thresholds)
    >>> hybrid_preds = hybrid_predict(texts, model, tokenizer, device)
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import List, Tuple, Optional, Dict, Any
import os
import warnings
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
    Compute inverse-frequency class weights for imbalanced multi-label data.

    Uses log1p smoothing, normalizes weights to mean=1, and caps extreme values
    to prevent training instability. Accepts numpy arrays or torch tensors.

    Args:
        labels: Binary labels (n_samples, n_classes).
        max_weight: Cap (default 10.0). None disables capping.

    Returns:
        Tensor of class weights normalized to mean=1.

    Raises:
        ValueError: If labels contain NaN/Inf or are empty.
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


def _fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
    """F-beta score (beta=1 → F1; beta=2 → recall weighted 2x; beta=0.5 → precision weighted)."""
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = (1 + beta ** 2) * precision + beta ** 2 * recall
    return (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0


def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, step: float = 0.01,
                         beta: float = 2.0, min_recall: float = 0.9) -> np.ndarray:
    """
    Find per-class thresholds on a **validation** set with a recall-first objective.

    Recall is prioritized for allergen detection (a missed allergen is the dangerous
    error — "false negative" means telling an allergic consumer a product is safe when
    it is not). Two knobs, used together by default:

    - ``beta``: F-beta objective. ``beta=2.0`` weights recall 4x vs precision
      (recall-first); ``beta=1.0`` recovers the old balanced-F1 objective.
    - ``min_recall``: among thresholds that reach at least this recall, pick the one
      with the best F-beta (avoids the trivial "predict everything → recall=1" point
      while still guaranteeing a recall floor).

    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        labels: True binary labels (n_samples, n_classes)
        step: Step size for threshold search
        beta: F-beta weighting (default 2.0 = recall-first)
        min_recall: minimum recall floor in [0, 1]. Set to None to skip the constraint.

    Returns:
        Array of optimal thresholds for each class
    """
    num_classes = probs.shape[1]
    best_thresholds = []

    for i in range(num_classes):
        # No positives in this class → the recall floor is vacuous and the F-beta is
        # 0 everywhere; keep the conservative default threshold (matches the old
        # balanced-F1 behaviour and avoids the unconstrained search picking an
        # arbitrary low threshold for a class that never occurs in validation).
        if (labels[:, i] == 1).sum() == 0:
            best_thresholds.append(0.5)
            continue
        best_t = 0.5
        best_score = -1.0
        best_t_unconstrained = 0.5
        best_score_unconstrained = -1.0
        for t in np.arange(0.01, 0.99, step):
            preds = (probs[:, i] >= t).astype(int)
            score = _fbeta_score(labels[:, i], preds, beta=beta)
            if score > best_score_unconstrained:
                best_score_unconstrained = score
                best_t_unconstrained = t
            if min_recall is not None:
                tp = float(((preds == 1) & (labels[:, i] == 1)).sum())
                fn = float(((preds == 0) & (labels[:, i] == 1)).sum())
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if recall < min_recall:
                    continue
            if score > best_score:
                best_score = score
                best_t = t
        # If no threshold reached the recall floor, fall back to the unconstrained best
        best_thresholds.append(best_t if best_score >= 0 else best_t_unconstrained)

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
                   batch_size: int = 8,
                   per_allergen_modes: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make hybrid predictions combining ML and rule-based approaches.

    Supports per-allergen hybrid strategies via *per_allergen_modes*.
    Available modes for each allergen:

    - ``None`` (default): uses the global *mode*.
    - ``'hard_override'`` / ``'rule_priority'``: if the rule engine detects
      the allergen, set prediction to 1 regardless of ML probability.

    Args:
        texts: List of input texts
        model: Trained MobileBERT model
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        thresholds: Optional array of thresholds for each class (default: 0.5 for all)
        rule_conf_threshold: Probability threshold used by 'high_confidence_bypass'
        mode: Global hybrid mode ('hard_override', 'soft', 'high_confidence_bypass')
        alpha: Weight for rule-based contribution in 'soft' mode
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for ML inference (default: 8)
        per_allergen_modes: Per-allergen override modes, e.g.
            ``{"tree_nuts": "rule_priority"}``.  Allergens not listed
            fall back to the global *mode*.

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

            # Resolve effective mode for this allergen
            effective_mode = (per_allergen_modes or {}).get(allergen, mode)

            if effective_mode in ('hard_override', 'rule_priority'):
                if rule_present:
                    hybrid_preds[i][j] = 1
            elif effective_mode == 'soft':
                final_prob = probs[i][j] + alpha * (1 if rule_present else 0)
                hybrid_preds[i][j] = 1 if final_prob > 0.5 else 0
            elif effective_mode == 'high_confidence_bypass':
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


# ═══════════════════════════════════════════════
# ASL, Calibration & Set-Size Regularization
# ═══════════════════════════════════════════════

class AsymmetricLoss(torch.nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Down-weights easy negatives more aggressively than easy positives,
    directly addressing the false-positive over-prediction problem.

    Reference:
        "Asymmetric Loss For Multi-Label Classification"
        Ridnik et al., 2021  https://arxiv.org/abs/2009.14119

    Args:
        gamma_plus:  Focusing for positives (default 0.0 — off).
        gamma_minus: Focusing for negatives (default 2.0).
        pos_weight:  Per-class positive-weight tensor (same semantics
                     as BCEWithLogitsLoss ``pos_weight``).
        reduction:   ``'mean'`` (default) or ``'sum'``.
    """

    def __init__(self, gamma_plus: float = 0.0, gamma_minus: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Asymmetric focusing weights
        #   positives: (1 - p)^γ₊ — hard positives have low p
        #   negatives: p^γ₋       — hard negatives have high p
        w_pos = (1.0 - probs) ** self.gamma_plus
        w_neg = probs ** self.gamma_minus
        loss_weight = targets * w_pos + (1.0 - targets) * w_neg

        # Class weights (per-class importance — e.g. inverse frequency)
        if self.pos_weight is not None:
            pw = self.pos_weight.view(1, -1) if self.pos_weight.dim() == 1 else self.pos_weight  # type: ignore[union-attr]
            loss_weight = loss_weight * (targets * pw + (1.0 - targets))

        # Detach loss_weight to prevent gradient flow through the focusing
        # weights — the weight tensor passed to BCEWithLogitsLoss must not
        # have requires_grad=True.
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=loss_weight.detach(), reduction=self.reduction)


def label_count_regularization(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.05,
    asymmetric: bool = True,
) -> torch.Tensor:
    """Penalise deviations from the expected number of positive labels.

    When *asymmetric* is ``True`` (default) only over-prediction is penalised
    — the model can always predict *fewer* labels without cost.  This directly
    targets false-positive inflation.

    Args:
        logits:     Raw model logits ``(batch, n_classes)``.
        targets:    Binary labels ``(batch, n_classes)``.
        alpha:      Regularisation strength.
        asymmetric: If ``True`` only penalise over-prediction (default).

    Returns:
        Scalar regularisation term (detached targets so gradients only flow
        through the prediction side).
    """
    probs = torch.sigmoid(logits)
    pred_count = probs.sum(dim=1)
    true_count = targets.float().sum(dim=1)

    diff = pred_count - true_count.detach()
    if asymmetric:
        diff = diff * (diff > 0).float()  # only over-prediction hurts
    return alpha * (diff ** 2).mean()


class TemperatureScaling:
    """Post-hoc temperature scaling for better probability calibration.

    Learns a single scalar temperature T on a validation set.  After scaling,
    probabilities become ``σ(logit / T)``.  T < 1 sharpens predictions, T > 1
    makes them more uniform.

    Usage::

        scaler = TemperatureScaling()
        t = scaler.calibrate(val_logits, val_labels)
        calibrated = scaler(test_logits)
    """

    def __init__(self, init_temp: float = 1.0):
        self.temperature = float(init_temp)

    def calibrate(self, logits: np.ndarray, labels: np.ndarray,
                  max_iter: int = 100) -> float:
        """Learn optimal temperature by minimising BCE NLL on *logits*."""
        logits_t = torch.tensor(logits, dtype=torch.float)
        labels_t = torch.tensor(labels, dtype=torch.float)
        temp = torch.nn.Parameter(torch.tensor(self.temperature))
        opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=max_iter)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(logits_t / temp.clamp(min=1e-6), labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(temp.item())
        return self.temperature

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to numpy logits."""
        return logits / max(self.temperature, 1e-6)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to a torch tensor."""
        return logits / max(self.temperature, 1e-6)


class ClassifierChainMobileBERT(torch.nn.Module):
    """MobileBERT with chained classifier heads for multi-label classification.

    Each head k receives the [CLS] embedding concatenated with sigmoid
    predictions from heads 0..k-1.  Teacher forcing (true labels) feeds
    previous-head signals during training; self-prediction is used at eval.

    Args:
        backbone_name: HuggingFace model name (e.g. ``"google/mobilebert-uncased"``).
        n_classes: Number of labels (8 for BIG8 allergens).
        chain_order: List of allergen name strings in the order heads
            should predict.  Determines label reordering indices.
        hidden_size: BERT hidden dimension (default 512 for MobileBERT).
    """

    def __init__(
        self,
        backbone_name: str = "google/mobilebert-uncased",
        n_classes: int = 8,
        chain_order: Optional[List[str]] = None,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.chain_order = chain_order or list(range(n_classes))

        # Indices for label reordering
        canonical = get_allergen_list()
        if isinstance(self.chain_order[0], str):
            self.chain_order_indices = [canonical.index(name) for name in self.chain_order]
        else:
            self.chain_order_indices = list(self.chain_order)
        self.inverse_order_indices = [0] * n_classes
        for chain_idx, canon_idx in enumerate(self.chain_order_indices):
            self.inverse_order_indices[canon_idx] = chain_idx
        self.register_buffer('_chain_ord', torch.tensor(self.chain_order_indices, dtype=torch.long))
        self.register_buffer('_inv_ord', torch.tensor(self.inverse_order_indices, dtype=torch.long))

        self.chain_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size + i, 1) for i in range(n_classes)
        ])

    def forward(self, input_ids, attention_mask=None, prev_preds=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]

        if prev_preds is not None:
            prev_chain = prev_preds[:, self._chain_ord]
        else:
            prev_chain = None

        logits = []
        h = cls
        for i in range(self.n_classes):
            logit = self.chain_heads[i](h)
            logits.append(logit)
            if i < self.n_classes - 1:
                if self.training and prev_chain is not None:
                    pred = prev_chain[:, i:i + 1].float()
                else:
                    pred = torch.sigmoid(logit).detach()
                h = torch.cat([h, pred], dim=-1)

        return torch.cat(logits, dim=-1)

    def reorder_logits_to_canonical(self, logits):
        """Convert logits from chain order back to canonical allergen order."""
        return logits[:, self._inv_ord]

    @classmethod
    def from_pretrained(cls, model_path: str, device=None):
        order_path = os.path.join(model_path, "chain_order.json")
        if not os.path.exists(order_path):
            raise FileNotFoundError(
                f"No chain_order.json found in {model_path}."
            )
        with open(order_path) as f:
            order = json.load(f)

        # Optional sidecar config so the loader uses the backbone / class count
        # the model was actually trained with instead of hardcoded defaults.
        cfg = {}
        cfg_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        backbone = cfg.get("backbone_name", "google/mobilebert-uncased")
        n_classes = cfg.get("n_classes")

        state = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # Backward compatibility: if no model_config.json, infer the number of
        # classes from the saved chain heads.
        if n_classes is None:
            n_heads = sum(1 for k in state if k.startswith("chain_heads.")
                          and k.endswith(".weight"))
            n_classes = n_heads or 8

        model = cls(backbone_name=backbone, n_classes=n_classes, chain_order=order)
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            warnings.warn(
                "ClassifierChainMobileBERT state dict did not match the "
                f"architecture exactly: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        return model


# Re-export commonly used functions
__all__ = [
    'load_model_and_tokenizer',
    'load_hybrid_config',
    'compute_class_weights',
    'find_best_thresholds',
    'predict_ml',
    'hybrid_predict',
    'apply_thresholds',
    'AsymmetricLoss',
    'label_count_regularization',
    'TemperatureScaling',
    'ClassifierChainMobileBERT',
]