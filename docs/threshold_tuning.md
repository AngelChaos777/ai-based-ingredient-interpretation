# Threshold Tuning Guide

> **Project:** Food Label Transparency for Filipino Consumers
> **Applies to:** Allergen Detection Model (MobileBERT, 8-class multi-label)

## Overview

MobileBERT outputs **raw logits**, not calibrated probabilities.  A sigmoid
converts logits to [0, 1] scores, but the default threshold of 0.5 is rarely
optimal for all classes — especially with severe class imbalance (tree_nuts at
1.5%, shellfish at 3%).

This guide explains how thresholds are optimized, how to interpret them, and
how to adjust them for different deployment trade-offs.

---

## How Thresholds Are Found (TL;DR)

The function `find_best_thresholds()` in `model_utils.py` searches a threshold
grid per class and maximizes F-beta (beta = 2 by default), subject to a
minimum recall of 0.9:

```python
thresholds = find_best_thresholds(
    probs_val, val_labels, step=0.01, beta=2.0, min_recall=0.9
)
```

- **Metric optimized:** per-class F-beta (beta = 2 weights recall more heavily)
- **Recall constraint:** at least 0.9 where achievable
- **Step size:** 0.01 (default; adjustable)
- **Range tested:** 0.01 to 0.98
- **Fallback:** if no threshold reaches the recall floor, the best unconstrained
  F-beta threshold is used; classes with no validation positives use 0.5

---

## Loading Production Thresholds

Do not treat thresholds printed in old reports as current model settings. Load
the thresholds produced by the evaluation notebook:

```python
import json
import numpy as np

with open("models/hybrid_config.json", encoding="utf-8") as f:
    config = json.load(f)
thresholds = np.asarray(config["ml_thresholds"])
```

Previous counter-example augmentation included terms chosen after inspecting
held-out test false positives. Metrics and thresholds from those runs are
exploratory; confirmatory results require retraining and evaluation on a fresh,
untouched test set.

### Threshold Interpretation

| Value | Meaning | Why |
|-------|---------|-----|
| Low | More predictions; generally higher recall and more false positives | The probability threshold is easier to cross |
| High | Fewer predictions; generally higher precision and more false negatives | The probability threshold is harder to cross |

---

## How to Run Threshold Tuning

### Prerequisites

You need:
- Predicted probabilities from the model on a **validation set** (hold-out
  data, not training or test)
- The true labels for that set
- The `model_utils` module (`notebooks/utils/model_utils.py`)

### Steps

```python
import numpy as np
from utils.model_utils import find_best_thresholds, apply_thresholds
from utils.evaluation import print_per_class_metrics, compute_per_class_metrics

# 1. Run model on validation set to get probabilities
_, probs_val = predict_ml(val_texts, model, tokenizer, device)
# probs_val.shape == (n_val_samples, 8)

# 2. Find optimal thresholds
thresholds = find_best_thresholds(
    probs_val, val_labels, step=0.01, beta=2.0, min_recall=0.9
)
print("Optimal thresholds:", thresholds)

# 3. Apply to validation set
preds_val = apply_thresholds(probs_val, thresholds)

# 4. Evaluate
print_per_class_metrics(val_labels, preds_val, ALLERGEN_NAMES)

# 5. Save for production
np.save("models/best_thresholds.npy", thresholds)
```

### Storage

| File | Format | Content |
|------|--------|---------|
| `models/best_thresholds.npy` | NumPy `.npy` | `float32` array of shape `(8,)` |
| `models/hybrid_config.json` | JSON | Same thresholds + hybrid mode config |

Both are kept in sync after optimization (the notebook writes both).

---

## Adjusting Thresholds for Different Trade-offs

### For higher recall (reduce false negatives)

Lower thresholds so the model flags more allergens.  Useful when:
- Deploying in a safety-critical context
- Users have severe allergies
- You prefer extra warnings over missed detections

```python
recall_thresholds = np.clip(thresholds * 0.5, 0.0, 1.0)
```

### For higher precision (reduce false positives)

Raise thresholds.  Useful when:
- Users report "alert fatigue" from too many warnings
- The application needs to minimize noise before user trust is established

```python
# Make threshold 25% stricter
precision_thresholds = np.maximum(thresholds, 0.5)
```

### Custom per-class rules

Instead of multiplying, set thresholds per class by business priority:

```python
# Apply class-specific changes to a copy of the loaded thresholds.
custom_thresholds = thresholds.copy()
# custom_thresholds[class_index] = desired_value
```

---

## Hybrid Inference Mode

The `hybrid_config.json` also controls the hybrid detection pipeline:

```python
mode = config.get("mode", "hard_override")
per_allergen_modes = config.get("per_allergen_modes", {})
```

### mode options

| Mode | Behavior |
|------|----------|
| `"hard_override"` / `"rule_priority"` | A positive rule match forces a positive prediction, regardless of ML probability. |
| `"ml_only"` | Use the ML prediction without rule-based overrides. |
| `"soft"` | Add the configured rule contribution to the ML probability and compare with 0.5. |
| `"high_confidence_bypass"` | Apply a rule override only when its ML probability is above `rule_conf_threshold` and below 0.9. |

### rule_conf_threshold

`rule_conf_threshold` is used by `high_confidence_bypass`; it does not gate
`hard_override` or `rule_priority`.

---

## Semantic Model Thresholds

The 40-category semantic model uses a **uniform 0.5 threshold** — every class
uses the same cutoff.  Per-class optimization is unnecessary because:

- Training set is large (14,898 samples)
- All 40 categories have F1 > 0.50 with 0.5 threshold
- No severe class imbalance (Macro F1 = 0.9632)

To adjust the semantic model threshold:

```python
semantic_preds = (semantic_probs >= 0.5).astype(int)
# Change 0.5 to a different value to trade precision ↔ recall globally
```

---

## Notes for Mobile Deployment

1. **Thresholds live client-side.**  Do not hardcode thresholds in the ONNX
   model — the model outputs raw logits, and the client applies sigmoid +
   thresholds.
2. **Config-driven:** Load thresholds from a configuration file (JSON or
   `.npy`) so they can be updated without redeploying the app.
3. **A/B testing:** Ship different thresholds to different user cohorts and
   measure false-positive rates to find the optimal balance for real users.
