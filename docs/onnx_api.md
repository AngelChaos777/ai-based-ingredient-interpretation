# ONNX Model API Reference

> **Project:** Food Label Transparency for Filipino Consumers
> **Target runtime:** ONNX Runtime Mobile (Android) / ONNX Runtime (CPU)

## Overview

Two fine-tuned MobileBERT models are exported to ONNX format for mobile/edge
deployment.  This document describes their input format, preprocessing
requirements, output shape, and integration contract.

---

## Model Inventory

| Model | File | Classes | Size |
|-------|------|---------|------|
| Allergen detection | `models/exported/allergen_model/allergen_model.onnx` | 8 | ~94 MB |
| Semantic classification | `models/exported/semantic_model/semantic_model.onnx` | 40 | ~94 MB |

*Export config is stored alongside each `.onnx` file as
`<model_name>_export_config.json`.*

---

## Common Properties

Both models share the same export configuration:

| Property | Value |
|----------|-------|
| Base architecture | `google/mobilebert-uncased` |
| ONNX opset | 18 |
| Dynamic axes | `batch_size` and `seq_len` are dynamic |
| Input 1 | **`input_ids`** — token IDs, shape `(batch_size, seq_len)`, dtype `int64` |
| Input 2 | **`attention_mask`** — attention mask, shape `(batch_size, seq_len)`, dtype `int64` |
| Output | **`logits`** — raw logits, shape `(batch_size, num_classes)`, dtype `float32` |
| Input names | `input_ids`, `attention_mask` |
| Output names | `logits` |

**Important:** The output is **raw logits**, not probabilities.  Apply
`sigmoid` per class, then compare against class-specific thresholds
(see [Threshold Tuning Guide](threshold_tuning.md)).

---

## Input Preprocessing

Both models use the same tokenizer and preprocessing pipeline.

### Tokenizer

- **Model:** `google/mobilebert-uncased` (HuggingFace `AutoTokenizer`)
- **Files:** `tokenizer.json`, `vocab.txt` shipped inside the model directory
- **Training truncation length:** 209 tokens (allergen model) / 32 tokens (semantic model)
- **Truncation:** Right-truncate texts longer than `max_length`
- **Padding:** Pad to `max_length` for static shapes, or pad to batch max with
  dynamic seq_len

### Input preparation (Python pseudo-code)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/mobilebert_allergen_final/")

inputs = tokenizer(
    texts,
    max_length=209,           # allergen model; use 32 for the semantic model
    padding="max_length",   # or "longest" for dynamic seq_len
    truncation=True,
    return_tensors="np",    # returns numpy arrays
)
# inputs["input_ids"]      — shape (batch_size, 209), dtype int64
# inputs["attention_mask"] — shape (batch_size, 209), dtype int64
```

### Preprocessing rules

1. **HTML stripping:** Remove HTML tags via `clean_html(text)` before tokenization
2. **Lowercasing:** Already done by MobileBERT tokenizer (uncased)
3. **Empty text:** Skip empty or whitespace-only texts before inference. Do not
   assume an empty input produces zero logits.

---

## Allergen Model

### Output classes (index order)

| Index | Allergen |
|-------|----------|
| 0 | milk |
| 1 | eggs |
| 2 | peanuts |
| 3 | tree_nuts |
| 4 | soy |
| 5 | wheat |
| 6 | fish |
| 7 | shellfish |

### Output to prediction

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("allergen_model.onnx")
input_name = sess.get_inputs()[0].name
mask_name = sess.get_inputs()[1].name

logits = sess.run(["logits"], {
    input_name: input_ids,
    mask_name: attention_mask,
})[0]                         # (batch_size, 8)

probs = 1 / (1 + np.exp(-logits))   # sigmoid

# Load thresholds for the exact exported model/configuration.
import json
with open("hybrid_config.json", encoding="utf-8") as f:
    config = json.load(f)
thresholds = np.asarray(config["ml_thresholds"])
predictions = (probs >= thresholds).astype(int)
```

This ONNX example applies only the ML thresholds. To reproduce hybrid
predictions, the client must also implement the configured rule engine and
apply `per_allergen_modes` from the configuration.

### Validation tolerance

Max numerical difference vs. PyTorch baseline: **1.14×10⁻⁵**
(well within tolerance of 5×10⁻³).

---

## Semantic Model

### Output classes (index order)

The 40 semantic categories in output order:

| Index | Category | Index | Category |
|-------|----------|-------|----------|
| 0 | food_additive | 20 | sodium_compound |
| 1 | preservative | 21 | sugar |
| 2 | flavor_enhancer | 22 | added_sugar |
| 3 | sweetener | 23 | fiber |
| 4 | emulsifier | 24 | vitamin |
| 5 | stabilizer | 25 | mineral |
| 6 | thickener | 26 | salt |
| 7 | antioxidant | 27 | yeast |
| 8 | acidulant | 28 | leavening_agent |
| 9 | colorant | 29 | gelling_agent |
| 10 | fat_source | 30 | humectant |
| 11 | oil_source | 31 | flavoring |
| 12 | protein_source | 32 | spice |
| 13 | carbohydrate_source | 33 | fruit_derived |
| 14 | animal_derived | 34 | vegetable_derived |
| 15 | plant_derived | 35 | fermented |
| 16 | milk_derivative | 36 | cured |
| 17 | egg_derivative | 37 | enzyme |
| 18 | soy_derivative | 38 | culture |
| 19 | wheat_derivative | 39 | salt_substitute |

### Output to prediction

```python
probs = 1 / (1 + np.exp(-logits))   # sigmoid
predictions = (probs >= 0.5).astype(int)   # default threshold = 0.5
```

The semantic model uses a uniform 0.5 threshold (no per-class optimization
needed — all 40 categories achieve F1 > 0.50 with the default threshold).

---

## ONNX Runtime Integration

### Android (Kotlin / Java)

```kotlin
// Add to build.gradle:
// implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'

val ortEnv = OrtEnvironment.getEnvironment()
val session = ortEnv.createSession("allergen_model.onnx", OrtSession.SessionOptions())

// inputIds: LongArray of token IDs, shape (1, 209)
// mask: LongArray of attention mask, shape (1, 209)
val inputTensor = OnnxTensor.createTensor(ortEnv, inputIds, longArrayOf(1, 209))
val maskTensor = OnnxTensor.createTensor(ortEnv, mask, longArrayOf(1, 209))

val results = session.run(mapOf("input_ids" to inputTensor, "attention_mask" to maskTensor))
val logits = results["logits"].get().asTensor().floatArray  // shape (1, 8)

// Apply sigmoid and thresholds client-side
```

### Python (server-side)

```python
import onnxruntime as ort

sess = ort.InferenceSession("allergen_model.onnx")
logits = sess.run(["logits"], {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})[0]
```

### Performance (CPU, batch-size 8)

| Engine | Mean | Median | P95 | P99 |
|--------|------|--------|-----|-----|
| PyTorch | 19.2 ms | 19.0 ms | 20.6 ms | 21.5 ms |
| ONNX Runtime | 11.2 ms | 11.2 ms | 11.5 ms | 11.5 ms |

ONNX Runtime runs **~1.7× faster** than PyTorch CPU inference on the
same hardware.

---

## Common Errors

### "Unexpected input" / input name mismatch
ONNX Runtime is sensitive to exact input names.  Use `sess.get_inputs()` to
inspect the expected names — never hardcode them.

### int64 vs int32
The exported model expects **int64** inputs for both `input_ids` and
`attention_mask`.  If your framework produces int32 (common in Android),
cast before calling `run()`:
- Java/Kotlin: use `LongBuffer` / `long[]`
- Python/NumPy: `input_ids.astype(np.int64)`

### Empty / all-padding input
A batch where every token is `[PAD]` is not a meaningful ingredient input.
The caller should detect and skip such inputs before inference.

---

## Model Directory Structure

```
models/
├── mobilebert_allergen_final/          # PyTorch checkpoint + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── mobilebert_semantic_final/          # PyTorch checkpoint + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
└── exported/
    └── allergen_model/
        ├── allergen_model.onnx
        └── allergen_model_export_config.json
        # (semantic model follows same structure)
```
