# Model Cards

> **Project:** Food Label Transparency for Filipino Consumers

---

## Model Card A: Allergen Detection (MobileBERT)

### Model Details

- **Developer:** Thesis project — Food Label Transparency for Filipino Consumers
- **Model type:** Fine-tuned `google/mobilebert-uncased` with a multi-label
  classification head (8 outputs + sigmoid)
- **Parameters:** ~25 million
- **Hidden size:** 512 (with bottleneck 128)
- **Transformer layers:** 24
- **License:** Apache 2.0 (base model)
- **Date:** June 2026
- **Export format:** PyTorch (`.safetensors`) + ONNX (`.onnx`)

### Intended Use

- **Primary task:** Detect which of the 8 FDA Big-8 allergens are present in
  food ingredient text
- **Input:** Ingredient statement text (e.g., "milk, sugar, wheat flour, eggs")
- **Output:** Multi-label binary vector (8 positions, one per allergen)
- **Target platform:** Android mobile app via ONNX Runtime Mobile
- **Environment:** Offline inference (no internet connection required)

### Training Data

| Source | Samples |
|--------|---------|
| Open Food Facts (Philippine-relevant) | 1,057 products |
| After augmentation (synonym + negative examples) | 1,952 training samples |

**Label source:** Rule-based keywords + Open Food Facts official tags combined
via `combine_allergen_labels()`. Labels are ~74% consistent with OFF official
tags (exact set match).

**Class distribution (original 1,057 samples):**

| Allergen | Positive % |
|----------|-----------|
| milk | 47% |
| wheat | 34% |
| soy | 26% |
| eggs | 9% |
| peanuts | 8% |
| fish | 7% |
| shellfish | 3% |
| tree_nuts | 1.5% |

**Severe class imbalance:** tree_nuts (1.5%), shellfish (3%), fish (7%) are
rare.  Class weights (inverse-frequency, capped at 10×) compensate during
training.

### Evaluation Data

- **Test set:** 159 samples (15% stratified holdout)
- **Validation set:** 159 samples (15%)
- **Training set:** 739 samples (70%) — augmented to 1,952

### Metrics

> **Exploratory historical results:** Earlier counter-example augmentation
> included terms selected after inspecting held-out test false positives. The
> figures below are not confirmatory and should not be used as evidence of
> generalization. Retraining and evaluation on a fresh, untouched test set are
> required before reporting confirmatory metrics.

#### ML-Only (MobileBERT with per-class thresholds)

| Metric | Value |
|--------|-------|
| Macro F1 | 0.8632 |
| Micro F1 | 0.9302 |
| Weighted Avg F1 | 0.9345 |

#### Per-class F1 (ML-Only)

| Allergen | F1 | Support |
|----------|----|---------|
| milk | 0.95 | 76 |
| eggs | 0.93 | 14 |
| peanuts | 0.96 | 12 |
| tree_nuts | 0.44 | 3 |
| soy | 0.90 | 41 |
| wheat | 0.97 | 55 |
| fish | 0.91 | 12 |
| shellfish | 0.83 | 4 |

#### Hybrid (ML + Rule override)

| Metric | Value |
|--------|-------|
| Macro F1 | 0.8746 |
| Micro F1 | 0.9352 |

### Thresholds

Thresholds are selected on validation data using a recall-first F-beta
objective (beta = 2, minimum recall = 0.9 where achievable). Load the
thresholds for a specific model from `models/hybrid_config.json`; previous
reported values are omitted because their associated training run used
test-informed augmentation.

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Epochs | 15 (early stopping patience=3) |
| Batch size | 8 |
| Max sequence length | 209 tokens |
| Weight decay | 0.01 |
| Gradient clipping | `max_norm=1.0` |
| Loss | Weighted BCEWithLogitsLoss |

### Known Limitations

1. **Rare-class weakness:** tree_nuts F1 = 0.44 — the model struggles with
   infrequent nut varieties (pecans, macadamia, etc.)
2. **Label noise:** Training labels are rule-generated, not human-annotated.
   The ~74% agreement with OFF tags means reported F1 may overestimate
   performance against true ground truth.
3. **OFF bias:** Trained primarily on Open Food Facts data — may not fully
   capture Philippine product formulations or Filipino ingredient names
   (though Filipino variants are handled by the rule engine, not the ML model).
4. **No may-contain detection:** The model does not address cross-contamination
   or "may contain" statements — handled separately by `extract_may_contain()`.

---

## Model Card B: Semantic Classification (MobileBERT)

### Model Details

- **Developer:** Thesis project — Food Label Transparency for Filipino Consumers
- **Model type:** Fine-tuned `google/mobilebert-uncased` with a multi-label
  classification head (40 outputs + sigmoid)
- **Parameters:** ~25 million
- **Hidden size:** 512 (with bottleneck 128)
- **Transformer layers:** 24
- **License:** Apache 2.0 (base model)
- **Date:** June 2026
- **Export format:** PyTorch (`.safetensors`)

### Intended Use

- **Primary task:** Classify each ingredient into 40 semantic categories
  (additive, preservative, protein source, sweetener, etc.)
- **Input:** Ingredient text (single ingredient or ingredient phrase)
- **Output:** Multi-label binary vector (40 positions, one per category)
- **Target platform:** Research / server-side analysis
- **Environment:** Python (PyTorch or ONNX Runtime)

### Training Data

| Source | Count |
|--------|-------|
| Training samples | 14,898 |
| Validation samples | 1,862 |
| Test samples | 1,863 |
| **Total** | **18,623** |
| Label source | Rule-based mapping (~395 rules covering 62% of unique ingredients) |

### Semantic Categories (40)

The categories are organized into seven configured groups; the table also
lists two additional source categories that are not assigned to a group.

| Group | Categories |
|-------|-----------|
| Additives & Preservatives | food_additive, preservative, antioxidant, acidulant, colorant |
| Flavor & Sweeteners | flavor_enhancer, sweetener, sugar, added_sugar, flavoring, spice, salt, yeast |
| Functional Ingredients | emulsifier, stabilizer, thickener, gelling_agent, leavening_agent, humectant |
| Macronutrients | fat_source, oil_source, protein_source, carbohydrate_source, fiber |
| Origin & Derivation | animal_derived, plant_derived, milk_derivative, egg_derivative, soy_derivative, wheat_derivative, fermented, cured, sodium_compound |
| Micronutrients | vitamin, mineral |
| Biological | enzyme, culture, salt_substitute |
| Additional | fruit_derived, vegetable_derived |

*Mergers: herb→spice, smoked→cured, prebiotic→fiber*

### Metrics

| Metric | Value |
|--------|-------|
| Macro F1 | 0.9632 |
| Micro F1 | 0.9796 |
| Best validation Macro F1 | 0.9476 (epoch 10) |
| Categories with F1 > 0.50 | **40/40** (no dead categories) |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Epochs | 10 |
| Batch size | 8 |
| Max sequence length | 32 tokens |
| Loss | Weighted BCEWithLogitsLoss |

### Known Limitations

1. **Rule-derived labels:** Same limitation as the allergen model — labels come
   from ~395 mapping rules, not human annotation.  Categories with few rules
   may have noisier labels.
2. **Fixed category set:** The 40 categories were chosen for this project's
   scope.  New ingredient types (e.g., novel proteins, modern additives) may
   not fit existing categories.
3. **Single-ingredient focus:** The model classifies individual ingredients —
   not whole product formulations.  For full-product analysis, aggregate over
   all ingredient tokens.

---

## Fairness & Ethical Considerations

1. **Population bias:** Models trained primarily on Open Food Facts data, which
   over-represents Western/European products.  Philippine-specific products
   and ingredients may be underrepresented.
2. **Health impact:** False negatives (missing an allergen) are far more
   consequential than false positives.  The hybrid pipeline prioritizes recall
   via `hard_override` mode — any rule-based detection overrides ML.
3. **Not a medical device:** This system is an educational/research tool for
   ingredient transparency, not a certified medical device.  Users with food
   allergies should always verify against official product labels.
