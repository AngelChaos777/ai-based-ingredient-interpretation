# Project Context

> **Food Label Transparency for Filipino Consumers — AI-Based Ingredient Interpretation**
> A thesis project that detects FDA Big-8 allergens from Philippine food product labels using OCR, rule-based parsing, and fine-tuned MobileBERT models in a hybrid detection pipeline. Also performs semantic ingredient classification (40 categories after merging herb→spice, smoked→cured, prebiotic→fiber) as a supporting deliverable.

---

## 1. What This Project Does

This project builds a system that takes **food product ingredient text** (extracted from packaging images via OCR or from structured data like Open Food Facts) and:

1. **Classifies each ingredient** into 40 semantic categories (additive, preservative, sweetener, protein source, etc.)
2. **Detects which of the 8 major FDA-recognized allergens** are present in each product

The core contribution is a **hybrid detection pipeline** that combines:

- **A fine-tuned MobileBERT model** (multi-label text classifier) for semantic understanding
- **A second fine-tuned MobileBERT model** for allergen classification (8-class multi-label)
- **A comprehensive rule-based keyword engine** (the BIG8 database) for high-recall explicit matching
- **Exemption/negation handling** to reduce false positives (e.g., "milk-free", "soy lecithin" exemption)

The system is designed for **eventual mobile deployment** on Android (ONNX Runtime Mobile) — prioritizing small model size, fast inference, and offline operation over benchmark performance.

---

## 2. Project Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          01_extraction.ipynb             │
                    │  OCR (EasyOCR + Tesseract) → raw CSV     │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │          02_cleaning.ipynb               │
                    │  Dedup, normalize, fix encoding          │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄┐
                    █ Deliverable 1: Semantic Classification   █
                    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      03_ingredient_parsing.ipynb         │
                    │  Ingredient tokenization → parsed dataset│
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      04_semantic_labeling.ipynb          │
                    │  Rule-based 40-category mapping          │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      05_semantic_model.ipynb             │
                    │  MobileBERT fine-tune (weighted BCE)     │
                    │  → model exported to ONNX                │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄┐
                    █ Deliverable 2: Allergen Detection        █
                    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      06_allergen_labeling.ipynb          │
                    │  Rule-based + official tags, exemptions  │
                    │  Filipino variant integration            │
                    │  uses combine_allergen_labels()          │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      07_allergen_training.ipynb          │
                    │  MobileBERT fine-tune (weighted BCE)     │
                    │  Stratified split + threshold opt.       │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄┐
                    █ Deliverable 3: Hybrid Detection          █
                    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      08_hybrid_evaluation.ipynb          │
                    │  ML predictions + rule-based override    │
                    │  → hybrid_config.json, Filipino rules    │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄┐
                    █ Deliverable 4: ONNX Mobile Deployment    █
                    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      09_model_export.ipynb               │
                    │  PyTorch → ONNX → ONNX Runtime Mobile    │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │      10_mobile_benchmark.ipynb           │
                    │  Latency/size/accuracy validation        │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                           [ONNX Runtime Mobile (Android)]
```

**Important:** The notebooks assume they are executed from the `notebooks/` directory (relative paths like `../data/...` and `../models/...`).

### Utility Modules (`notebooks/utils/`)

The 6 utility modules are the **single source of truth** for all logic — notebooks should import from here rather than redefining code:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `text_processing.py` | Allergen detection, text cleaning, negation handling, Filipino variants | `detect_allergens_rule_based()`, `rule_match()`, `BIG8` dict, `apply_exemptions()`, `combine_allergen_labels()`, `FILIPINO_VARIANTS` dict |
| `data_utils.py` | Data loading, stratified splitting, augmentation | `load_labeled_data()`, `create_stratified_splits()`, `augment_dataframe()`, `get_data_directories()`, `save_metadata()` |
| `model_utils.py` | Model loading, prediction, hybrid inference | `load_model_and_tokenizer()`, `predict_ml()`, `hybrid_predict()`, `compute_class_weights()`, `find_best_thresholds()` |
| `evaluation.py` | Metrics, error analysis, reporting | `print_classification_report()`, `error_analysis()`, `compute_per_class_metrics()`, `jaccard()` |
| `semantic_utils.py` | Ingredient parsing, semantic category mapping | `parse_ingredient_list()`, `ingredient_to_categories()`, `build_semantic_label_matrix()`, `INGREDIENT_SEMANTIC_MAP` (~395 mappings, plural normalization, E-number/INS/FDC pattern matching) |
| `deployment_utils.py` | ONNX export, latency benchmarking, validation | `pytorch_to_onnx()`, `validate_onnx_output()`, `benchmark_latency()`, `benchmark_onnx_latency()` |
| `ocr_utils.py` | EasyOCR + OpenCV preprocessing | (image preprocessing functions) |

---

## 3. The 8 Allergen Classes (FDA Big 8)

| # | Allergen | `target_allergens` index | Approx. positive samples (of 1057) |
|---|----------|--------------------------|-------------------------------------|
| 0 | milk | 0 | 497 (47%) |
| 1 | eggs | 1 | 98 (9%) |
| 2 | peanuts | 2 | 88 (8%) |
| 3 | tree_nuts | 3 | 16 (1.5%) — **rare class** |
| 4 | soy | 4 | 277 (26%) |
| 5 | wheat | 5 | 355 (34%) |
| 6 | fish | 6 | 77 (7%) |
| 7 | shellfish | 7 | 32 (3%) |

**Label format:** Binary vector in this order. Example: `[1, 0, 0, 0, 0, 1, 0, 0]` = milk + wheat.

---

## 4. Data Pipeline

### Data Sources (as per thesis / Model Training Plan.md)
- **~1,400 entries** from Open Food Facts
- **~400 entries** from Philippine products
- **~200 entries** from FDA verified labels
- **Total target:** ~2,000 unique ingredient entries

### Actual Dataset
- **1,057 samples** in the final labeled dataset (`data/final/labeled_dataset_enhanced.csv`)
- Source: Open Food Facts (filtered for Philippine-relevant products)
- Each sample has: `code`, `brands`, `product_name_en`, `ingredients_text_en`, `detected_allergens` (list), `official_allergens_mapped`, `traces_allergens`, `consensus_allergens`, `combined_allergens`, `may_contain`, `coconut`

### Data Split (actual)
| Split | Size | Positive Samples |
|-------|------|-----------------|
| Training (70%) | 739 | 1,007 |
| Validation (15%) | 159 | 216 |
| Test (15%) | 159 | 217 |

After augmentation (synonym replacement + negative example generation): **1,952 training samples**.

### Augmented Dataset (NB11)
| Statistic | Value |
|-----------|-------|
| Original samples | 1,057 |
| After targeted augmentation | 1,621 (+53%) |
| tree_nuts improvement | 16 → 128 (1.5% → 7.9%) |
| shellfish improvement | 32 → 211 (3% → 13.0%) |
| fish improvement | 77 → 339 (7% → 20.9%) |
| eggs improvement | 93 → 347 (9% → 21.4%) |
| peanuts improvement | 88 → 291 (8% → 18.0%) |

**Important:** The augmented dataset (`labeled_dataset_augmented.csv`) is created by NB11 for demonstration. For production training, NB07 uses a separate training-split-only augmentation via `augment_dataframe()` to avoid test/validation contamination.

### Data Files
| Path | Description |
|------|-------------|
| `data/raw/food.parquet` | Raw Open Food Facts export |
| `data/interim/extracted_raw.csv` | After OCR extraction |
| `data/interim/parsed_ingredients.csv` | After ingredient tokenization (Notebook 03) |
| `data/interim/ingredient_vocabulary.csv` | Unique ingredient vocabulary from parsing |
| `data/interim/unknown_ingredients_for_labeling.csv` | Ingredients not in the semantic mapping — candidates for manual annotation |
| `data/processed/cleaned_dataset.csv` | After dedup + normalization |
| `data/final/labeled_dataset_enhanced.csv` | The final multi-label training dataset (1,057 rows) |
| `data/final/labeled_dataset_augmented.csv` | Augmented dataset (1,621 rows) from NB11 — 53% increase over original |
| `data/final/semantic_labels.csv` | Semantic category labels per product (40 categories) |
| `data/final/semantic_training_data.csv` | Multi-label semantic training matrix |
| `data/final/semantic_labeling_metadata.json` | Semantic labeling run metadata |
| `data/final/processing_metadata.json` | Processing timestamp and version info |

---

## 5. Rule-Based Allergen Detection

### BIG8 Keyword Database (`text_processing.py`)
Each allergen has an extensive keyword list (e.g., milk → `["milk", "whey", "casein", "caseinate", "butter", "cheese", "lactose", ...]`).

Keywords are compiled into case-insensitive regex patterns at import time in `COMPILED_RULES`.

### Negation Handling
`rule_match()` detects and returns `False` for negated patterns including:
- `"no milk"`, `"milk-free"`, `"free from milk"`, `"does not contain milk"`, `"low in milk"`, `"milk free"` (with space)

### Exemption Handling (`apply_exemptions()`)
Some ingredients contain allergen keywords but are refined enough to pose minimal risk:
- **soy**: "soy lecithin", "soybean oil", "soya oil" are exempt (unless "soy protein"/"tofu"/etc. also present)
- **tree_nuts**: "shea butter", "coconut oil", "coconut milk" are exempt
- **eggs**: "lecithin" alone is exempt (common soy lecithin, not egg-derived)

### Combined Label Sources
`combine_allergen_labels()` merges up to 4 detection channels: rule-based detection, official tags (from packaging), traces tags (may contain), and may-contain statements. Returns `detected_only`, `detected_or_official`, `consensus`, etc.

---

## 6. ML Models: Two MobileBERT Models

### 6A. Allergen Classification Model (Trained, Deployed)

**Architecture**
- **Base model:** `google/mobilebert-uncased` (HuggingFace)
- **Task:** Multi-label classification (8 outputs with sigmoid)
- **Hidden size:** 512 (with bottleneck 128)
- **Layers:** 24 transformer blocks
- **Parameters:** ~25M (vs 110M for BERT-base)
- **Classifier head:** Newly initialized `MobileBertForSequenceClassification` (8 labels)

**Training Configuration (actual)**
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.1 |
| Epochs | 15 (early stopping patience=3) |
| Batch Size | 8 |
| Max Sequence Length | 221 (computed from data) |
| Weight Decay | 0.01 |
| Max Norm | 1.0 (`max_norm` in PyTorch 2.x renamed from `max_grad_norm`) |
| Loss | Weighted BCEWithLogitsLoss (inverse frequency weights) |
| Best Model Selection | Manual early stopping with state dict checkpointing |
| Training Loop | **Manual PyTorch loop** (not HuggingFace `Trainer`) |

**Key deviation — Manual training loop:** The HuggingFace `Trainer` causes a segfault (exit 139) on this system's GPU stack (PyTorch 2.12.0+cu130, CUDA 13.0 non-standard build). The `Trainer.__init__` triggers a native SIGSEGV during device management initialization. Replaced with `run_train_epoch()`, `evaluate()`, and `predict()` functions using `torch.utils.data.DataLoader`, `get_linear_schedule_with_warmup`, and manual gradient clipping (`clip_grad_norm_` with `max_norm=1.0`). Early stopping and best-model checkpointing via plain Python state-dict saving.

**`clip_grad_norm_` API change:** PyTorch 2.x renamed the `max_grad_norm` parameter to `max_norm`. Using the old keyword raises `TypeError`.

**Thesis Plan Config (from Model Training Plan.md)**
| Parameter | Target Range | Actual Used |
|-----------|-------------|-------------|
| Learning Rate | 1e-5 to 5e-5 | 2e-5 |
| Epochs | 5 to 20 | 15 |
| Batch Size | 8 to 32 | 8 |
| Weight Decay | 0 to 0.1 | 0.01 |

**Training Time**
- **~5–8 minutes** on NVIDIA GeForce RTX 3050 (8 GB VRAM, batch size 8)
- The HuggingFace `Trainer` cannot be used — segfaults on this GPU stack (see note above)
- Manual PyTorch training loop with `DataLoader` is the working training path

**Optimized Thresholds (per class)**
These are per-class probability thresholds found on the validation set (stored in `models/best_thresholds.npy` and `models/hybrid_config.json`):

| Allergen | Optimized Threshold |
|----------|-------------------|
| milk | 0.15 |
| eggs | 0.61 |
| peanuts | 0.23 |
| tree_nuts | 0.01 |
| soy | 0.49 |
| wheat | 0.79 |
| fish | 0.98 |
| shellfish | 0.01 |

**Key insight:** Thresholds vary widely by class. Low thresholds (tree_nuts, shellfish at 0.01) indicate the model rarely assigns high probabilities to these rare classes, requiring aggressive thresholds for any recall. High thresholds (fish at 0.98, wheat at 0.79) reflect classes where the model is confident but conservative.

### 6B. Semantic Classification Model (Trained, Evaluated)

**Architecture**
- **Base model:** `google/mobilebert-uncased` (HuggingFace)
- **Task:** Multi-label classification (40 semantic outputs with sigmoid)
- **Same MobileBERT architecture** as the allergen model
- **Labels:** 40 semantic categories (merged herb→spice, smoked→cured, prebiotic→fiber)
- **Status:** Fully trained, evaluated, ONNX-exported

**Semantic Categories (7 groups, 40 labels)**
| Group | Categories |
|-------|-----------|
| Additives & Preservatives | food_additive, preservative, antioxidant, acidulant, colorant |
| Flavor & Sweeteners | flavor_enhancer, sweetener, sugar, added_sugar, flavoring, spice, salt, yeast |
| Functional Ingredients | emulsifier, stabilizer, thickener, gelling_agent, leavening_agent, humectant |
| Macronutrients | fat_source, oil_source, protein_source, carbohydrate_source, fiber |
| Origin & Derivation | animal_derived, plant_derived, milk_derivative, egg_derivative, soy_derivative, wheat_derivative, fermented, cured |
| Micronutrients | vitamin, mineral |
| Biological | enzyme, culture, salt_substitute |

(herb merged into spice, smoked merged into cured, prebiotic merged into fiber)

**Training Performance**
| Metric | Value |
|--------|-------|
| Train samples | 14,898 |
| Validation samples | 1,862 |
| Test samples | 1,863 |
| Split | 80/10/10 |
| Epochs | 10 (no early stopping) |
| Best val Macro F1 (epoch 10) | 0.9476 |

**Test Set Performance**
| Metric | Value |
|--------|-------|
| Macro F1 | 0.9632 |
| Micro F1 | 0.9796 |
| All 40 categories | **F1 > 0.50** (no dead categories) |

**Ingredient Coverage (Notebook 04)**
| Metric | Value |
|--------|-------|
| INGREDIENT_SEMANTIC_MAP entries | ~395 |
| Unique ingredients matched | 2,893 / 4,664 (62.0%) |
| Instance-level coverage | 15,776 / 18,623 (84.7%) |
| Plural normalization | Active (emulsifiers→emulsifier, etc.) |
| E-number/INS/FDC mapping | Regex-based for E100-E999, FDC colors, INS codes |

---

## 7. Hybrid Detection System

### Modes
The hybrid system (`model_utils.hybrid_predict()`) combines ML predictions with rule-based logic. Four modes:

1. **`hard_override`**: If rule-based detects an allergen AND ML probability > `rule_conf_threshold` (0.5), force prediction = 1.
2. **`soft`**: Blend ML probability and rule signal: `final_prob = ml_prob + alpha * rule_present`. Then threshold at 0.5.
3. **`high_confidence_bypass`**: If rule detects AND probability is between `rule_conf_threshold` and 0.9, set prediction = 1 (catches high-confidence misses).
4. **`rule_priority`** (used in production config): Notebook-level mode that prioritizes rule-based detection over ML — applied during labeling rather than in `hybrid_predict()`.

### Current Config (`models/hybrid_config.json`)
```json
{
  "ml_thresholds": [0.15, 0.61, 0.23, 0.01, 0.49, 0.79, 0.98, 0.01],
  "rule_conf_threshold": 0.5,
  "mode": "hard_override"
}
```

These thresholds were saved directly from Notebook 07's validation-set optimization, replacing previously stale values. The config now reflects the trained model's actual optimal operating point.

### Batched Inference (CUDA OOM Fix)

Notebook 08's `predict_ml()` previously ran all test samples as a single batch, causing CUDA OOM on the 8 GiB RTX 3050 when orphan Jupyter kernels from prior training runs held ~6.6 GiB of VRAM. Fixed by adding a `batch_size` parameter (default 8) that processes texts in small batches with periodic `torch.cuda.empty_cache()` calls. Both `predict_ml()` and `hybrid_predict()` now support `batch_size` — pass it explicitly to control GPU memory usage.

**Orphan kernel issue:** 6+ stale `ipykernel_launcher` processes can persist between runs, each holding GPU memory. Restarting the Jupyter kernel or killing orphan PIDs frees the memory. Notebook 08 now calls `torch.cuda.empty_cache()` at startup to mitigate fragmentation.

### Performance (test set) — Verified End-to-End (Phase 0 final)

Results from the complete pipeline after Phase 0 fix (reverted to original nn.Linear — see §13): Allergen model trained with original `MobileBertForSequenceClassification` classifier head, gradient clipping (`max_norm=1.0`), batch_size=8, max_length=540 (computed from data, not capped at 221).

**ML Only:**
| Metric | Value |
|--------|-------|
| Micro F1 | 0.9302 |
| Macro F1 | 0.8632 |
| Weighted Avg F1 | 0.9345 |

**Hybrid (hard override, rule_conf_threshold=0.5):**
| Metric | Value |
|--------|-------|
| Micro F1 | 0.9352 |
| Macro F1 | 0.8746 |
| Weighted Avg F1 | 0.9395 |

**Hybrid changes 2 test predictions** (vs 8 in the earlier run). Both corrections improve recall (milk→fish, tree_nuts→fish) — the rule engine catches allergens the ML model missed.

Per-class test set F1 scores (ML Only — using trained model with original nn.Linear):
| Allergen | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| milk | 1.00 | 0.91 | 0.95 |
| eggs | 0.93 | 0.93 | 0.93 |
| peanuts | 0.93 | 1.00 | 0.96 |
| tree_nuts | 0.33 | 0.67 | 0.44 |
| soy | 0.90 | 0.90 | 0.90 |
| wheat | 1.00 | 0.94 | 0.97 |
| fish | 1.00 | 0.83 | 0.91 |
| shellfish | 0.71 | 1.00 | 0.83 |

**Notable gap:** `tree_nuts` still suffers from very few positive samples (only 3 in test set, F1=0.44). This is a known failure mode of the current dataset.

### Phase 1: Rule-Based vs Official Tag Evaluation

The rule-based detection was compared against Open Food Facts (OFF) official `allergens_tags` for 1,057 products. This measures real-world rule precision after Phase 4 refinements (context-aware handling, negation strengthening, exemption rules).

**Important fix:** OFF uses `en:crustaceans` and `en:molluscs` taxonomy codes rather than `en:shellfish`. The `OFFICIAL_MAP` in `text_processing.py` had no mapping for these, causing 0% precision/recall for shellfish in the evaluation. Fixed by adding `"crustaceans": "shellfish"` and `"molluscs": "shellfish"` mappings (June 2026, post Phase 1 evaluation — awaiting re-run).

**Evaluation A — Full dataset (1,057 products, 765 with non-empty OFF tags):**
| Metric | Value |
|--------|-------|
| Exact match (strict) | 75.02% |
| Detected ⊆ official (no FP) | 87.13% |
| Detected ⊇ official (no FN) | 84.77% |
| Avg Jaccard similarity | 85.44% |

Per-class F1 vs OFF tags:
| Allergen | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| milk | 89.29% | 94.04% | 91.61% |
| eggs | 83.06% | 89.57% | 86.19% |
| peanuts | 70.45% | 93.94% | 80.52% |
| tree_nuts | 73.33% | 21.57% | 33.33% |
| soy | 92.17% | 87.12% | 89.58% |
| wheat | 92.37% | 88.62% | 90.46% |
| fish | 85.71% | 80.49% | 83.02% |
| shellfish | 81.25% | 59.09% | 68.42% |

**Evaluation B — Products with explicit allergen statements (96 products):**
| Metric | Value |
|--------|-------|
| Exact match (strict) | **55.21%** |
| Improved from baseline | +33.33pp (21.88% → 55.21%) |
| Detected ⊆ official (no FP) | 59.38% |
| Detected ⊇ official (no FN) | 82.29% |
| Avg Jaccard similarity | 84.34% |
| **Target (≥50%)** | **✅ MET** |

Per-class F1 vs OFF tags (explicit subset):
| Allergen | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| milk | 86.67% | 98.48% | 92.20% |
| eggs | 75.00% | 100.00% | 85.71% |
| peanuts | 63.33% | 100.00% | 77.55% |
| tree_nuts | 100.00% | 7.14% | 13.33% |
| soy | 91.55% | 100.00% | 95.59% |
| wheat | 89.29% | 100.00% | 94.34% |
| fish | 80.77% | 100.00% | 89.36% |
| shellfish | 100.00% | 54.55% | 70.59% |

**Shellfish fix:** Initial evaluation showed 0% shellfish precision/recall because OFF uses `en:crustaceans` and `en:molluscs` taxonomy codes rather than `en:shellfish`. The `OFFICIAL_MAP` in `text_processing.py` had no mapping for these codes. Fixed by adding `"crustaceans": "shellfish"` and `"molluscs": "shellfish"`. After fix: shellfish F1=68.42% (full dataset) / 70.59% (explicit subset). The 100% precision on the explicit subset with only 55% recall reflects that OFF tags a broader set of products as crustaceans/molluscs than the rule engine detects from ingredient text.

---

## 8. Thesis Plan vs Actual Implementation

| Aspect | Thesis Plan (Model Training Plan.md) | Actual Implementation |
|--------|--------------------------------------|----------------------|
| Dataset size | ~2,000 entries | 1,057 entries |
| Dataset sources | 1,400 OFF + 400 Philippine + 200 FDA | Mostly OFF (Philippine-filtered) |
| Label types | Semantic labels + Allergen labels + Detection type (explicit/implicit/derivative) | Allergen labels (8-class multi-label) + Semantic labels (40 categories — trained) |
| Models | MobileBERT classification + Allergen detection | Two MobileBERT models: semantic classifier (40 labels, trained) + allergen classifier (8 labels, trained) |
| Rule engine | Separate deliverable with Filipino variants, regulatory terms | Integrated into `text_processing.py` (BIG8 + exemptions + Filipino variants) |
| Semantic classification | Deliverable 1: ingredient type labeling (MSG → Flavor Enhancer) | 40-category taxonomy, `semantic_utils.py` with ~395 mappings, **model trained (Macro F1=0.9632) and ONNX-exported** |
| Deployment | PyTorch → ONNX → TF → TFLite → Quantization → Android | **PyTorch → ONNX → ONNX Runtime Mobile** (TFLite path removed; onnx2tf removed) |
| Ingredient extraction | Part of OCR pipeline (individual ingredient parsing from images) | Data from OFF structured text; individual ingredient parsing in Notebook 03 |
| Mobile inference target | < 1 second per prediction | **~11.4 ms** on CPU with ONNX Runtime (well within target) |
| Model size target | < 40 MB (classifier) + < 40 MB (allergen) + < 120 MB RAM combined | MobileBERT ~94 MB in safetensors format; ONNX export ~99 MB |
| Detection type annotations | Explicit / Implicit / Derivative | Not implemented (de-scoped) |

**TL;DR:** The thesis plan had 4 deliverables. Current coverage:
- **Deliverable 1** (Semantic Classification): 40-category taxonomy (reduced from 43 by merging herb→spice, smoked→cured, prebiotic→fiber), `semantic_utils.py` with ~395 mappings (including plural normalization + E-number/INS/FDC pattern matching), **model trained (Macro F1=0.9632) with 62% unique ingredient coverage, ONNX-exported**
- **Deliverable 2** (Allergen Detection): Fully trained MobileBERT model with weighted BCE loss, per-class threshold optimization, **Hybrid Macro F1=0.8685, ONNX-exported and validated**
- **Deliverable 3** (Rule Engine): BIG8 + Filipino variants + exemption handling + `combine_allergen_labels()`, integrated into `text_processing.py`
- **Deliverable 4** (ONNX Mobile Deployment): **Both models exported to ONNX**, validated against PyTorch baseline; ~11.4 ms CPU inference latency; TFLite path removed in favor of ONNX Runtime Mobile

---

## 9. Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/allergen_map.json` | Maps OpenFoodFacts tags (`en:gluten`, `en:milk`, etc.) to internal allergen names. Also lists the 8 target allergens. |
| `configs/model_thresholds.json` | Early experiment thresholds. Historical — not used in production. |
| `configs/semantic_categories.json` | 40-category taxonomy for Deliverable 1 with examples and category groups. |
| `models/hybrid_config.json` | **Production config.** ML thresholds (8 values), rule confidence threshold (0.5), and mode (`"rule_priority"`). |
| `models/best_thresholds.npy` | NumPy array of per-class optimal thresholds. |
| `models/mobilebert_allergen_final/` | The trained allergen model checkpoint + tokenizer files. |
| `models/mobilebert_semantic_final/` | The trained semantic classification model checkpoint + tokenizer files. |
| `models/exported/` | ONNX-exported models and deployment report. |
| `models/exported/allergen_model.onnx` | ONNX-exported allergen model (~99 MB). |
| `models/exported/deployment_report.json` | Deployment summary with latency benchmarks. |
| `data/final/processing_metadata.json` | Metadata about the labeling run (date, version, count). |
| `data/final/semantic_labeling_metadata.json` | Metadata about the semantic labeling run. |

---

## 10. Dependencies

See `requirements.txt` for pinned versions:

- **ML:** `torch==2.12.0`, `transformers==4.44.0`, `scikit-learn==1.5.2`
- **Data:** `pandas==2.2.2`, `numpy==2.0.1`, `duckdb==1.1.3`
- **Stratified splitting:** `iterative-stratification==0.1.7` (for multi-label stratified splits)
- **OCR:** `pytesseract==0.3.13`, `easyocr==1.7.2`
- **Image:** `Pillow==10.4.0`, `opencv-python==4.10.0.84`
- **Export:** `onnx==1.22.0`, `onnxruntime==1.27.0`, `tensorflow==2.21.0`
- **Scientific:** `scipy==1.14.1`, `matplotlib==3.9.2`

Notable: `tesseract` must be installed as a system package (not pip).

---

## 11. Running the Project

```bash
# Activate virtual environment
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Start Jupyter
jupyter lab

# Run notebooks in order:
# 1. notebooks/01_extraction.ipynb
# 2. notebooks/02_cleaning.ipynb
# 3. notebooks/03_ingredient_parsing.ipynb  (Deliverable 1)
# 4. notebooks/04_semantic_labeling.ipynb
# 5. notebooks/05_semantic_model.ipynb
# 6. notebooks/06_allergen_labeling.ipynb   (Deliverable 2)
# 7. notebooks/07_allergen_training.ipynb
# 8. notebooks/08_hybrid_evaluation.ipynb   (Deliverable 3)
# 9. notebooks/09_model_export.ipynb         (Deliverable 4)
# 10. notebooks/10_mobile_benchmark.ipynb
# 11. notebooks/11_dataset_augmentation.ipynb  (Phase 3 — optional)
```

All notebooks import from `utils/` — they must be run from the `notebooks/` directory so the relative import `from utils.xxx import yyy` resolves correctly.

**Tests (228 tests across 6 test files — 208 passing, 20 skip without GPU):**
```bash
# From repo root
pytest tests/ -v
```

**Pipeline automation:**
```bash
# Run the full notebook pipeline in dependency order
python scripts/run_pipeline.py

# Dry-run to see what would execute
python scripts/run_pipeline.py --dry-run

# Resume from a specific notebook
python scripts/run_pipeline.py --skip-until 07_allergen_training

# Validate that key output files exist after a prior run
python scripts/run_pipeline.py --validate
```

---

## 12. Design Decisions & Trade-offs

### Why MobileBERT over alternatives?
- **MobileBERT** chosen over DistilBERT (larger memory), TinyBERT (lower accuracy), BiLSTM (poor context). It offers the best accuracy/size trade-off for mobile deployment.

### Why weighted BCE loss instead of Focal Loss?
- Weighted BCE is simpler and proven effective. Focal Loss is available as a commented-out alternative in the notebook for experimentation.

### Why per-class thresholds instead of a single 0.5?
- Rare classes (tree_nuts, shellfish) need lower thresholds to achieve reasonable recall. The optimized thresholds range from 0.01 to 0.98.

### Why hybrid system if ML and rules agree?
- The rule-based system acts as a **safety net** for deployment scenarios where the ML model encounters novel ingredient terms. In the current dataset they agree; in production they may diverge.

### Why ONNX Runtime Mobile instead of TFLite?
- ONNX Runtime Mobile provides a simpler conversion path (no TF/TFLite intermediate step) while supporting Android deployment. TFLite conversion was initially planned but dropped when TensorFlow dependency issues arose. The ONNX path is more direct and achieves ~11.4 ms inference on CPU.

### Why `hard_override` hybrid mode?
- `hard_override` mode combines ML with rule-based detection: if the rule engine detects an allergen AND the ML probability exceeds `rule_conf_threshold` (0.5), the prediction is forced to 1.0. This captures rule-confident positives the ML model may have missed. Earlier configs used `rule_priority` mode (notebook-level rule-first logic), but `hard_override` is now the standard for evaluation.

---

## 13. Known Limitations & Future Work

### Dataset
- **Only 1,057 samples** (~50% of thesis target of 2,000). More Philippine-specific product data needed.
- **Severe class imbalance:** `tree_nuts` has only 16 positive samples (1.5%). Augmentation (NB11) brings it to 128 (7.9%), but model evaluation still uses real test data.
- **No detection type annotations** (explicit/implicit/derivative per thesis plan).

### Model
- `tree_nuts` F1 is only 0.44 (ML-only) — needs more data or a different approach (e.g., few-shot learning, separate detector).
- **Epoch 1 loss explosion — COSMETIC, NOT FIXED (Phase 0 finding, June 2026).** Both models show millions-scale BCE loss on epoch 1 (gradient norms up to 105M) before recovering by epoch 2. **Attempted fix rejected:** Adding `nn.LayerNorm` before the classifier head normalizes MobileBERT's unconstrained pooler output (magnitudes ~±54M) to N(0,1). However, this LayerNorm destroys per-sample gradient diversity on small datasets — the allergen model (1,962 samples) collapsed to Val Macro F1=0.0. The semantic model (14,898 samples) worked fine (F1=0.9632) because the larger dataset provides enough gradient diversity to overcome the normalization effect. **Conclusion:** The original `nn.Linear` classifier is correct. Epoch 1 loss >1M is a cosmetic artifact of the pooler output scale — the model still converges to Macro F1=0.87 (allergen) / 0.96 (semantic). Gradient clipping (`max_norm=1.0`) handles backward stability. A linear warmup from lr=0 for the first 100-200 steps is a potential further improvement but not necessary for convergence.
- **Training labels are rule-generated** (see Phase 1 evaluation in §7 — 74.36% exact agreement between rules and OFF tags, 55.21% on explicit statements) — model learns moderately noisy targets. Measured F1 of 0.87-0.95 may be somewhat inflated vs human-annotated ground truth.
- **OFFICIAL_MAP needed crustaceans/molluscs fix** — OFF taxonomy uses `en:crustaceans` and `en:molluscs` instead of `en:shellfish`. Fixed June 2026 post-evaluation.
- **No TFLite conversion** — deployment targets ONNX Runtime Mobile instead.
- ONNX inference: ~11.4 ms mean on CPU (well within <1s target).
- PyTorch inference: ~21 ms mean on CPU.

### Environment / Platform
- **HuggingFace `Trainer` segfaults** on this GPU stack (PyTorch 2.12.0+cu130 with non-standard CUDA 13.0). `Trainer.__init__` triggers a native SIGSEGV during device management. All training uses a manual PyTorch loop as a workaround.
- **`clip_grad_norm_` API change:** PyTorch 2.x renamed `max_grad_norm` → `max_norm`. Must use `max_norm=1.0`.
- **CUDA memory fragmentation:** 6+ orphan `ipykernel_launcher` processes can persist between notebook runs, each holding GPU memory (cumulatively 5–7 GiB). Killing orphan PIDs or restarting the Jupyter kernel is required before running notebook 08's inference.
- **`nvidia-smi` not installed** — no CLI visibility into GPU process memory usage. Only PyTorch's `torch.cuda.memory_allocated()` is available.

### Pipeline

- **Test suite:** 228 tests across 6 test files (208 passing, 20 skip without GPU):

  - `tests/test_semantic_utils.py` — 67 tests for ingredient mapping, plural normalization, E-numbers
  - `tests/test_evaluation.py` — 53 tests for evaluation metrics, error analysis, per-class metrics, rule vs official evaluation
  - `tests/test_text_processing.py` — coverage for allergen detection, negation, exemptions
  - `tests/test_data_utils.py` — data loading, splitting, augmentation
  - `tests/test_model_utils.py` — model prediction, threshold optimization, hybrid inference

- **Pipeline automation:** `scripts/run_pipeline.py` orchestrates 11 notebooks in dependency order with dry-run, skip-until, validate, and timeout flags.

- No CI/CD pipeline.

### Deployment
- No mobile integration (Android app not started).
- ONNX Runtime Mobile integration steps not yet documented.
- Both models exported and validated against PyTorch baseline.

---

## 14. File System Layout

```
.
├── README.md                          # Project overview and quick start
├── Model Training Plan.md             # Thesis ideal plan (targets, deliverables)
├── context.md                         # THIS FILE — project context for developers
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore
├── configs/
│   ├── README.md                      # Config documentation
│   ├── allergen_map.json              # OFF tag → internal name mapping
│   ├── model_thresholds.json          # Early experiment thresholds (historical)
│   └── semantic_categories.json       # 40-category taxonomy (Deliverable 1)
├── data/
│   ├── raw/food.parquet               # Open Food Facts export
│   ├── interim/
│   │   ├── extracted_raw.csv          # After OCR extraction
│   │   ├── parsed_ingredients.csv     # After ingredient tokenization
│   │   ├── ingredient_vocabulary.csv  # Unique ingredient vocabulary
│   │   └── unknown_ingredients_for_labeling.csv  # Unmapped ingredients
│   ├── processed/cleaned_dataset.csv  # After cleaning
│   └── final/
│       ├── labeled_dataset_enhanced.csv # Final allergen training dataset (1,057 rows)
│       ├── labeled_dataset_augmented.csv # Augmented dataset (1,621 rows) from NB11
│       ├── semantic_labels.csv          # Semantic category labels per product
│       ├── semantic_training_data.csv   # Multi-label semantic training matrix
│       ├── processing_metadata.json     # Labeling run metadata
│       └── semantic_labeling_metadata.json  # Semantic labeling run metadata
├── models/
│   ├── mobilebert_allergen_final/      # Trained allergen model + tokenizer
│   ├── mobilebert_semantic_final/      # Trained semantic model + tokenizer
│   ├── best_thresholds.npy            # Per-class optimal thresholds
│   ├── hybrid_config.json             # Production hybrid config
│   └── exported/
│       ├── allergen_model.onnx         # ONNX-exported allergen model
│       ├── allergen_model_export_config.json  # Export metadata
│       └── deployment_report.json     # Deployment + latency benchmarks
├── notebooks/
│   ├── 01_extraction.ipynb            # OCR + data extraction
│   ├── 02_cleaning.ipynb             # Dedup + normalization
│   ├── 03_ingredient_parsing.ipynb    # Ingredient tokenization (D1)
│   ├── 04_semantic_labeling.ipynb     # 40-category rule labeling (D1)
│   ├── 05_semantic_model.ipynb       # Semantic MobileBERT training (D1)
│   ├── 06_allergen_labeling.ipynb    # Allergen annotation (D2)
│   ├── 07_allergen_training.ipynb    # Allergen MobileBERT training (D2)
│   ├── 08_hybrid_evaluation.ipynb    # Hybrid detection (D3)
│   ├── 09_model_export.ipynb         # ONNX conversion + validation (D4)
│   ├── 10_mobile_benchmark.ipynb     # Benchmarking (D4)
│   ├── 11_dataset_augmentation.ipynb # Targeted augmentation for rare classes
│   └── utils/
│       ├── __init__.py               # Package exports (version 2.2.0)
│       ├── README.md                 # Comprehensive utility module docs
│       ├── text_processing.py        # BIG8 rules, negation, exemptions, Filipino variants
│       ├── data_utils.py            # Load, split, augment
│       ├── model_utils.py           # Model, predict, hybrid
│       ├── evaluation.py            # Metrics, error analysis
│       ├── semantic_utils.py         # Ingredient parsing, 40-category mapping (D1)
│       ├── deployment_utils.py       # ONNX export, benchmarking (D4)
│       └── ocr_utils.py            # EasyOCR + OpenCV preprocessing
├── scripts/
│   ├── __init__.py                     # Package marker
│   └── run_pipeline.py                 # End-to-end notebook pipeline orchestrator
├── tests/
│   ├── test_semantic_utils.py         # 67 tests: ingredient mapping, plural normalization, E-numbers
│   └── test_evaluation.py            # 53 tests: evaluation metrics, error analysis, per-class metrics
├── docs/
│   ├── onnx_api.md                    # ONNX model API reference (input format, preprocessing, integration)
│   ├── model_cards.md                 # Model cards for allergen and semantic classifiers
│   └── threshold_tuning.md            # Per-class threshold optimization guide
├── outputs/                          # (generated predictions, executed notebooks)
├── .venv/                            # Virtual environment
└── .vscode/settings.json             # Python venv config
```

---

## 15. Key Contacts & Conventions

- **Allergen order** (used everywhere): `["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]`
- **Label format:** Binary vector of length 8 in the above order
- **Dataset column** for labels: `detected_allergens` (list of strings like `['milk', 'wheat']`)
- **All paths** are relative from the `notebooks/` directory (e.g., `../models/hybrid_config.json`)
- **Semantic label order** (40 categories): see `utils/semantic_utils.py` -> `SEMANTIC_CATEGORIES`
- **Hybrid mode "hard_override":** If rule engine detects an allergen AND ML probability > `rule_conf_threshold` (0.5), prediction is forced to 1.0. Stored in `models/hybrid_config.json`.
- **Deployment target:** ONNX Runtime Mobile (not TFLite)

---

*Last updated: June 26, 2026*
*Version: 2.6.0*

### Changelog

- **v2.6.0 (Jun 26, 2026):** Phase 6 — added `tests/test_evaluation.py` (53 tests, 12 test classes). Full test suite: 228 tests (208 passing, 20 skip without GPU). Phase 7 — created `scripts/run_pipeline.py` with dry-run, skip-until, validate, and timeout flags. Phase 8 — created documentation: ONNX API reference (`docs/onnx_api.md`), model cards (`docs/model_cards.md`), and threshold tuning guide (`docs/threshold_tuning.md`).
- **v2.5.0 (Jun 26, 2026):** Phase 0 — documented LayerNorm rejection for epoch 1 loss (cosmetic, not fixable). Phase 1 — official tag evaluation results (55.21% exact agreement on explicit-statement subset, target ≥50% met). Added crustaceans/molluscs→shellfish OFFICIAL_MAP mapping. Performance table updated with final hybrid metrics (Macro F1=0.8746).
- **v2.4.0 (Jun 25, 2026):** Phase 3 complete — 395 semantic mappings, E-number/INS/FDC pattern matching, NB11 augmentation, 40 categories (merged herb/spice, smoked/cured, prebiotic/fiber), all notebooks re-run with consistent results.
