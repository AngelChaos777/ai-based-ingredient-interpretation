# Project Context

> **Food Label Transparency for Filipino Consumers — AI-Based Ingredient Interpretation**
> A thesis project that detects FDA Big-8 allergens from Philippine food product labels using OCR, rule-based parsing, and fine-tuned MobileBERT models in a hybrid detection pipeline. Also performs semantic ingredient classification (41 categories) as a supporting deliverable.

---

## 1. What This Project Does

This project builds a system that takes **food product ingredient text** (extracted from packaging images via OCR or from structured data like Open Food Facts) and:

1. **Classifies each ingredient** into 41 semantic categories (additive, preservative, sweetener, protein source, etc.)
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
                    │  Rule-based 41-category mapping          │
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
| `semantic_utils.py` | Ingredient parsing, semantic category mapping | `parse_ingredient_list()`, `ingredient_to_categories()`, `build_semantic_label_matrix()`, `INGREDIENT_SEMANTIC_MAP` (300+ mappings) |
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
| `data/final/semantic_labels.csv` | Semantic category labels per product (41 categories) |
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
| milk | 0.05 |
| eggs | 0.02 |
| peanuts | 0.35 |
| tree_nuts | 0.50 |
| soy | 0.73 |
| wheat | 0.13 |
| fish | 0.08 |
| shellfish | 0.50 |

**Key insight:** Thresholds vary widely by class. Low thresholds (milk, eggs) reflect easy-to-detect allergens where the model assigns very low probabilities even for true positives due to class imbalance or ambiguous patterns. High thresholds (soy, tree_nuts, shellfish) indicate the model is more confident but needs to be conservative to avoid false positives.

### 6B. Semantic Classification Model (Trained, ONNX Exported)

**Architecture**
- **Base model:** `google/mobilebert-uncased` (HuggingFace)
- **Task:** Multi-label classification (43 semantic outputs with sigmoid)
- **Same MobileBERT architecture** as the allergen model
- **Labels:** 43 semantic categories across 7 groups
- **Status:** Fully trained, exported to ONNX

**Semantic Categories (7 groups, 43 labels)**
| Group | Categories |
|-------|-----------|
| Additives & Preservatives | food_additive, preservative, antioxidant, acidulant, colorant |
| Flavor & Sweeteners | flavor_enhancer, sweetener, sugar, added_sugar, flavoring, spice, herb, salt, yeast |
| Functional Ingredients | emulsifier, stabilizer, thickener, gelling_agent, leavening_agent, humectant |
| Macronutrients | fat_source, oil_source, protein_source, carbohydrate_source, fiber |
| Origin & Derivation | animal_derived, plant_derived, milk_derivative, egg_derivative, soy_derivative, wheat_derivative, fermented, smoked, cured |
| Micronutrients | vitamin, mineral |
| Biological | enzyme, culture, prebiotic, salt_substitute |

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
  "ml_thresholds": [0.05, 0.02, 0.35, 0.5, 0.73, 0.13, 0.08, 0.5],
  "rule_conf_threshold": 0.5,
  "mode": "rule_priority"
}
```

### Batched Inference (CUDA OOM Fix)

Notebook 08's `predict_ml()` previously ran all test samples as a single batch, causing CUDA OOM on the 8 GiB RTX 3050 when orphan Jupyter kernels from prior training runs held ~6.6 GiB of VRAM. Fixed by adding a `batch_size` parameter (default 8) that processes texts in small batches with periodic `torch.cuda.empty_cache()` calls. Both `predict_ml()` and `hybrid_predict()` now support `batch_size` — pass it explicitly to control GPU memory usage.

**Orphan kernel issue:** 6+ stale `ipykernel_launcher` processes can persist between runs, each holding GPU memory. Restarting the Jupyter kernel or killing orphan PIDs frees the memory. Notebook 08 now calls `torch.cuda.empty_cache()` at startup to mitigate fragmentation.

### Performance (test set) — Verified End-to-End

Results from the complete, freshly-verified pipeline (batch_size=8, max_length=221, hard_override mode):

**ML Only:**
| Metric | Value |
|--------|-------|
| Micro F1 | 0.9256 |
| Macro F1 | 0.8487 |
| Weighted Avg F1 | 0.9335 |

**Hybrid (hard override, rule_conf_threshold=0.5):**
| Metric | Value |
|--------|-------|
| Micro F1 | 0.9431 |
| Macro F1 | 0.8822 |
| Weighted Avg F1 | 0.9535 |

Per-class test set F1 scores (Hybrid — hard override):
| Allergen | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| milk | 0.96 | 0.95 | 0.95 |
| eggs | 0.93 | 1.00 | 0.97 |
| peanuts | 1.00 | 1.00 | 1.00 |
| tree_nuts | 0.20 | 0.67 | 0.31 |
| soy | 0.98 | 0.93 | 0.95 |
| wheat | 1.00 | 0.94 | 0.97 |
| fish | 1.00 | 1.00 | 1.00 |
| shellfish | 0.83 | 1.00 | 0.91 |

**Hard override mode improves over ML-only** — macro F1 rises from 0.8487 → 0.8822 (+3.9%). The rule-based system, when its confidence exceeds the threshold, catches missed positives that the ML model assigned low probability to. On this test set, **14 samples had prediction changes** between ML-only and hybrid modes.

**Notable gap:** `tree_nuts` still suffers from very few positive samples (only 3 in test set, F1=0.31). The low precision (0.20) indicates the hybrid override may trigger false positives for this rare class. This is a known failure mode of the current dataset.

---

## 8. Thesis Plan vs Actual Implementation

| Aspect | Thesis Plan (Model Training Plan.md) | Actual Implementation |
|--------|--------------------------------------|----------------------|
| Dataset size | ~2,000 entries | 1,057 entries |
| Dataset sources | 1,400 OFF + 400 Philippine + 200 FDA | Mostly OFF (Philippine-filtered) |
| Label types | Semantic labels + Allergen labels + Detection type (explicit/implicit/derivative) | Allergen labels (8-class multi-label) + Semantic labels (43 categories — trained) |
| Models | MobileBERT classification + Allergen detection | Two MobileBERT models: semantic classifier (43 labels, trained) + allergen classifier (8 labels, trained) |
| Rule engine | Separate deliverable with Filipino variants, regulatory terms | Integrated into `text_processing.py` (BIG8 + exemptions + Filipino variants) |
| Semantic classification | Deliverable 1: ingredient type labeling (MSG → Flavor Enhancer) | 43-category taxonomy defined, `semantic_utils.py` with 300+ mappings, **model trained and ONNX-exported** |
| Deployment | PyTorch → ONNX → TF → TFLite → Quantization → Android | **PyTorch → ONNX → ONNX Runtime Mobile** (TFLite path removed; onnx2tf removed) |
| Ingredient extraction | Part of OCR pipeline (individual ingredient parsing from images) | Data from OFF structured text; individual ingredient parsing in Notebook 03 |
| Mobile inference target | < 1 second per prediction | **~11.4 ms** on CPU with ONNX Runtime (well within target) |
| Model size target | < 40 MB (classifier) + < 40 MB (allergen) + < 120 MB RAM combined | MobileBERT ~94 MB in safetensors format; ONNX export ~99 MB |
| Detection type annotations | Explicit / Implicit / Derivative | Not implemented (de-scoped) |

**TL;DR:** The thesis plan had 4 deliverables. Current coverage:
- **Deliverable 1** (Semantic Classification): 43-category taxonomy, `semantic_utils.py` with 300+ mappings, **model trained and ONNX-exported**
- **Deliverable 2** (Allergen Detection): Fully trained MobileBERT model with weighted BCE loss, per-class threshold optimization, **ONNX-exported and validated**
- **Deliverable 3** (Rule Engine): BIG8 + Filipino variants + exemption handling + `combine_allergen_labels()`, integrated into `text_processing.py`
- **Deliverable 4** (ONNX Mobile Deployment): **Both models exported to ONNX**, validated against PyTorch baseline; ~11.4 ms CPU inference latency; TFLite path removed in favor of ONNX Runtime Mobile

---

## 9. Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/allergen_map.json` | Maps OpenFoodFacts tags (`en:gluten`, `en:milk`, etc.) to internal allergen names. Also lists the 8 target allergens. |
| `configs/model_thresholds.json` | Early experiment thresholds. Historical — not used in production. |
| `configs/semantic_categories.json` | 43-category taxonomy for Deliverable 1 with examples and category groups. |
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
```

All notebooks import from `utils/` — they must be run from the `notebooks/` directory so the relative import `from utils.xxx import yyy` resolves correctly.

---

## 12. Design Decisions & Trade-offs

### Why MobileBERT over alternatives?
- **MobileBERT** chosen over DistilBERT (larger memory), TinyBERT (lower accuracy), BiLSTM (poor context). It offers the best accuracy/size trade-off for mobile deployment.

### Why weighted BCE loss instead of Focal Loss?
- Weighted BCE is simpler and proven effective. Focal Loss is available as a commented-out alternative in the notebook for experimentation.

### Why per-class thresholds instead of a single 0.5?
- Rare classes (tree_nuts, shellfish) need lower thresholds to achieve reasonable recall. The optimized thresholds range from 0.02 to 0.73.

### Why hybrid system if ML and rules agree?
- The rule-based system acts as a **safety net** for deployment scenarios where the ML model encounters novel ingredient terms. In the current dataset they agree; in production they may diverge.

### Why ONNX Runtime Mobile instead of TFLite?
- ONNX Runtime Mobile provides a simpler conversion path (no TF/TFLite intermediate step) while supporting Android deployment. TFLite conversion was initially planned but dropped when TensorFlow dependency issues arose. The ONNX path is more direct and achieves ~11.4 ms inference on CPU.

### Why `rule_priority` mode?
- In the hybrid detection config, `rule_priority` mode indicates the labeling notebook weights rule-based matches higher than ML predictions. This is handled at the notebook level rather than in `hybrid_predict()`.

---

## 13. Known Limitations & Future Work

### Dataset
- **Only 1,057 samples** (~50% of thesis target of 2,000). More Philippine-specific product data needed.
- **Severe class imbalance:** `tree_nuts` has only 16 positive samples.
- **No detection type annotations** (explicit/implicit/derivative per thesis plan).

### Model
- `tree_nuts` F1 is only 0.31–0.50 — needs more data or a different approach.
- **Semantic model trained but not evaluated** — Notebook 05 runs but evaluation metrics not yet published.
- **No TFLite conversion** — deployment targets ONNX Runtime Mobile instead.
- ONNX inference: ~11.4 ms mean on CPU (well within <1s target).
- PyTorch inference: ~21 ms mean on CPU.

### Environment / Platform
- **HuggingFace `Trainer` segfaults** on this GPU stack (PyTorch 2.12.0+cu130 with non-standard CUDA 13.0). `Trainer.__init__` triggers a native SIGSEGV during device management. All training uses a manual PyTorch loop as a workaround.
- **`clip_grad_norm_` API change:** PyTorch 2.x renamed `max_grad_norm` → `max_norm`. Must use `max_norm=1.0`.
- **CUDA memory fragmentation:** 6+ orphan `ipykernel_launcher` processes can persist between notebook runs, each holding GPU memory (cumulatively 5–7 GiB). Killing orphan PIDs or restarting the Jupyter kernel is required before running notebook 08's inference.
- **`nvidia-smi` not installed** — no CLI visibility into GPU process memory usage. Only PyTorch's `torch.cuda.memory_allocated()` is available.

### Pipeline
- No formal test suite for the utility modules.
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
│   └── semantic_categories.json       # 43-category taxonomy (Deliverable 1)
├── data/
│   ├── raw/food.parquet               # Open Food Facts export
│   ├── interim/
│   │   ├── extracted_raw.csv          # After OCR extraction
│   │   ├── parsed_ingredients.csv     # After ingredient tokenization
│   │   ├── ingredient_vocabulary.csv  # Unique ingredient vocabulary
│   │   └── unknown_ingredients_for_labeling.csv  # Unmapped ingredients
│   ├── processed/cleaned_dataset.csv  # After cleaning
│   └── final/
│       ├── labeled_dataset_enhanced.csv # Final allergen training dataset
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
│   ├── 04_semantic_labeling.ipynb     # 43-category rule labeling (D1)
│   ├── 05_semantic_model.ipynb       # Semantic MobileBERT training (D1)
│   ├── 06_allergen_labeling.ipynb    # Allergen annotation (D2)
│   ├── 07_allergen_training.ipynb    # Allergen MobileBERT training (D2)
│   ├── 08_hybrid_evaluation.ipynb    # Hybrid detection (D3)
│   ├── 09_model_export.ipynb         # ONNX conversion + validation (D4)
│   ├── 10_mobile_benchmark.ipynb     # Benchmarking (D4)
│   └── utils/
│       ├── __init__.py               # Package exports (version 2.2.0)
│       ├── README.md                 # Comprehensive utility module docs
│       ├── text_processing.py        # BIG8 rules, negation, exemptions, Filipino variants
│       ├── data_utils.py            # Load, split, augment
│       ├── model_utils.py           # Model, predict, hybrid
│       ├── evaluation.py            # Metrics, error analysis
│       ├── semantic_utils.py         # Ingredient parsing, 43-category mapping (D1)
│       ├── deployment_utils.py       # ONNX export, benchmarking (D4)
│       └── ocr_utils.py            # EasyOCR + OpenCV preprocessing
├── outputs/                          # (generated predictions)
├── .venv/                            # Virtual environment
└── .vscode/settings.json             # Python venv config
```

---

## 15. Key Contacts & Conventions

- **Allergen order** (used everywhere): `["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]`
- **Label format:** Binary vector of length 8 in the above order
- **Dataset column** for labels: `detected_allergens` (list of strings like `['milk', 'wheat']`)
- **All paths** are relative from the `notebooks/` directory (e.g., `../models/hybrid_config.json`)
- **Semantic label order** (43 categories): see `utils/semantic_utils.py` -> `SEMANTIC_CATEGORIES`
- **Hybrid mode "rule_priority":** Used in config to indicate notebook-level rule-priority labeling. This mode is not in `hybrid_predict()` — it is consumed at the notebook level.
- **Deployment target:** ONNX Runtime Mobile (not TFLite)

---

*Last updated: June 26, 2026*
*Version: 2.3.0*
