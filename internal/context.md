# Project Context

> **Food Label Transparency for Filipino Consumers — AI-Based Ingredient Interpretation**
> A thesis project that detects FDA Big-8 allergens from Philippine food product labels using OCR, rule-based parsing, and a fine-tuned MobileBERT model in a hybrid detection pipeline.

---

## 1. What This Project Does

This project builds a system that takes **food product ingredient text** (extracted from packaging images via OCR or from structured data like Open Food Facts) and detects which of the **8 major FDA-recognized allergens** are present in each product. The core contribution is a **hybrid detection pipeline** that combines:

- **A fine-tuned MobileBERT model** (multi-label text classifier) for semantic understanding
- **A comprehensive rule-based keyword engine** (the BIG8 database) for high-recall explicit matching
- **Exemption/negation handling** to reduce false positives (e.g., "milk-free", "soy lecithin" exemption)

The system is designed for **eventual mobile deployment** on Android (TensorFlow Lite) — prioritizing small model size, fast inference, and offline operation over benchmark performance.

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
                    ┌────────────────▼────────────────────────┐
                    │      03_labeling_enhanced.ipynb          │
                    │  Rule-based allergen annotation →        │
                    │  multi-label training dataset            │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │          04_model_training.ipynb         │
                    │  MobileBERT fine-tune (weighted BCE)     │
                    │  Stratified split + augmentation         │
                    │  Threshold optimization per class        │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │          05_hybrid.ipynb                 │
                    │  ML predictions + rule-based override    │
                    │  → hybrid_config.json                   │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │    [06_ocr_hybrid_pipeline.ipynb]        │
                    │    (deleted — planned end-to-end demo)   │
                    └─────────────────────────────────────────┘
```

**Important:** The notebooks assume they are executed from the `notebooks/` directory (relative paths like `../data/...` and `../models/...`).

### Utility Modules (`notebooks/utils/`)

The 4 utility modules are the **single source of truth** for all logic — notebooks should import from here rather than redefining code:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `text_processing.py` | Allergen detection, text cleaning, negation handling | `detect_allergens_rule_based()`, `rule_match()`, `BIG8` dict, `apply_exemptions()`, `clean_html()`, `tokenize_ingredients()`, `combine_allergen_labels()` |
| `data_utils.py` | Data loading, stratified splitting, augmentation | `load_labeled_data()`, `create_stratified_splits()`, `augment_dataframe()`, `get_data_directories()`, `save_metadata()` |
| `model_utils.py` | Model loading, prediction, hybrid inference | `load_model_and_tokenizer()`, `predict_ml()`, `hybrid_predict()`, `compute_class_weights()`, `find_best_thresholds()` |
| `evaluation.py` | Metrics, error analysis, reporting | `print_classification_report()`, `error_analysis()`, `compute_per_class_metrics()`, `jaccard()` |

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
| `data/processed/cleaned_dataset.csv` | After dedup + normalization |
| `data/final/labeled_dataset_enhanced.csv` | The final multi-label training dataset (1,057 rows) |
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

## 6. ML Model: MobileBERT

### Architecture
- **Base model:** `google/mobilebert-uncased` (HuggingFace)
- **Task:** Multi-label classification (8 outputs with sigmoid)
- **Hidden size:** 512 (with bottleneck 128)
- **Layers:** 24 transformer blocks
- **Parameters:** ~25M (vs 110M for BERT-base)
- **Classifier head:** Newly initialized `MobileBertForSequenceClassification` (8 labels)

### Training Configuration (actual)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.1 |
| Epochs | 15 (early stopping patience=3) |
| Batch Size | 8 |
| Max Sequence Length | 221 (95th percentile) |
| Weight Decay | 0.01 |
| Max Grad Norm | 1.0 |
| Loss | Weighted BCEWithLogitsLoss (inverse frequency weights) |
| Best Model Selection | Macro F1 |

### Thesis Plan Config (from Model Training Plan.md)
| Parameter | Target Range | Actual Used |
|-----------|-------------|-------------|
| Learning Rate | 1e-5 to 5e-5 | 2e-5 |
| Epochs | 5 to 20 | 15 |
| Batch Size | 8 to 32 | 8 |
| Weight Decay | 0 to 0.1 | 0.01 |

### Training Time
- **~9 minutes** on CPU (no GPU available in this run)
- The thesis mentions ~25 min on RTX 3060 (6 GB VRAM)

### Optimized Thresholds (per class)
These are per-class probability thresholds found on the validation set (stored in `models/best_thresholds.npy` and `models/hybrid_config.json`):

| Allergen | Threshold |
|----------|-----------|
| milk | 0.08 |
| eggs | 0.57 |
| peanuts | 0.04 |
| tree_nuts | 0.01 |
| soy | 0.17 |
| wheat | 0.03 |
| fish | 0.01 |
| shellfish | 0.03 |

**Key insight:** Most thresholds are well below 0.5 — the model learns conservative probabilities but the optimal operating point is low to maximize recall (which aligns with the thesis priority of recall over precision for allergens).

---

## 7. Hybrid Detection System

### Modes
The hybrid system (`model_utils.hybrid_predict()`) combines ML predictions with rule-based logic. Three modes:

1. **`hard_override`** (default, used in config): If rule-based detects an allergen AND ML probability > `rule_conf_threshold` (0.5), force prediction = 1.
2. **`soft`**: Blend ML probability and rule signal: `final_prob = ml_prob + alpha * rule_present`. Then threshold at 0.5.
3. **`high_confidence_bypass`**: If rule detects AND probability is between `rule_conf_threshold` and 0.9, set prediction = 1 (catches high-confidence misses).

### Current Config (`models/hybrid_config.json`)
```json
{
  "ml_thresholds": [0.08, 0.57, 0.04, 0.01, 0.17, 0.03, 0.01, 0.03],
  "rule_conf_threshold": 0.5,
  "mode": "rule_priority"
}
```

### Performance (test set)

**ML-only** vs **Hybrid** — in the current dataset, the rule-based and ML systems largely agree. The hybrid mode showed zero differences from ML-only on the test set, suggesting that for this dataset, the ML model has already learned the patterns the rule system captures.

| Metric | value |
|--------|-------|
| Micro F1 | 0.9303 |
| Macro F1 | 0.8595 |
| Weighted Avg F1 | 0.9337 |

Per-class test set F1 scores:
| Allergen | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| milk | 0.96 | 0.95 | 0.95 |
| eggs | 1.00 | 0.93 | 0.96 |
| peanuts | 0.86 | 0.92 | 0.89 |
| tree_nuts | 0.40 | 0.67 | 0.50 |
| soy | 0.89 | 0.95 | 0.92 |
| wheat | 0.96 | 0.98 | 0.97 |
| fish | 0.73 | 1.00 | 0.85 |
| shellfish | 0.71 | 1.00 | 0.83 |

**Notable gap:** `tree_nuts` suffers from very few positive samples (only 3 in test set). This is a known failure mode of the current dataset.

---

## 8. Thesis Plan vs Actual Implementation

| Aspect | Thesis Plan (Model Training Plan.md) | Actual Implementation |
|--------|--------------------------------------|----------------------|
| Dataset size | ~2,000 entries | 1,057 entries |
| Dataset sources | 1,400 OFF + 400 Philippine + 200 FDA | Mostly OFF (Philippine-filtered) |
| Label types | Semantic labels (additive, fat, etc.) + Allergen labels + Detection type (explicit/implicit/derivative) | Allergen labels only (8-class multi-label) |
| Models | MobileBERT classification + Allergen detection (could be 2 models) | Single MobileBERT for multi-label allergen classification |
| Rule engine | Separate deliverable with Filipino variants, regulatory terms | Integrated into `text_processing.py` (BIG8 + exemptions) |
| Semantic classification | Deliverable 1: ingredient type labeling (MSG → Flavor Enhancer) | Not implemented |
| Deployment | PyTorch → ONNX → TF → TFLite → Quantization → Android | Model trained and saved; no TFLite conversion yet |
| Ingredient extraction | Part of OCR pipeline (individual ingredient parsing from images) | Data comes from OFF structured text; OCR exists but is optional |
| Mobile inference target | < 1 second per prediction | ~12 ms on CPU per ingredient (potential benchmark) |
| Model size target | < 40 MB (classifier) + < 40 MB (allergen) + < 120 MB RAM combined | MobileBERT ~25M params — size not yet measured for TFLite |

**TL;DR:** The thesis plan was ambitious with 4 deliverables. The actual implementation successfully delivered the **core allergen detection** pipeline (Deliverable 2 + 3 partially) using a single MobileBERT model. The semantic classification model (Deliverable 1) and TFLite deployment (Deliverable 4) are future work.

---

## 9. Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/allergen_map.json` | Maps OpenFoodFacts tags (`en:gluten`, `en:milk`, etc.) to internal allergen names. Also lists the 8 target allergens. |
| `configs/model_thresholds.json` | Early experiment thresholds. Not used in production — see `hybrid_config.json`. |
| `models/hybrid_config.json` | **Production config.** ML thresholds (8 values), rule confidence threshold (0.5), and mode (`"rule_priority"`). |
| `models/best_thresholds.npy` | NumPy array of per-class optimal thresholds (same as `ml_thresholds` in hybrid_config). |
| `models/mobilebert_allergen_final/` | The trained model checkpoint + tokenizer files. |
| `data/final/processing_metadata.json` | Metadata about the labeling run (date, version, count). |

---

## 10. Dependencies

See `requirements.txt` for pinned versions:

- **ML:** `torch==2.4.1`, `transformers==4.44.0`, `scikit-learn==1.5.2`
- **Data:** `pandas==2.2.2`, `numpy==2.0.1`, `duckdb==1.1.3`
- **Stratified splitting:** `iterative-stratification==0.1.7` (for multi-label stratified splits)
- **OCR:** `pytesseract==0.3.13`, `easyocr==1.8.3`
- **Image:** `Pillow==10.4.0`

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
# 3. notebooks/03_labeling_enhanced.ipynb
# 4. notebooks/04_model_training.ipynb
# 5. notebooks/05_hybrid.ipynb
```

All notebooks import from `utils/` — they must be run from the `notebooks/` directory so the relative import `from utils.xxx import yyy` resolves correctly.

---

## 12. Design Decisions & Trade-offs

### Why MobileBERT over alternatives?
- **MobileBERT** chosen over DistilBERT (larger memory), TinyBERT (lower accuracy), BiLSTM (poor context). It offers the best accuracy/size trade-off for mobile deployment.

### Why weighted BCE loss instead of Focal Loss?
- Weighted BCE is simpler and proven effective. Focal Loss is available as a commented-out alternative in the notebook for experimentation.

### Why per-class thresholds instead of a single 0.5?
- Rare classes (tree_nuts, shellfish) need lower thresholds to achieve reasonable recall. The optimized thresholds range from 0.01 to 0.57.

### Why hybrid system if ML and rules agree?
- The rule-based system acts as a **safety net** for deployment scenarios where the ML model encounters novel ingredient terms. In the current dataset they agree; in production they may diverge.

### Why no TFLite deployment yet?
- The thesis prioritized model training and evaluation accuracy first. TFLite conversion, quantization, and Android integration are documented as the next phase.

---

## 13. Known Limitations & Future Work

### Dataset
- **Only 1,057 samples** (~50% of thesis target of 2,000). More Philippine-specific product data needed.
- **Severe class imbalance:** `tree_nuts` has only 16 positive samples.
- **No semantic labels** (ingredient type classification per Deliverable 1).
- **No detection type annotations** (explicit/implicit/derivative per thesis plan).

### Model
- `tree_nuts` F1 is only 0.50 — needs more data or a different approach.
- **No TFLite conversion yet** — the thesis targets Android deployment.
- Inference latency measured only in notebook context (~12 ms/ingredient on CPU).

### Pipeline
- Notebook 06 (end-to-end OCR+hybrid demo) was deleted.
- No formal test suite for the utility modules.
- No CI/CD pipeline.

### Deployment
- Filipino ingredient variants/vocabulary not yet integrated into the rule engine.
- No mobile integration.

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
│   └── model_thresholds.json          # Early experiment thresholds
├── data/
│   ├── raw/food.parquet               # Open Food Facts export
│   ├── interim/extracted_raw.csv      # After OCR extraction
│   ├── processed/cleaned_dataset.csv   # After cleaning
│   └── final/
│       ├── labeled_dataset_enhanced.csv # Final multi-label dataset
│       └── processing_metadata.json    # Labeling run metadata
├── models/
│   ├── mobilebert_allergen_final/      # Trained model + tokenizer
│   ├── best_thresholds.npy            # Per-class optimal thresholds
│   └── hybrid_config.json             # Production hybrid config
├── notebooks/
│   ├── 01_extraction.ipynb            # OCR + data extraction
│   ├── 02_cleaning.ipynb             # Dedup + normalization
│   ├── 03_labeling_enhanced.ipynb     # Allergen annotation
│   ├── 04_model_training.ipynb        # MobileBERT fine-tuning
│   ├── 05_hybrid.ipynb               # Hybrid detection + evaluation
│   ├── utils/
│   │   ├── __init__.py               # Package exports (version 2.1.0)
│   │   ├── README.md                 # Comprehensive utility module docs
│   │   ├── text_processing.py        # BIG8 rules, negation, exemptions
│   │   ├── data_utils.py            # Load, split, augment
│   │   ├── model_utils.py           # Model, predict, hybrid
│   │   ├── evaluation.py            # Metrics, error analysis
│   │   └── ocr_utils.py            # EasyOCR + OpenCV preprocessing
│   └── models/                       # (checkpoints)
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

---

*Last updated: June 25, 2026*
*Version: 2.1.0*
