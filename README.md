# 🏷️ Food Label Transparency for Filipino Consumers

**AI-based ingredient interpretation to detect allergens in food labels**  
A comprehensive notebook pipeline that extracts, processes, and analyzes food ingredient data using OCR, rule-based methods, and a MobileBERT model for multi-label allergen classification.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-ffb6c1)](https://huggingface.co/)

---

## 📚 Table of Contents
- [Overview](#overview)
- [Notebook Pipeline](#notebook-pipeline)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Key Outputs](#key-outputs)
- [License](#license)

---

## 🔬 Overview
This project aims to enhance food label transparency for Filipino consumers by automatically detecting allergens from ingredient lists printed on food packaging. The pipeline combines:
- **Optical Character Recognition (OCR)** to extract text from label images
- **Rule-based parsing** to structure ingredient lists
- **MobileBERT fine-tuning** for multi-label allergen classification
- **Hybrid detection** merging ML predictions with rule-based patterns for high-confidence results

The system targets the eight major allergens recognized by the FDA: Milk, Eggs, Peanuts, Tree Nuts, Soy, Wheat, Fish, and Shellfish.

Data used in this project is sourced from OpenFoodFacts.

---

## 📓 Notebook Pipeline
Each notebook represents a stage in the end-to-end workflow. Run them sequentially for best results.
`00_pipeline_overview.ipynb` is a markdown-only map of the whole pipeline, its two deliverables,
and the methodology guarantees (no augmentation leakage, validation-only model selection,
single held-out test evaluation, honest hybrid reporting).

### 1. Data Extraction (`01_extraction.ipynb`)
- Extracts ingredient text from food label images using **pytesseract** (OCR)
- Parses raw OCR output into structured ingredient entries
- Saves intermediates to **DuckDB** and **CSV** for downstream use

### 2. Data Cleaning (`02_cleaning.ipynb`)
- Deduplicates records and removes invalid/empty entries
- Normalizes ingredient names (lowercase, stripping punctuation, handling synonyms)
- Addresses missing values and inconsistencies
- Outputs a cleaned dataset ready for annotation

### Deliverable 1 — Semantic Classification (Notebooks 03–05)

### 3. Ingredient Parsing (`03_ingredient_parsing.ipynb`)
- Loads cleaned ingredient text from Notebook 02
- Runs `parse_ingredient_list()` to tokenize each product's ingredient list into individual ingredients
- Normalizes ingredient names and removes duplicates per product
- Builds an ingredient-level dataset for semantic classification
- Exports `parsed_ingredients.csv` for downstream labeling

### 4. Semantic Labeling (`04_semantic_labeling.ipynb`)
- Applies `ingredient_to_categories()` rule-based classifier from semantic_utils
- Maps 300+ known ingredients to 41 semantic categories (additives, flavors, macronutrients, etc.)
- Identifies unknown ingredients for manual annotation
- Builds a multi-label training matrix (binary vectors per ingredient × category)
- Exports `semantic_training_data.csv` for model training

### 5. Semantic Model Training (`05_semantic_model.ipynb`)
- Fine-tunes **MobileBERT** (Hugging Face Transformers) for multi-label semantic classification
- 41 output categories across 7 groups (additives, flavors, functional, macronutrients, etc.)
- Implements stratified train/validation/test splits
- Uses **Weighted Binary Cross-Entropy** loss to mitigate class imbalance
- Optimizes per-class probability thresholds, saves model and evaluation metrics

### Deliverable 2 — Allergen Detection (Notebooks 06–10)

### 6. Allergen Labeling (`06_allergen_labeling.ipynb`)
- Assigns allergen labels to each product using FDA-defined BIG8 categories
- Combines rule-based detection with official OpenFoodFacts allergen tags
- Processes "may contain" statements, traces tags, and exemption rules
- Includes Filipino variant detection merged into the BIG8 dictionary
- **Validates the rule engine against official OFF tags** (`evaluate_rule_vs_official`)
- Exports `labeled_dataset_enhanced.csv` for model training

### 7. Training Data Preparation (`07_training_data_preparation.ipynb`)
- **Splits first** (stratified 70/15/15) — augmentation never leaks into validation/test
- Augments the **training split only** (synonym replacement, targeted oversampling, counter-examples)
- Applies optional label-quality cleaning (`clean_training_data`)
- Saves canonical split files to `data/final/splits/`

### 8. Allergen Model Training (`08_allergen_training.ipynb`)
- Trains candidate MobileBERT classifiers: weighted **BCE**, **Focal loss**, **ASL**, and a 5-seed **ensemble**
- **Selects the winner on the validation set** — the test set is never touched here
- Saves the winner as the canonical `mobilebert_allergen_final` model

### 9. Allergen Evaluation (`09_allergen_evaluation.ipynb`)
- Tunes per-class probability thresholds **on validation**
- Runs the **single, final held-out test evaluation**
- Compares **ML predictions vs official OFF tags** for a real-world signal
- Quantifies seed variance across the ensemble

### Deliverable 3 — Hybrid Detection (Notebook 10)

### 10. Hybrid Evaluation (`10_hybrid_evaluation.ipynb`)
- Compares rule-based, ML, and hybrid systems three ways (rule vs ML vs hybrid)
- **Reports honestly how many predictions the hybrid override actually changes** (helped / hurt / neutral)
- Includes per-allergen hybrid modes (rule-priority for rare classes) and a tree-nuts case study
- Saves the production `hybrid_config.json`

### Deliverable 4 — Deployment (Notebooks 11–12)

### 11. Model Export (`11_model_export.ipynb`)
- Converts trained PyTorch models to ONNX format
- Validates ONNX output against the PyTorch baseline
- Benchmarks PyTorch vs ONNX inference latency
- Generates a deployment readiness report

### 12. Mobile Benchmark (`12_mobile_benchmark.ipynb`)
- Benchmarks ONNX models against mobile acceptance criteria (latency, size, accuracy)
- Validates deployment viability for ONNX Runtime Mobile

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.9
- Git
- Tesseract OCR installed (see [tesseract setup](https://github.com/tesseract-ocr/tesseract))
- (Optional) GPU for faster training

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/food-label-transparency.git
cd food-label-transparency

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
> **Tip:** If you don’t have a `requirements.txt`, run:
> ```bash
> pip install torch transformers pandas numpy scikit-learn pillow pytesseract duckdb onnx onnxruntime tensorflow scipy matplotlib tqdm jupyter
> ```

### Running the Notebooks
Launch Jupyter Lab/Notebook and execute the notebooks in order:
```bash
jupyter lab   # or jupyter notebook
```
Then open:
1. `00_pipeline_overview.ipynb`
2. `01_extraction.ipynb`
3. `02_cleaning.ipynb`
4. `03_ingredient_parsing.ipynb` *(Deliverable 1 — Semantic Classification)*
5. `04_semantic_labeling.ipynb`
6. `05_semantic_model.ipynb`
7. `06_allergen_labeling.ipynb` *(Deliverable 2 — Allergen Detection)*
8. `07_training_data_preparation.ipynb`
9. `08_allergen_training.ipynb`
10. `09_allergen_evaluation.ipynb`
11. `10_hybrid_evaluation.ipynb` *(Deliverable 3 — Hybrid Detection)*
12. `11_model_export.ipynb` *(Deliverable 4 — Deployment)*
13. `12_mobile_benchmark.ipynb`

Each notebook contains executable cells with clear markdown explanations.

---

## 📊 Data Flow
```mermaid
flowchart TD
    A[Food Label Images] --> B[01 Extraction: OCR + Parsing]
    B --> C[02 Cleaning: Deduplication + Normalization]
    C --> D[03 Ingredient Parsing: Tokenize → Ingredient-Level Dataset]
    D --> E[04 Semantic Labeling: Rule-Based Category Mapping]
    E --> F[05 Semantic Model: MobileBERT Multi-Label Training]
    F --> G[06 Allergen Labeling: Rule-Based + Official-Tag Validation]
    G --> H[07 Data Prep: Stratified Split → Augment Train Only]
    H --> I[08 Allergen Training: BCE / Focal / ASL / Ensemble]
    I --> J[09 Evaluation: Val Thresholds + Held-Out Test]
    J --> K[10 Hybrid Evaluation: ML + Rule-Based]
    K --> L[11 Model Export: PyTorch → ONNX]
    L --> M[12 Mobile Benchmark: ONNX Runtime Mobile]
    M --> N[Allergen Predictions per Product]
```

---

## 🔧 Configuration

### Key Directories
| Path | Description |
|------|-------------|
| `notebooks/` | Jupyter notebooks for each pipeline stage (run from this directory) |
| `notebooks/utils/` | Python utility modules (text_processing, semantic_utils, model_utils, etc.) |
| `data/raw/` | Raw data source — OpenFoodFacts parquet dump |
| `data/processed/` | Cleaned, deduplicated product-level dataset |
| `data/interim/` | Intermediate pipeline outputs (extracted, parsed, vocabulary) |
| `data/final/` | Labeled datasets for training (semantic + allergen) |
| `models/mobilebert_allergen_final/` | Fine‑tuned MobileBERT allergen classifier weights & tokenizer |
| `models/mobilebert_semantic_final/` | Fine‑tuned MobileBERT semantic classifier weights & tokenizer |
| `models/exported/` | ONNX exported deployment models |
| `models/hybrid_config.json` | Hybrid ML + rule thresholds and parameters |
| `configs/` | JSON configuration (allergen map, semantic categories, model thresholds) |
| `scripts/` | Pipeline orchestration and validation scripts |

### Allergen Classes
1. Milk  
2. Eggs  
3. Peanuts  
4. Tree Nuts  
5. Soy  
6. Wheat  
7. Fish  
8. Shellfish  

*(Modify `configs/allergen_map.json` if you need to add/remove classes.)*

### Threshold Usage
The model uses probability thresholds for each allergen class to convert model outputs to binary predictions. These thresholds are optimized during validation to maximize F1-score for each class:

- **ML thresholds**: Found in `../models/hybrid_config.json` (optimized for hybrid system)
- **Alternative thresholds**: Found in `../configs/model_thresholds.json` (from earlier experiments)

To use custom thresholds in inference, load them and pass to the prediction function:
```python
import json
import numpy as np

# Load thresholds
with open('../models/hybrid_config.json', 'r') as f:
    config = json.load(f)
thresholds = np.array(config["ml_thresholds"])

# Use in prediction
preds, probs = predict_ml(texts, thresholds=thresholds)
```

---

## 📈 Model Performance

### Baseline Model (Notebook 08)
| Metric | Milk | Eggs | Peanuts | Tree Nuts | Soy | Wheat | Fish | Shellfish |
|--------|------|------|---------|-----------|-----|-------|------|-----------|
| Precision | 0.97 | 1.00 | 0.80 | 0.91 | 0.93 | 1.00 | 1.00 | 1.00 |
| Recall | 0.92 | 0.90 | 1.00 | 0.96 | 1.00 | 0.98 | 1.00 | 0.60 |
| F1‑Score | 0.95 | 0.95 | 0.89 | 0.93 | 0.96 | 0.99 | 1.00 | 0.75 |

### Tree Nuts Case Study (Notebook 10)
| Metric | Milk | Eggs | Peanuts | Tree Nuts | Soy | Wheat | Fish | Shellfish |
|--------|------|------|---------|-----------|-----|-------|------|-----------|
| Precision | 1.00 | 1.00 | 0.99 | 0.93 | 0.99 | 1.00 | 0.98 | 1.00 |
| Recall | 0.98 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.98 | 1.00 |
| F1‑Score | 0.99 | 1.00 | 0.99 | 0.96 | 0.98 | 1.00 | 0.98 | 1.00 |

- **Framework:** Hugging Face `transformers` + `pytorch`
- **Base Model:** `google/mobilebert-uncased`
- **Loss:** Weighted Binary Cross‑Entropy (inverse class frequency) for baseline, Asymmetric Loss for optimized
- **Optimizer:** AdamW with linear warm‑up and cosine decay
- **Training Time:** ~25 min on a single RTX 3060 (6 GB VRAM) for baseline
- **Inference Latency:** ~12 ms per ingredient on CPU

*Numbers shown are from the pre-restructure runs. Re-run `08_allergen_training.ipynb` → `09_allergen_evaluation.ipynb` and the tree-nuts case study in `10_hybrid_evaluation.ipynb` to regenerate them on the current pipeline.*

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **GPU OOM** | Reduce `batch_size` in `08_allergen_training.ipynb`; enable `gradient_accumulation_steps`. |
| **Slow OCR** | Ensure input images are ≥300 DPI; apply deskewing and contrast adjustment (see OpenCV snippets in the extraction notebook). |
| **Low Recall for Rare Allergens** | Increase class weight for that allergen in the loss function; collect more labeled examples. |
| **Module Not Found (e.g., `duckdb`)** | Install via `pip install duckdb`; verify you’re using the correct virtual environment. |
| **Threshold Tuning** | Adjust thresholds in `09_allergen_evaluation.ipynb` → `find_best_thresholds`; optimize using validation precision‑recall curves. |

---

## 📦 Key Outputs

- **`models/mobilebert_allergen_final/`** – Trained MobileBERT checkpoint & tokenizer
- **`models/mobilebert_semantic_final/`** – Trained MobileBERT semantic classifier & tokenizer
- **`models/exported/allergen_model/allergen_model.onnx`** – ONNX-exported model for mobile deployment
- **`configs/model_thresholds.json`** – Optimal probability thresholds per allergen
- **`configs/allergen_map.json`** – Allergen keyword mappings with Filipino variants

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Last updated: June 26, 2026*  
*Version: 2.3.0*