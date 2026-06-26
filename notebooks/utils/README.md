# Utility Modules — Food Label Transparency

Centralized functionality powering allergen detection, data processing, model training, and evaluation across the notebook pipeline. Refactored to reduce code duplication by ~90% across notebooks 03–05.

## Module Structure

| Module | Path | Purpose |
| ------ | ---- | ------- |
| Text Processing | `utils/text_processing.py` | Allergen keyword matching, BIG8 database, context disambiguation, negation handling, Filipino variants |
| Data Management | `utils/data_utils.py` | CSV loading, multi-label stratified splitting, data augmentation, metadata persistence |
| Model Utilities | `utils/model_utils.py` | Model/tokenizer loading, class weights, ML + hybrid prediction, threshold optimization |
| Evaluation | `utils/evaluation.py` | Classification reports, per-class metrics, confusion matrices, error analysis, Jaccard similarity |
| Semantic Classification | `utils/semantic_utils.py` | Ingredient-to-category mapping (41 categories), ingredient parsing, binary matrix conversion |
| OCR | `utils/ocr_utils.py` | Image preprocessing + EasyOCR text extraction for label images |
| Deployment | `utils/deployment_utils.py` | PyTorch → ONNX export, output validation, latency benchmarking, deployment reports |

## Quick Start

```python
from utils.text_processing import detect_allergens_rule_based, clean_html
from utils.model_utils import load_model_and_tokenizer, hybrid_predict
from utils.evaluation import print_classification_report

# Rule-based detection
text = clean_html("<div>milk, wheat flour</div>")
allergens = detect_allergens_rule_based(text)  # ["milk", "wheat"]

# Hybrid ML + rule prediction
model, tokenizer, device = load_model_and_tokenizer("../models/mobilebert_allergen_final/")
preds, confs = hybrid_predict(texts, model, tokenizer, device, thresholds=[0.15, 0.61, ...])
```

## Allergen Classes (BIG8)

`milk`, `eggs`, `peanuts`, `tree_nuts`, `soy`, `wheat`, `fish`, `shellfish`

## Label Formats

- **List**: `["milk", "wheat"]`
- **Binary vector**: `[1, 0, 0, 0, 0, 1, 0, 0]` (BIG8 order)
- **JSON**: `{"detected_only": ["milk"], "consensus": ["milk"]}`

## Growing the Allergen Database

1. Add keywords to `BIG8[allergen]` in `text_processing.py`
2. Add exemption rules to `apply_exemptions()` if needed
3. Add context rules to `AMBIGUOUS_KEYWORDS` if ambiguous terms are introduced
4. Add Filipino variants to `FILIPINO_VARIANTS`

## Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_text_processing_tree_nuts.py -v
```

See `tests/` for test suites covering keyword matching, exemptions, negation handling, and Filipino variants.
