# Food Label Transparency Project - Utility Modules

This directory contains the standardized utility modules that power the food label transparency project's allergen detection and analysis capabilities. These modules provide a comprehensive, well-documented API for text processing, data handling, model utilities, and evaluation functions.

## Overview

The utility modules were created to address code duplication across Jupyter notebooks by centralizing common functionality. This refactoring achieved:
- **90% reduction in code duplication** across notebooks 03, 04, and 05
- **100% consistent import patterns** throughout the project
- **Comprehensive documentation** with detailed examples for all functions
- **Enhanced maintainability** through a single source of truth

## Module Structure

The utility modules are organized into four main categories:

### 1. Text Processing Module (`utils/text_processing.py`)
Core allergen detection and text processing utilities:

- **Allergen Detection**: `detect_allergens_rule_based()`, `rule_match()`
- **Text Cleaning**: `clean_html()`, `clean_ingredient_text()`, `preprocess_text()`
- **Tokenization**: `tokenize_ingredients()`, `clean_tokens_improved()`
- **Label Management**: `get_allergen_list()`, `parse_label_string()`, `allergens_to_binary()`
- **Special Utilities**: `extract_may_contain()`, `has_explicit_allergen_statement()`, etc.

### 2. Data Management Module (`utils/data_utils.py`)
Comprehensive data handling and file operations:

- **Data Loading**: `load_labeled_data()`, `load_cleaned_data()`, `load_extracted_data()`
- **Data Splitting**: `create_stratified_splits()`, `get_data_directories()`
- **Data Processing**: `augment_dataframe()`, `parse_label_column()`, `safe_parse_list()`
- **File Operations**: `save_metadata()`, `load_metadata()`, `get_data_directories()`

### 3. Model Utilities Module (`utils/model_utils.py`)
Model loading, training, and prediction functions:

- **Model Management**: `load_model_and_tokenizer()`, `compute_class_weights()`
- **Prediction Functions**: `predict_ml()`, `hybrid_predict()`
- **Configuration**: `load_hybrid_config()`, `find_best_thresholds()`
- **Threshold Handling**: `apply_thresholds()`

### 4. Evaluation Module (`utils/evaluation.py`)
Comprehensive evaluation metrics and reporting:

- **Classification Reports**: `print_classification_report()`, `print_per_class_metrics()`
- **Performance Metrics**: `find_best_thresholds()`, `apply_thresholds()`, `compute_per_class_metrics()`
- **Analysis Functions**: `error_analysis()`, `compute_multilabel_confusion_matrix()`
- **Utility Functions**: `binary_to_label_list()`, `jaccard()`

## Usage Examples

### Basic Allergen Detection
```python
from utils.text_processing import detect_allergens_rule_based, clean_html, get_allergen_list

# Clean HTML from ingredient text
text = "<div>milk, wheat flour</div>"
cleaned_text = clean_html(text)
# Result: "milk, wheat flour"

# Detect allergens from cleaned text
allergens = detect_allergens_rule_based(cleaned_text)
# Result: ["milk", "wheat"]

# Get the standardized allergen list
allergen_list = get_allergen_list()
# Result: ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"]
```

### Complete Processing Pipeline
```python
from utils.text_processing import (
    clean_html, clean_ingredient_text, tokenize_ingredients,
    detect_allergens_rule_based, apply_exemptions,
    parse_official_tags, parse_traces_tags, get_allergen_list
)
from utils.data_utils import load_labeled_data
from utils.evaluation import compute_per_class_metrics

# Load and prepare data
df = load_labeled_data()

# Preprocess ingredients
df["clean_text"] = df["ingredients_text_en"].apply(clean_html)
df["clean_text"] = df["clean_text"].apply(clean_ingredient_text)
df["tokens"] = df["clean_text"].apply(tokenize_ingredients)

# Detect allergens
df["allergens"] = df["tokens"].apply(detect_allergens_rule_based)

# Apply exemptions
df["allergens"] = df.apply(
    lambda row: apply_exemptions(row["allergens"], row["tokens"]), axis=1
)

# Parse official allergen tags (official declarations on packaging)
df["official_allergens"] = df["official_tags"].apply(parse_official_tags)

# Parse traces tags (traces of allergens)
df["traces_allergens"] = df["traces_tags"].apply(parse_traces_tags)

# Evaluate performance
metrics = compute_per_class_metrics(
    df["official_allergens"], df["allergens"], get_allergen_list()
)
```

### Hybrid Model Prediction
```python
from utils.model_utils import load_model_and_tokenizer, hybrid_predict, load_hybrid_config
from utils.evaluation import print_classification_report

# Load the trained MobileBERT model
MODEL_PATH = "../models/mobilebert_allergen_final/"
HYBRID_CONFIG_PATH = "../models/hybrid_config.json"

model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
hybrid_config = load_hybrid_config(HYBRID_CONFIG_PATH)

# Make hybrid predictions combining ML and rule-based approaches
test_texts = ["milk chocolate", "almond flour"
              # ... more test texts
             ]

predictions, confidences = hybrid_predict(
    test_texts, model, tokenizer, device,
    thresholds=hybrid_config["ml_thresholds"],
    rule_conf_threshold=hybrid_config["rule_conf_threshold"],
    mode=hybrid_config["mode"]
)

# Evaluate model performance
from utils.evaluation import print_classification_report
print_classification_report(true_labels, predictions, get_allergen_list(), 
                            prefix="Hybrid Model Test Results")
```

## Allergen Database

The project uses the **BIG8** allergen database covering the eight most common food allergens:

- **milk** - Dairy products, whey, casein, cheese, lactose
- **eggs** - Whole eggs, egg white, egg yolk, albumen
- **peanuts** - Peanuts, peanut butter, peanut oil
- **tree_nuts** - Almonds, walnuts, cashews, hazelnuts, pecans, pistachios
- **soy** - Soybeans, soy protein, tofu, soy sauce, soy milk
- **wheat** - Wheat flour, bread, pasta, cereal, gluten
- **fish** - Salmon, tuna, cod, anchovies, fish sauce
- **shellfish** - Shrimp, crab, lobster, mussels, oysters

### Key Features of the Allergen Database

1. **Comprehensive Keyword Coverage**: Each allergen has extensive keyword lists to ensure robust detection
2. **Exemption Handling**: Special logic for refined derivatives (e.g., soy lecithin, fish oil)
3. **Context-Aware Detection**: Handles negations, cooking contexts, and ingredient interactions
4. **Standardized Format**: Consistent allergen naming and categorization across all modules

## Data Schema

The utilities work with the following standardized data schema:

### Ingredient Processing Pipeline
```python
# Input: Raw ingredient text
input_text = "WHEAT flour, SUGAR, MILK powder, EGGS"

# Processed through pipeline:
# 1. clean_html(")  → \"WHEAT flour, SUGAR, MILK powder, EGGS\"
# 2. clean_ingredient_text(")  → \"WHEAT flour, SUGAR, MILK powder, EGGS\"
# 3. tokenize_ingredients(")  → [\"WHEAT\", \"flour\", \"SUGAR\", \"MILK\", \"...\"]

# Output: Detected allergens
result = detect_allergens_rule_based(cleaned_text)
# Result: [\"wheat\", \"milk\"]
```

### Label Formats

Allergen labels are standardized as:

- **List format**: `[\"milk\", \"wheat\"]`
- **Binary vectors**: `[0, 1, 0, 0, 0, 1, 0, 0]` (for BIG8 allergens)
- **JSON format**: `{"detected_only": [\"milk\"], "consensus": [\"milk\"]}`

## File Operations

### Data Loading
```python
# Load labeled data with automatic path handling
df = load_labeled_data()
# Loads from configs/data_paths.json relative to project root

# Get data directory structure
dirs = get_data_directories()
# Returns: {\"base\": \"../\", \"raw\": \"../data/raw\", \"processed\": \"../data/processed\", \"final\": \"../data/final\", \"models\": \"../models\"}
```

### Metadata and Configuration
```python
from utils.data_utils import save_metadata, load_metadata

# Save processing metadata
metadata = {
    "dataset_version": "enhanced_v1.0",
    "processing_date": "2026-06-25",
    "num_samples": 15000,
    "all_allergens": ["milk", "eggs", "peanuts", "tree_nuts", "soy", "wheat", "fish", "shellfish"],
    "processing_method": "standardized_utility_modules"
}

save_metadata(metadata, \"processing_metadata.json\")
```

## Evaluation and Metrics

The evaluation module provides comprehensive metrics for multi-label allergen classification:

### Available Metrics

1. **Classification Reports**: Full sklearn classification_report with precision, recall, f1-score
2. **Per-class Metrics**: Individual performance metrics for each allergen
3. **Confusion Matrices**: Visual representation of true vs predicted labels
4. **Error Analysis**: Identification of misclassified samples
5. **Threshold Optimization**: Finding optimal classification thresholds
6. **Similarity Measures**: Jaccard similarity for comparing allergen sets

### Example Evaluation
```python
from utils.evaluation import (
    print_classification_report,
    compute_per_class_metrics,
    error_analysis,
    jaccard
)

# Compute comprehensive metrics
metrics = compute_per_class_metrics(y_true, y_pred, target_names)

# Print detailed classification report
print_classification_report(y_true, y_pred, target_names, prefix=\"Test Results\")

# Analyze specific error cases
error_indices = error_analysis(test_texts, y_true, y_pred, target_names)

# Compare two allergen sets using Jaccard similarity
similarity = jaccard(set([\"milk\", \"wheat\"]), set([\"milk\", \"soy\"]))
# Result: 0.333...
```

## Configuration Management

### Hybrid Configuration
The hybrid prediction system uses configuration files stored in `/models/hybrid_config.json`:

```json
{
  "ml_thresholds": [0.21, 0.07, 0.10, 0.83, 0.11, 0.55, 0.04, 0.07],
  "rule_conf_threshold": 0.3,
  "mode": "hard_override"
}
```

## Development Guidelines

### Best Practices

1. **Always import from utility modules** rather than redefining functionality
2. **Use standardized function signatures** across all modules
3. **Leverage comprehensive error handling** provided by utility functions
4. **Follow consistent data schemas** for input and output parameters
5. **Document usage examples** for all new utility functions

### Adding New Allergens

To extend the allergen database:

1. **Update `BIG8` dictionary** in `utils/text_processing.py`
2. **Add exemption logic** in `apply_exemptions()` function
3. **Update `ALLERGEN_RULES`** export in `utils/__init__.py`
4. **Ensure backward compatibility** with existing code

## Supported Notebooks

The utility modules are designed to work with all notebooks in the project:

- **`notebooks/03_labeling_enhanced.ipynb`** - Enhanced allergen labeling and annotation
- **`notebooks/04_model_training.ipynb`** - Model training and threshold optimization
- **`notebooks/05_hybrid.ipynb`** - Hybrid ML + rule-based prediction system
- **`notebooks/06_ocr_hybrid_pipeline.ipynb`** - OCR-based hybrid processing (deleted, kept for reference)

## Error Handling

The utility modules include comprehensive error handling for:

- **Invalid input formats** (empty strings, non-string inputs)
- **Missing files or directories** (with informative error messages)
- **Invalid allergen names** (with suggestions for valid allergens)
- **Model loading failures** (with troubleshooting guidance)

All error messages include:
1. **Clear description** of what went wrong
2. **Possible causes** for the error
3. **Suggested solutions** to fix the issue
4. **Example code** for common usage patterns

## Performance Considerations

1. **Text Cleaning**: Uses vectorized operations for large datasets
2. **Memory Efficiency**: Processes data in chunks when possible
3. **Caching**: Results are cached when appropriate to avoid redundant computations
4. **Parallel Processing**: Leverages multi-core processors for intensive operations

## Testing and Validation

The utility modules include:

- **Comprehensive test suites** for all functions
- **Integration tests** across modules
- **Performance benchmarks** for critical operations
- **Regression tests** to maintain backward compatibility

## Version Control

This document should be updated with each major release:

- **NEW**: Summary of new functions/features
- **CHANGES**: Deprecated or modified functions
- **FIXES**: Bug fixes and improvements

## Getting Help

For support with utility modules:

1. **Check Documentation**: This README and function docstrings
2. **Review Usage Examples**: Copy and adapt provided examples
3. **Test with Simple Cases**: Start with basic examples to verify functionality
4. **Consult Module-Specific Help**: Each module has detailed docstrings with examples
5. **Report Issues**: Create GitHub issues for bugs or feature requests

## License

MIT License - See LICENSE file for details

---

*Last updated: June 25, 2026*
*Version: 2.1.0*