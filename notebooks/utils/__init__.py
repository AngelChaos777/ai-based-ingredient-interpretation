"""
Food Label Transparency Project - Utility Modules

This package provides modular utilities for food label allergen detection, processing, and analysis.

Available Modules:
- text_processing: Allergen detection, text cleaning, token parsing, and rule-based matching
- data_utils: Data loading, splitting, augmentation, and file I/O utilities
- model_utils: Model loading, tokenization, prediction, and hybrid inference utilities
- evaluation: Evaluation metrics, reporting, and error analysis utilities
- ocr_utils: OCR and computer vision utilities for image-based label processing

Usage:
    from utils import load_labeled_data, detect_allergens_rule_based, get_allergen_list
    from utils.model_utils import load_model_and_tokenizer, hybrid_predict
    from utils.evaluation import print_classification_report, error_analysis
"""

from .text_processing import (
    clean_html,
    preprocess_text,
    rule_match,
    detect_allergens_rule_based,
    clean_ingredient_text,
    tokenize_ingredients,
    parse_label_string,
    get_allergen_list,
    allergens_to_binary,
    extract_may_contain,
    has_explicit_allergen_statement,
    parse_traces_tags,
    parse_official_tags,
    apply_exemptions,
    BIG8,
    ALLERGEN_RULES,
    OFFICIAL_MAP,
    COMPILED_RULES,
    NEGATION_PATTERN,
)

from .data_utils import (
    load_extracted_data,
    load_cleaned_data,
    load_labeled_data,
    parse_label_column,
    create_stratified_splits,
    augment_dataframe,
    labels_to_binary,
    safe_parse_list,
    save_metadata,
    load_metadata,
    get_data_directories,
)

from .model_utils import (
    load_model_and_tokenizer,
    load_hybrid_config,
    compute_class_weights,
    find_best_thresholds,
    predict_ml,
    rule_match,
    hybrid_predict,
    apply_thresholds,
)

from .evaluation import (
    print_classification_report,
    print_per_class_metrics,
    find_best_thresholds,
    apply_thresholds,
    compute_per_class_metrics,
    compute_multilabel_confusion_matrix,
    plot_confusion_matrices,
    error_analysis,
    binary_to_label_list,
)

# Version information
__version__ = "2.1.0"
__author__ = "Food Label Transparency Team"
__license__ = "MIT"