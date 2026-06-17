# Food Label Transparency Notebooks

Comprehensive notebook pipeline for extracting, processing, and analyzing food ingredient data to detect allergens using a hybrid ML + rule-based approach.

## 📚 Notebook Pipeline

### **1. Data Extraction** (`01_extraction.ipynb`)
- Extracts ingredient text from food label images using OCR
- Parses structured ingredient lists
- Stores data in DuckDB/CSV format

### **2. Data Cleaning** (`02_cleaning.ipynb`)
- Removes duplicates and invalid entries
- Normalizes ingredient names and formats
- Handles missing values and data quality issues
- Prepares data for annotation

### **3. Enhanced Labeling** (`03_labeling_enhanced.ipynb`)
- Annotates ingredients with allergen classifications
- Maps ingredients to FDA-defined allergen categories
- Generates labeled training dataset
- Validates annotation consistency

### **4. Model Training** (`04_model_training.ipynb`)
- Trains MobileBERT model for multi-label allergen classification
- Implements stratified train/validation/test splits
- Handles class imbalance with weighted BCE loss
- Optimizes thresholds for allergen predictions

### **5. Hybrid Detection** (`05_hybrid.ipynb`)
- Combines rule-based patterns with ML predictions
- Loads pre-trained MobileBERT model
- Evaluates hybrid predictions on test set
- Generates allergen confidence scores
- Produces production-ready configuration

### **6. OCR + Hybrid Pipeline** (`06_ocr_hybrid_pipeline.ipynb`)
- End-to-end simulation: Image → OCR → Predictions
- Integrates OCR extraction with hybrid model
- Generates final allergen reports
- Provides error analysis and performance metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers pandas numpy scikit-learn pillow pytesseract duckdb
```

### Running the Pipeline

1. **Start with extraction** (if you have food label images):
   ```bash
   jupyter notebook 01_extraction.ipynb
   ```

2. **Clean the extracted data**:
   ```bash
   jupyter notebook 02_cleaning.ipynb
   ```

3. **Create labels for training**:
   ```bash
   jupyter notebook 03_labeling_enhanced.ipynb
   ```

4. **Train the model** (optional, if fine-tuning):
   ```bash
   jupyter notebook 04_model_training.ipynb
   ```

5. **Test hybrid predictions**:
   ```bash
   jupyter notebook 05_hybrid.ipynb
   ```

6. **Run end-to-end pipeline**:
   ```bash
   jupyter notebook 06_ocr_hybrid_pipeline.ipynb
   ```

## 📊 Data Flow

```
Food Label Images
       ↓
   [01 Extraction] → Raw Text + Ingredients
       ↓
   [02 Cleaning] → Cleaned Ingredients
       ↓
   [03 Labeling] → Labeled Dataset
       ↓
   [04 Training] → Trained Model
       ↓
   [05 Hybrid] → ML + Rules
       ↓
   [06 Pipeline] → Allergen Predictions
```

## 🔧 Configuration

### Key Paths
- `../models/mobilebert_allergen/`: Trained model directory
- `../data/raw/`: Raw extracted data
- `../data/final/`: Cleaned and labeled data

### Allergen Categories
- Milk
- Eggs
- Peanuts
- Tree Nuts
- Soy
- Wheat
- Fish
- Shellfish

## 📈 Model Performance

- **Training framework**: Hugging Face Transformers
- **Model**: MobileBERT (optimized for mobile deployment)
- **Loss function**: Weighted Binary Cross-Entropy
- **Optimization**: Adam with warmup and learning rate scheduling

## 🛠️ Troubleshooting

### GPU Memory Issues
- Reduce batch size in training notebooks
- Use gradient accumulation for effective larger batches
- Consider using CPU for testing/inference

### OCR Accuracy
- Ensure good quality images (300+ DPI)
- Pre-process images (deskew, contrast enhancement)
- Use pytesseract with proper configuration

### Model Threshold Tuning
- Adjust thresholds in `05_hybrid.ipynb` for different recall/precision trade-offs
- Use validation set to optimize thresholds

## 📝 Notes

- All notebooks assume consistent data directory structure
- Hybrid model combines rule-based patterns for high-confidence allergens
- ML model provides additional detection for implicit allergens
- Production configuration saved in `model_thresholds.json`

## 🔍 Key Outputs

- **Trained model**: Saved in `../models/` directory
- **Predictions**: CSV with ingredient-allergen mappings
- **Thresholds**: JSON configuration for production deployment
- **Evaluation metrics**: Precision, recall, F1 scores per allergen

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Purpose**: Food Label Transparency Project for Filipino Consumers
