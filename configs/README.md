# Configuration Files

This directory contains configuration files for the food label transparency project.

## Files

### `allergen_map.json`
Mapping from OpenFoodFacts allergen tags to internal allergen names used in the model.
- Used in the labeling notebook (03_labeling_enhanced.ipynb) to convert official tags to our 8 allergen classes
- Contains both the mapping dictionary and the list of target allergens

### `model_thresholds.json`
Optimal probability thresholds per allergen class from initial experiments.
- Format: {"allergen_name": threshold_value, ...}
- These thresholds were obtained from early validation experiments
- Note: For production/hybrid system, use the thresholds in `../models/hybrid_config.json` instead

## Usage

To use these configurations in your code:

```python
import json
from pathlib import Path

# Load allergen map
with open("../configs/allergen_map.json") as f:
    allergen_config = json.load(f)
allergen_mapping = allergen_config["mapping"]
target_allergens = allergen_config["allergens"]

# Load thresholds (for reference only - see hybrid_config.json for production thresholds)
with open("../configs/model_thresholds.json") as f:
    model_thresholds = json.load(f)

# For production use, load the hybrid config instead:
with open("../models/hybrid_config.json") as f:
    hybrid_config = json.load(f)
ml_thresholds = hybrid_config["ml_thresholds"]  # Array in same order as target_allergens
rule_conf_threshold = hybrid_config["rule_conf_threshold"]
mode = hybrid_config["mode"]
```