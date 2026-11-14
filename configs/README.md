# Configuration

This directory contains the version-controlled configuration used by the
notebook pipeline.

| File | Purpose |
| --- | --- |
| [`allergen_map.json`](allergen_map.json) | Maps Open Food Facts allergen tags to the project's eight allergen classes. |
| [`semantic_categories.json`](semantic_categories.json) | Defines the 40 semantic ingredient categories and their examples. |
| [`model_thresholds.json`](model_thresholds.json) | Historical thresholds from early experiments; retained for reference and not used for production inference. |

## Production thresholds

Notebook 09 writes the selected allergen thresholds and hybrid settings to
`models/hybrid_config.json`. This file is a generated model artifact and is
excluded from Git along with model checkpoints. Regenerate it by running the
evaluation notebook; do not use the historical `model_thresholds.json` values
as a substitute.

## Loading the allergen map

```python
import json

with open("../configs/allergen_map.json", encoding="utf-8") as file:
    config = json.load(file)

allergen_mapping = config["mapping"]
target_allergens = config["allergens"]
```
