# Project Memory File – Enhancing Food Label Transparency for Filipino Consumers

## Goal
Create an AI‑based ingredient interpretation system that makes food label information clear, accurate, and accessible for Filipino consumers.

## Success Criteria
- Model achieves ≥90 % F1 on a held‑out validation set of ingredient extraction tasks.
- End‑user UI returns correct ingredient translations for ≥95 % of test queries.
- Documentation and reproducibility allow a new developer to set up the pipeline in <2 hours.

## Current Phase
Exploration & Dataset Preparation

## Done
- Literature review of allergen‑detection and ingredient‑translation models.
- Collected initial raw label dataset (≈5k Filipino food product images).\n## In Progress
- OCR pipeline for extracting text from label images.
- Initial data cleaning and annotation schema design.

## Next Up
1. Finalise OCR preprocessing (image resizing, binarisation).
2. Annotate a pilot set of 500 label texts with ground‑truth ingredient lists.
3. Define evaluation metrics and baseline model architecture.

## Architecture Overview
| File/Component | Purpose | Key Constraints |
|---|---|---|
| `data/ocr/` | OCR scripts & models | Must handle mixed‑language (Tagalog/English) text |
| `data/annotations/` | CSV/JSON of ingredient lists | Consistent schema, Unicode support |
| `models/` | Training scripts, config files | Reproducible seeds, GPU‑agnostic |
| `src/` | Core pipeline (parsing, translation) | Modular, testable |
| `notebooks/` | Exploration & visualisation | Clear narrative for reviewers |

## Decisions Log
| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026‑07‑06 | Use EasyOCR for initial text extraction | Works well on mixed fonts, open‑source | Active |
| 2026‑07‑06 | Store annotations as UTF‑8 JSON lines | Simpler streaming for large sets | Active |

## Open Questions / Blockers
- Which language model architecture will best handle code‑mixed Filipino text?
- How to evaluate allergen detection accuracy across diverse product categories?

## Session Primer (for next session)
- Verify OCR pipeline on a sample of 20 images.
- Review pilot annotation guidelines with the team.
- Decide on baseline model (e.g., mBERT vs. XLM‑R).

*End of PMF*