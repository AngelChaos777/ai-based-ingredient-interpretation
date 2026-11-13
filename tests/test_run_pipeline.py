import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pipeline


def test_missing_expected_outputs_fail_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "NOTEBOOK_ORDER", ["02_cleaning"])
    monkeypatch.setattr(run_pipeline, "EXPECTED_OUTPUTS", {"02_cleaning": ["missing.csv"]})
    monkeypatch.setattr(run_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path / "executed")
    monkeypatch.setattr(run_pipeline, "run_notebook", lambda *args, **kwargs: (True, 0.0))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])

    assert run_pipeline.main() == 1


def test_expected_output_paths_match_notebook_exports():
    assert run_pipeline.EXPECTED_OUTPUTS["02_cleaning"] == [
        "data/processed/cleaned_dataset.csv"
    ]
    assert run_pipeline.EXPECTED_OUTPUTS["04_semantic_labeling"] == [
        "data/final/semantic_training_data.csv"
    ]
    assert run_pipeline.EXPECTED_OUTPUTS["06_allergen_labeling"] == [
        "data/final/labeled_dataset_enhanced.csv"
    ]
