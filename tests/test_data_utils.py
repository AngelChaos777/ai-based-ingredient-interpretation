"""
Unit tests for data_utils.py — data loading, label normalization, augmentation.

Run with: python -m pytest tests/test_data_utils.py -v
"""
import sys
import numpy as np
import json
from pathlib import Path

_utils_dir = Path(__file__).resolve().parents[1] / "notebooks" / "utils"
sys.path.insert(0, str(_utils_dir))

import pytest


@pytest.fixture(autouse=True)
def import_module():
    global du
    try:
        import data_utils as du
    except ImportError as e:
        pytest.skip(f"data_utils not importable: {e}")


# ──────────────────────────────────────────────
# serialize_list_for_csv
# ──────────────────────────────────────────────

class TestSerializeListForCsv:
    def test_empty_list(self):
        assert du.serialize_list_for_csv([]) == "[]"

    def test_single_item(self):
        assert du.serialize_list_for_csv(["milk"]) == '["milk"]'

    def test_multiple_items(self):
        result = du.serialize_list_for_csv(["milk", "wheat", "eggs"])
        assert json.loads(result) == ["milk", "wheat", "eggs"]

    def test_uses_json_not_python_repr(self):
        """JSON uses double quotes, Python repr uses single quotes."""
        result = du.serialize_list_for_csv(["milk"])
        assert '"milk"' in result
        assert "'milk'" not in result


# ──────────────────────────────────────────────
# normalize_list_column
# ──────────────────────────────────────────────

class TestNormalizeListColumn:
    def test_already_a_list(self, monkeypatch):
        import pandas as pd
        series = pd.Series([["milk"], ["wheat"], []])
        result = du.normalize_list_column(series)
        assert result.iloc[0] == ["milk"]
        assert result.iloc[1] == ["wheat"]
        assert result.iloc[2] == []

    def test_json_string_format(self):
        import pandas as pd
        series = pd.Series(['["milk", "wheat"]', '[]'])
        result = du.normalize_list_column(series)
        assert result.iloc[0] == ["milk", "wheat"]
        assert result.iloc[1] == []

    def test_python_repr_format(self):
        import pandas as pd
        series = pd.Series(["['milk', 'wheat']", "[]"])
        result = du.normalize_list_column(series)
        assert result.iloc[0] == ["milk", "wheat"]
        assert result.iloc[1] == []

    def test_nan_values(self):
        import pandas as pd
        import numpy as np
        series = pd.Series([np.nan, None, ["milk"]])
        result = du.normalize_list_column(series)
        assert result.iloc[0] == []
        assert result.iloc[1] == []
        assert result.iloc[2] == ["milk"]

    def test_mixed_types(self):
        """Handle a series with mixed list and string entries."""
        import pandas as pd
        series = pd.Series([["milk"], '["wheat"]', "[]"])
        result = du.normalize_list_column(series)
        assert result.iloc[0] == ["milk"]
        assert result.iloc[1] == ["wheat"]
        assert result.iloc[2] == []


# ──────────────────────────────────────────────
# normalize_dict_column
# ──────────────────────────────────────────────

class TestNormalizeDictColumn:
    def test_already_a_dict(self):
        import pandas as pd
        original = {"key": "value"}
        series = pd.Series([original])
        result = du.normalize_dict_column(series)
        assert result.iloc[0] == original

    def test_json_string(self):
        import pandas as pd
        series = pd.Series(['{"key": "value"}'])
        result = du.normalize_dict_column(series)
        assert result.iloc[0] == {"key": "value"}

    def test_nan_values(self):
        import pandas as pd
        import numpy as np
        series = pd.Series([np.nan, None])
        result = du.normalize_dict_column(series)
        assert result.iloc[0] == {}
        assert result.iloc[1] == {}


# ──────────────────────────────────────────────
# validate_label_consistency
# ──────────────────────────────────────────────

class TestValidateLabelConsistency:
    def test_valid_labels_no_issues(self):
        import pandas as pd
        df = pd.DataFrame({
            "detected_allergens": [
                ["milk", "wheat"],
                ["eggs"],
                [],
            ]
        })
        issues = du.validate_label_consistency(df)
        # Since detected_allergens values are lists, no string-type issues
        assert len(issues) == 0

    def test_string_labels_noted(self):
        import pandas as pd
        df = pd.DataFrame({
            "detected_allergens": [
                "['milk', 'wheat']",
                "[]",
            ]
        })
        issues = du.validate_label_consistency(df)
        assert len(issues) > 0
        assert any("string repr" in issue for issue in issues)

    def test_nan_detected(self):
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "detected_allergens": [np.nan, ["milk"]]
        })
        issues = du.validate_label_consistency(df)
        assert len(issues) > 0
        assert any("NaN" in issue for issue in issues)


# ──────────────────────────────────────────────
# safe_parse_list
# ──────────────────────────────────────────────

class TestSafeParseList:
    def test_parses_list_string(self):
        assert du.safe_parse_list("['milk', 'wheat']") == ["milk", "wheat"]

    def test_parses_empty_list(self):
        assert du.safe_parse_list("[]") == []

    def test_handles_list_input(self):
        assert du.safe_parse_list(["milk", "wheat"]) == ["milk", "wheat"]

    def test_handles_invalid(self):
        assert du.safe_parse_list("not a list") == []


# ──────────────────────────────────────────────
# labels_to_binary
# ──────────────────────────────────────────────

class TestLabelsToBinary:
    def test_basic_conversion(self):
        classes = ["milk", "eggs", "peanuts"]
        label_list = [["milk"], ["eggs", "peanuts"], []]
        result = du.labels_to_binary(label_list, classes)
        assert result.shape == (3, 3)
        assert list(result[0]) == [1, 0, 0]
        assert list(result[1]) == [0, 1, 1]
        assert list(result[2]) == [0, 0, 0]

    def test_empty_input(self):
        result = du.labels_to_binary([], ["milk", "eggs"])
        assert result.shape == (0, 2)


# ──────────────────────────────────────────────
# get_data_directories
# ──────────────────────────────────────────────

class TestGetDataDirectories:
    def test_returns_all_keys(self):
        dirs = du.get_data_directories()
        expected_keys = {"base", "raw", "interim", "processed", "final",
                         "models", "configs", "outputs", "notebooks"}
        assert expected_keys.issubset(dirs.keys())

    def test_paths_are_absolute(self):
        dirs = du.get_data_directories()
        for key, path in dirs.items():
            assert path.startswith("/")
