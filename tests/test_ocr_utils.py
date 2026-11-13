import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from utils.ocr_utils import extract_text_simple, extract_text_with_preprocessing


def test_simple_ocr_raises_if_reader_fails():
    reader = MagicMock()
    reader.readtext.side_effect = ValueError("reader failed")

    with pytest.raises(RuntimeError, match="OCR extraction failed"):
        extract_text_simple("label.png", reader)


def test_preprocessed_ocr_raises_when_both_attempts_fail(monkeypatch):
    import utils.ocr_utils as ocr_utils

    monkeypatch.setattr(
        ocr_utils.cv2, "imread", lambda _: np.zeros((11, 11, 3), dtype=np.uint8)
    )
    reader = MagicMock()
    reader.readtext.side_effect = [ValueError("processed failed"), ValueError("fallback failed")]

    with pytest.raises(RuntimeError, match="preprocessing failed"):
        extract_text_with_preprocessing("label.png", reader)

    assert reader.readtext.call_count == 2
