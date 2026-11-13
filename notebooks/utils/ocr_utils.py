"""
OCR utilities for food label transparency project.
"""

import cv2
import numpy as np
from typing import Optional
import easyocr


def extract_text_with_preprocessing(image_path: str, reader: easyocr.Reader) -> str:
    """
    Extract text from image with OpenCV preprocessing to improve OCR accuracy.

    Args:
        image_path: Path to the input image
        reader: Initialized easyocr.Reader object

    Returns:
        Extracted text string
    """
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to improve OCR accuracy
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Run OCR on processed image
        results = reader.readtext(processed, detail=0)
        return " ".join(results)
    except Exception as preprocessing_error:
        # Fallback to original method if OpenCV processing fails
        try:
            results = reader.readtext(image_path, detail=0)
            return " ".join(results)
        except Exception as fallback_error:
            raise RuntimeError(
                f"OCR extraction failed for {image_path}; preprocessing failed "
                f"({preprocessing_error}) and the original-image fallback failed "
                f"({fallback_error})"
            ) from fallback_error


def extract_text_simple(image_path: str, reader: easyocr.Reader) -> str:
    """
    Extract text from image using easyocr directly (no preprocessing).

    Args:
        image_path: Path to the input image
        reader: Initialized easyocr.Reader object

    Returns:
        Extracted text string
    """
    try:
        results = reader.readtext(image_path, detail=0)
        return " ".join(results)
    except Exception as error:
        raise RuntimeError(f"OCR extraction failed for {image_path}") from error


def preprocess_ocr_image(image_path: str) -> Optional[np.ndarray]:
    """
    Preprocess image for OCR using OpenCV.

    Args:
        image_path: Path to the input image

    Returns:
        Preprocessed image as numpy array, or None if loading fails
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return processed
    except Exception:
        return None