import cv2
import numpy as np

from app.notebook_preprocessing import NotebookPreprocessResult, NotebookPreprocessor


def test_preprocess_roi_to_model_input_returns_224_rgb_float32():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    roi = np.full((120, 120), 128, dtype=np.uint8)

    result = preprocessor.preprocess_roi_to_model_input(roi)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32


def test_extract_full_hand_roi_returns_none_for_blank_frame():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = preprocessor.extract_full_hand_roi(frame)

    assert result is None


def test_preprocess_result_has_quality_fields():
    result = NotebookPreprocessResult(
        roi=np.zeros((100, 100), dtype=np.uint8),
        model_input=np.zeros((224, 224, 3), dtype=np.float32),
        bbox=(10, 20, 110, 120),
        rotation_degrees=3.0,
        roi_size=100,
        contour_area=5000.0,
    )

    assert result.bbox == (10, 20, 110, 120)
    assert result.rotation_degrees == 3.0
    assert result.roi_size == 100
    assert result.contour_area == 5000.0


def test_threshold_mask_rejects_empty_hand():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    gray = np.zeros((640, 480), dtype=np.uint8)

    mask = preprocessor._threshold_hand(gray)

    assert mask.sum() == 0


def test_find_largest_contour_returns_none_for_empty_mask():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    mask = np.zeros((640, 480), dtype=np.uint8)

    contour = preprocessor._find_largest_contour(mask)

    assert contour is None


def test_preprocess_full_hand_uses_rembg_disabled_path_for_tests():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (220, 120), (420, 420), (255, 255, 255), -1)

    prepared = preprocessor._prepare_hand_mask_input(frame)

    assert prepared.ndim == 2
    assert prepared.dtype == np.uint8


def test_calculate_roi_returns_none_with_too_few_minima():
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    mask = np.zeros((800, 640), dtype=np.uint8)
    gray = np.zeros((640, 480), dtype=np.uint8)
    contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)

    result = preprocessor._calculate_roi(mask, gray, contour)

    assert result is None


def test_extract_full_hand_roi_rejects_tiny_roi(monkeypatch):
    preprocessor = NotebookPreprocessor(rembg_enabled=False)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(
        preprocessor,
        "_calculate_roi",
        lambda mask, gray, contour: (np.zeros((20, 20), dtype=np.uint8), (0, 0, 20, 20), 0.0),
    )

    result = preprocessor.extract_full_hand_roi(frame)

    assert result is None
