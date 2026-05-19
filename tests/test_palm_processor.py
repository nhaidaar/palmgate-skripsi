from types import SimpleNamespace

import numpy as np
import pytest
from app.palm_processor import PalmProcessor


@pytest.fixture
def processor():
    proc = PalmProcessor(model_path=None, hand_model_path=None)
    yield proc
    proc.close()


def test_extract_palm_roi_no_hand(processor):
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.extract_palm_roi(black_frame)
    assert result is None


def test_apply_clahe(processor):
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    enhanced = processor.apply_clahe(gray_img)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_preprocess_roi(processor):
    roi = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    processed = processor.preprocess_roi(roi)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32


def test_palm_processor_uses_notebook_rembg_config(monkeypatch):
    monkeypatch.setenv("NOTEBOOK_REMBG_ENABLED", "0")

    import app.config as config
    import app.palm_processor as palm_processor_module
    import importlib
    importlib.reload(config)
    importlib.reload(palm_processor_module)

    proc = palm_processor_module.PalmProcessor(model_path=None, hand_model_path=None)
    try:
        assert proc.notebook_preprocessor.rembg_enabled is False
    finally:
        proc.close()


def test_get_embedding_from_notebook_frame_returns_none_when_preprocessing_fails(processor):
    class FakeNotebookPreprocessor:
        def extract_full_hand_roi(self, frame):
            return None

    processor.notebook_preprocessor = FakeNotebookPreprocessor()

    result = processor.get_embedding_from_notebook_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is None


def test_get_embedding_from_notebook_frame_runs_inference(processor, monkeypatch):
    class FakeResult:
        model_input = np.ones((224, 224, 3), dtype=np.float32)

    class FakeNotebookPreprocessor:
        def extract_full_hand_roi(self, frame):
            return FakeResult()

    processor.notebook_preprocessor = FakeNotebookPreprocessor()
    monkeypatch.setattr(processor, "_run_inference", lambda processed: np.ones(4, dtype=np.float32))

    result = processor.get_embedding_from_notebook_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    np.testing.assert_array_equal(result, np.ones(4, dtype=np.float32))


def fake_landmarks(points):
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    for index, point in points.items():
        landmarks[index] = SimpleNamespace(x=point[0], y=point[1])
    return landmarks


def test_registration_guidance_metrics_reports_no_hand(processor):
    class FakeLandmarker:
        def close(self):
            pass

        def detect(self, image):
            return SimpleNamespace(hand_landmarks=[])

    processor._hand_landmarker = FakeLandmarker()

    metrics = processor.get_registration_guidance_metrics(
        np.full((100, 200, 3), 120, dtype=np.uint8),
        previous_metrics=None,
    )

    assert metrics["hand_detected"] is False
    assert metrics["hand_clipped"] is True
    assert metrics["steady"] is False


def test_registration_guidance_metrics_uses_mediapipe_landmarks(processor):
    class FakeLandmarker:
        def close(self):
            pass

        def detect(self, image):
            return SimpleNamespace(hand_landmarks=[fake_landmarks({
                0: (0.50, 0.80),
                5: (0.35, 0.45),
                9: (0.50, 0.42),
                17: (0.65, 0.45),
            })])

    processor._hand_landmarker = FakeLandmarker()

    metrics = processor.get_registration_guidance_metrics(
        np.full((100, 200, 3), 120, dtype=np.uint8),
        previous_metrics={
            "hand_detected": True,
            "height_ratio": 0.38,
            "rotation_degrees": 0.0,
            "center_x_ratio": 0.50,
        },
    )

    assert metrics["hand_detected"] is True
    assert metrics["hand_clipped"] is False
    assert metrics["height_ratio"] > 0
    assert 0.45 <= metrics["center_x_ratio"] <= 0.55
    assert abs(metrics["rotation_degrees"]) < 1.0
    assert metrics["brightness"] == 120.0
    assert "blur_score" in metrics
    assert metrics["steady"] is True


def test_registration_guidance_metrics_detects_clipped_hand(processor):
    class FakeLandmarker:
        def close(self):
            pass

        def detect(self, image):
            return SimpleNamespace(hand_landmarks=[fake_landmarks({
                0: (0.01, 0.80),
                5: (0.35, 0.45),
                9: (0.50, 0.42),
                17: (0.65, 0.45),
            })])

    processor._hand_landmarker = FakeLandmarker()

    metrics = processor.get_registration_guidance_metrics(
        np.full((100, 200, 3), 120, dtype=np.uint8),
        previous_metrics=None,
    )

    assert metrics["hand_clipped"] is True
