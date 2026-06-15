import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def encoded_image():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", frame)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii")


def test_recognize_full_frame_uses_notebook_embedding(monkeypatch):
    import app.main as main

    class FakeProcessor:
        def __init__(self):
            self.used_notebook = False

        def get_embedding_from_notebook_frame(self, frame, tta_enabled=False):
            self.used_notebook = True
            self.tta_enabled = tta_enabled
            return np.ones(4, dtype=np.float32)

        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.1,
                "closest_match": None,
                "user_id": None,
            }

    class FakeDB:
        def get_all_embeddings(self):
            return []

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            pass

    processor = FakeProcessor()
    monkeypatch.setattr(main, "palm_processor", processor)
    monkeypatch.setattr(main, "db", FakeDB())
    client = TestClient(app)

    response = client.post("/api/recognize", json={"image": encoded_image(), "is_roi": False})

    assert response.status_code == 200
    assert processor.used_notebook is True


def test_recognize_uses_recognition_tta_flag(monkeypatch):
    import app.main as main
    import app.routes.recognize as recognize_route

    class FakeProcessor:
        def __init__(self):
            self.tta_enabled = None

        def get_embedding_from_notebook_frame(self, frame, tta_enabled=False):
            self.tta_enabled = tta_enabled
            return np.ones(4, dtype=np.float32)

        def compute_similarity(self, embedding, stored, threshold):
            return {"status": "DENIED", "name": "Unknown", "similarity": 0.1, "closest_match": None, "user_id": None}

    class FakeDB:
        def get_all_embeddings(self):
            return []

        def add_access_log(self, *args, **kwargs):
            pass

    fake_processor = FakeProcessor()
    monkeypatch.setattr(main, "palm_processor", fake_processor)
    monkeypatch.setattr(main, "db", FakeDB())
    monkeypatch.setattr(recognize_route, "RECOGNITION_TTA_ENABLED", True)
    monkeypatch.setattr(recognize_route, "decode_base64_image", lambda image: np.zeros((2, 2, 3), dtype=np.uint8))

    client = TestClient(app)
    response = client.post("/api/recognize", json={"image": "img"})

    assert response.status_code == 200
    assert fake_processor.tta_enabled is True
