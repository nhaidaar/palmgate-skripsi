import numpy as np
from fastapi.testclient import TestClient

from app.main import app


class FakeDB:
    def __init__(self):
        self.individual_embeddings = None
        self.embedding_hands = None

    def get_all_embeddings(self):
        return []

    def add_user(self, name, embedding, individual_embeddings=None, embedding_hands=None):
        self.name = name
        self.embedding = embedding
        self.individual_embeddings = individual_embeddings
        self.embedding_hands = embedding_hands
        return 123


class FakeProcessor:
    def __init__(self, duplicate=False):
        self.next_value = 0
        self.duplicate = duplicate
        self.similarity_checks = 0

    def get_embedding_from_roi(self, frame, rotation_angle):
        self.next_value += 1
        return np.full(4, self.next_value, dtype=np.float32)

    def get_embedding(self, frame):
        return self.get_embedding_from_roi(frame, 0.0)

    def compute_similarity(self, embedding, stored, threshold):
        self.similarity_checks += 1
        if self.duplicate:
            return {"status": "ALLOWED", "name": "Existing", "similarity": 0.9}
        return {"status": "DENIED", "name": "Unknown", "similarity": 0.1}


def test_register_rejects_fewer_than_ten_images():
    client = TestClient(app)

    response = client.post("/api/register", json={"name": "Alice", "images": ["img"] * 9})

    assert response.status_code == 400
    assert "5 left" in response.json()["detail"]
    assert "5 right" in response.json()["detail"]


def test_register_rejects_missing_hand_labels():
    client = TestClient(app)

    response = client.post("/api/register", json={"name": "Alice", "images": ["img"] * 10})

    assert response.status_code == 400
    assert "5 left" in response.json()["detail"]
    assert "5 right" in response.json()["detail"]


def test_register_rejects_unbalanced_hand_labels():
    client = TestClient(app)

    response = client.post(
        "/api/register",
        json={"name": "Alice", "images": ["img"] * 10, "hands": ["left"] * 10},
    )

    assert response.status_code == 400
    assert "5 left" in response.json()["detail"]
    assert "5 right" in response.json()["detail"]


def test_register_saves_ten_embeddings_with_hand_labels(monkeypatch):
    import app.main as main
    import app.routes.register as register_route

    fake_db = FakeDB()
    fake_processor = FakeProcessor()
    monkeypatch.setattr(main, "db", fake_db)
    monkeypatch.setattr(main, "palm_processor", fake_processor)
    monkeypatch.setattr(register_route, "decode_base64_image", lambda image: np.zeros((2, 2, 3), dtype=np.uint8))
    client = TestClient(app)

    hands = ["left"] * 5 + ["right"] * 5
    response = client.post(
        "/api/register",
        json={"name": "Alice", "images": ["img"] * 10, "hands": hands, "is_roi": True},
    )

    assert response.status_code == 200
    assert len(fake_db.individual_embeddings) == 10
    assert fake_db.embedding_hands == hands
    assert fake_processor.similarity_checks == 10


def test_register_rejects_duplicate_candidate_embedding(monkeypatch):
    import app.main as main
    import app.routes.register as register_route

    monkeypatch.setattr(main, "db", FakeDB())
    monkeypatch.setattr(main, "palm_processor", FakeProcessor(duplicate=True))
    monkeypatch.setattr(register_route, "decode_base64_image", lambda image: np.zeros((2, 2, 3), dtype=np.uint8))
    client = TestClient(app)

    response = client.post(
        "/api/register",
        json={
            "name": "Alice",
            "images": ["img"] * 10,
            "hands": ["left"] * 5 + ["right"] * 5,
            "is_roi": True,
        },
    )

    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]
