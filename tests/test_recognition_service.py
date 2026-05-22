import logging
import numpy as np



def test_match_and_log_allowed_result(caplog):
    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "ALLOWED",
                "name": "Naufal",
                "similarity": 0.91,
                "closest_match": "Naufal",
                "user_id": 1,
            }

    class FakeDB:
        def __init__(self):
            self.logged = None

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            self.logged = (user_id, matched_name, status, similarity, duration_ms, description)

    from app.services.recognition_service import match_embedding_and_log

    caplog.set_level(logging.INFO, logger="palmgate")

    db = FakeDB()
    result = match_embedding_and_log(FakeProcessor(), db, np.ones(4, dtype=np.float32), 0.75, duration_ms=321)

    assert result["status"] == "ALLOWED"
    assert db.logged == (1, "Naufal", "ALLOWED", 0.91, 321, None)
    assert "ALLOWED | user=Naufal | similarity=0.9100" in caplog.text


def test_match_and_log_denied_result_describes_closest_match():
    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.63,
                "closest_match": "Naufal",
                "user_id": None,
            }

    class FakeDB:
        def __init__(self):
            self.logged = None

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            self.logged = (user_id, matched_name, status, similarity, duration_ms, description)

    from app.services.recognition_service import match_embedding_and_log

    db = FakeDB()
    result = match_embedding_and_log(FakeProcessor(), db, np.ones(4, dtype=np.float32), 0.75, duration_ms=456)

    assert result["status"] == "DENIED"
    assert db.logged == (None, "Unknown", "DENIED", 0.63, 456, "similar to Naufal")
