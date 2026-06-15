import os
import tempfile
import numpy as np
import pytest
from app.database import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


def test_add_user(db):
    embedding = np.random.rand(128).astype(np.float32)
    user_id = db.add_user("TestUser", embedding, nim="000")
    assert user_id is not None
    assert user_id > 0


def test_get_all_users(db):
    emb1 = np.random.rand(128).astype(np.float32)
    emb2 = np.random.rand(128).astype(np.float32)
    db.add_user("Alice", emb1, nim="001")
    db.add_user("Bob", emb2, nim="002")
    users = db.get_all_users()
    assert len(users) == 2
    assert users[0]["name"] == "Alice"
    assert users[1]["name"] == "Bob"


def test_get_all_embeddings(db):
    emb = np.random.rand(128).astype(np.float32)
    db.add_user("Alice", emb, nim="001")
    embeddings = db.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0]["name"] == "Alice"
    np.testing.assert_array_almost_equal(embeddings[0]["embedding"], emb, decimal=5)


def test_add_user_stores_embedding_hand_labels(db):
    emb1 = np.ones(4, dtype=np.float32)
    emb2 = np.ones(4, dtype=np.float32) * 2

    db.add_user(
        "Alice",
        emb1,
        nim="001",
        individual_embeddings=[emb1, emb2],
        embedding_hands=["left", "right"],
    )

    embeddings = db.get_all_embeddings()
    assert len(embeddings) == 2
    assert embeddings[0]["hand"] == "left"
    assert embeddings[1]["hand"] == "right"


def test_get_all_embeddings_returns_unknown_hand_for_legacy_user(db):
    emb = np.ones(4, dtype=np.float32)
    db.add_user("Legacy", emb, nim="999")

    embeddings = db.get_all_embeddings()
    assert embeddings[0]["hand"] == "unknown"


def test_add_user_rejects_mismatched_embedding_hands_before_insert(db):
    emb = np.ones(4, dtype=np.float32)

    with pytest.raises(ValueError, match="embedding_hands"):
        db.add_user(
            "Alice",
            emb,
            nim="001",
            individual_embeddings=[emb, emb],
            embedding_hands=["left"],
        )

    assert db.get_all_users() == []


def test_add_user_rejects_empty_embedding_hands_before_insert(db):
    emb = np.ones(4, dtype=np.float32)

    with pytest.raises(ValueError, match="embedding_hands"):
        db.add_user(
            "Alice",
            emb,
            nim="001",
            individual_embeddings=[emb],
            embedding_hands=[],
        )

    assert db.get_all_users() == []


def test_add_user_requires_nim(db):
    emb = np.ones(128, dtype=np.float32)

    try:
        db.add_user("Alice", emb, nim=" ")
    except ValueError as exc:
        assert "NIM is required" in str(exc)
    else:
        raise AssertionError("Expected missing NIM to be rejected")


def test_get_all_users_includes_nim(db):
    emb = np.ones(128, dtype=np.float32)

    db.add_user("Alice", emb, nim="12345")

    users = db.get_all_users()
    assert users[0]["nim"] == "12345"
    assert users[0]["name"] == "Alice"


def test_duplicate_nim_is_rejected(db):
    emb = np.ones(128, dtype=np.float32)

    db.add_user("Alice", emb, nim="12345")

    try:
        db.add_user("Bob", emb, nim="12345")
    except ValueError as exc:
        assert "NIM already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate NIM to be rejected")


def test_delete_user(db):
    emb = np.random.rand(128).astype(np.float32)
    user_id = db.add_user("ToDelete", emb, nim="003")
    assert db.delete_user(user_id) is True
    assert len(db.get_all_users()) == 0


def test_add_access_log(db):
    db.add_access_log(
        user_id=None,
        matched_name="Unknown",
        status="DENIED",
        similarity=0.3,
        duration_ms=123,
        description="similar to Naufal",
    )
    logs = db.get_access_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["status"] == "DENIED"
    assert logs[0]["matched_name"] == "Unknown"
    assert logs[0]["duration_ms"] == 123
    assert logs[0]["description"] == "similar to Naufal"


def test_delete_user_preserves_access_logs(db):
    emb = np.random.rand(128).astype(np.float32)
    user_id = db.add_user("LoggedUser", emb, nim="004")
    db.add_access_log(user_id=user_id, matched_name="LoggedUser", status="ALLOWED", similarity=0.9)

    assert db.delete_user(user_id) is True

    logs = db.get_access_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["matched_name"] == "LoggedUser"
    assert logs[0]["user_id"] is None


def test_get_access_logs_ordered(db):
    db.add_access_log(None, "First", "DENIED", 0.1)
    db.add_access_log(None, "Second", "DENIED", 0.2)
    logs = db.get_access_logs(limit=10)
    assert logs[0]["matched_name"] == "Second"
