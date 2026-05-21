import subprocess
import sqlite3
import sys
from pathlib import Path

import numpy as np

from app.database import Database


class FakePreprocessResult:
    def __init__(self, roi):
        self.roi = roi


class FakePreprocessor:
    def extract_full_hand_roi(self, frame):
        return FakePreprocessResult(frame[:, :, 0])

    def preprocess_roi_to_model_input(self, roi):
        value = int(round(float(roi.mean())))
        return np.full((224, 224, 3), value, dtype=np.float32)


class FakePalmProcessor:
    def _run_inference(self, processed):
        value = float(processed[0, 0, 0])
        return np.array([value, value + 1, value + 2, value + 3], dtype=np.float32)


def test_create_seed_variants_returns_seven_mild_variants():
    from app.services.seed_users import create_seed_variants

    roi = np.full((80, 80), 128, dtype=np.uint8)

    variants = create_seed_variants(roi)

    assert [variant.name for variant in variants] == [
        "original",
        "rotate_left",
        "rotate_right",
        "zoom_in",
        "zoom_out",
        "shift_left",
        "shift_right",
    ]
    assert all(variant.roi.shape == roi.shape for variant in variants)


def test_build_seed_embedding_averages_best_five_augmented_embeddings():
    from app.services.seed_users import build_seed_embedding

    frame = np.full((80, 80, 3), 20, dtype=np.uint8)
    processor = FakePalmProcessor()
    preprocessor = FakePreprocessor()

    result = build_seed_embedding(frame, processor, preprocessor)

    assert result.variant_count == 7
    assert result.selected_count == 5
    np.testing.assert_allclose(result.embedding, np.array([20, 21, 22, 23], dtype=np.float32))
    assert len(result.individual_embeddings) == 1
    np.testing.assert_allclose(result.individual_embeddings[0], result.embedding)


def test_seed_users_from_directory_uses_file_stems_and_preserves_logs(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "alice.JPG").write_bytes(b"image")
    (seed_dir / "bob.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    db.add_access_log(None, "Unknown", "DENIED", 0.2)

    def read_image(path):
        value = 10 if path.stem == "alice" else 30
        return np.full((80, 80, 3), value, dtype=np.uint8)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        FakePreprocessor(),
        read_image=read_image,
    )

    assert summary.created == ["alice", "bob"]
    assert summary.skipped == []
    assert db.count_access_logs() == 1
    assert [user["name"] for user in db.get_all_users()] == ["alice", "bob"]
    stored = db.get_all_embeddings()
    assert len(stored) == 2


def test_seed_users_from_person_folders_stores_all_valid_real_capture_embeddings(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "Dataset_Webcam"
    seed_dir.mkdir()
    for person in ["Afrizal", "Naufal"]:
        person_dir = seed_dir / person
        person_dir.mkdir()
        for index in range(6):
            (person_dir / f"capture_{index}.jpg").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")

    def read_image(path):
        base = 10 if path.parent.name == "Afrizal" else 30
        return np.full((80, 80, 3), base + int(path.stem.split("_")[-1]), dtype=np.uint8)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        FakePreprocessor(),
        read_image=read_image,
    )

    assert summary.created == ["Afrizal", "Naufal"]
    assert summary.skipped == []
    assert summary.failed == {}
    assert [user["name"] for user in db.get_all_users()] == ["Afrizal", "Naufal"]
    assert len(db.get_all_embeddings()) == 12
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            """
            SELECT users.name, COUNT(user_embeddings.id)
            FROM users
            JOIN user_embeddings ON users.id = user_embeddings.user_id
            GROUP BY users.id
            ORDER BY users.id
            """
        ).fetchall()
    assert rows == [("Afrizal", 6), ("Naufal", 6)]


def test_seed_users_skips_existing_names_by_default(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "alice.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    db.add_user("alice", np.ones(4, dtype=np.float32), individual_embeddings=[np.ones(4, dtype=np.float32)])

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        FakePreprocessor(),
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == []
    assert summary.skipped == ["alice"]
    assert len(db.get_all_users()) == 1


def test_seed_users_replace_removes_users_but_keeps_access_logs(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "alice.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    user_id = db.add_user("old", np.ones(4, dtype=np.float32), individual_embeddings=[np.ones(4, dtype=np.float32)])
    db.add_access_log(user_id, "old", "ALLOWED", 0.9)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        FakePreprocessor(),
        replace_users=True,
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == ["alice"]
    assert [user["name"] for user in db.get_all_users()] == ["alice"]
    assert db.count_access_logs() == 1
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT user_id FROM access_logs").fetchone()[0] is None


def test_seed_script_can_be_executed_directly():
    result = subprocess.run(
        [sys.executable, "scripts/seed_users.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Seed PalmGate users" in result.stdout


def test_seed_script_defaults_to_production_preprocessing_without_rembg():
    source = Path("scripts/seed_users.py").read_text()

    assert "NotebookPreprocessor(rembg_enabled=NOTEBOOK_REMBG_ENABLED)" in source
