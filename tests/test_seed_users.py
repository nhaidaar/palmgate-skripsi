import subprocess
import sys
from pathlib import Path

import numpy as np

from app.database import Database


ROOT = Path(__file__).resolve().parent.parent


class FakePalmProcessor:
    def get_embedding(self, frame_rgb, tta_enabled=False):
        value = float(frame_rgb[0, 0, 0])
        return np.array([value, value + 1, value + 2, value + 3], dtype=np.float32)


def test_parse_seed_identity_requires_nim_name_format():
    from app.services.seed_users import parse_seed_identity

    assert parse_seed_identity("001_alice") == ("001", "alice")

    try:
        parse_seed_identity("alice")
    except RuntimeError as exc:
        assert "nim_name" in str(exc)
    else:
        raise AssertionError("Expected missing NIM to fail")


def test_build_seed_embedding_uses_runtime_embedding_path():
    from app.services.seed_users import build_seed_embedding

    frame = np.full((80, 80, 3), 20, dtype=np.uint8)

    result = build_seed_embedding(frame, FakePalmProcessor())

    assert result.variant_count == 1
    assert result.selected_count == 1
    np.testing.assert_allclose(result.embedding, np.array([20, 21, 22, 23], dtype=np.float32))
    assert len(result.individual_embeddings) == 1
    np.testing.assert_allclose(result.individual_embeddings[0], result.embedding)


def test_seed_users_from_directory_uses_nim_file_stems_and_preserves_logs(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "001_alice.JPG").write_bytes(b"image")
    (seed_dir / "002_bob.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    db.add_access_log(None, "Unknown", "DENIED", 0.2)

    def read_image(path):
        value = 10 if path.stem.endswith("alice") else 30
        return np.full((80, 80, 3), value, dtype=np.uint8)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        read_image=read_image,
    )

    assert summary.created == ["alice", "bob"]
    assert summary.skipped == []
    assert db.count_access_logs() == 1
    assert [user["nim"] for user in db.get_all_users()] == ["001", "002"]
    assert [user["name"] for user in db.get_all_users()] == ["alice", "bob"]
    assert len(db.get_all_embeddings()) == 2


def test_seed_users_from_person_folders_stores_one_template_per_person(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "Dataset_Webcam"
    seed_dir.mkdir()
    for label in ["001_Afrizal", "002_Naufal"]:
        person_dir = seed_dir / label
        person_dir.mkdir()
        for index in range(6):
            (person_dir / f"capture_{index}.jpg").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")

    def read_image(path):
        base = 10 if path.parent.name.endswith("Afrizal") else 30
        return np.full((80, 80, 3), base + int(path.stem.split("_")[-1]), dtype=np.uint8)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        read_image=read_image,
    )

    assert summary.created == ["Afrizal", "Naufal"]
    assert summary.skipped == []
    assert summary.failed == {}
    assert [user["nim"] for user in db.get_all_users()] == ["001", "002"]
    assert [user["name"] for user in db.get_all_users()] == ["Afrizal", "Naufal"]
    assert len(db.get_all_embeddings()) == 2


def test_seed_users_skips_existing_names_by_default(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "001_alice.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    db.add_user("alice", np.ones(4, dtype=np.float32), nim="001", individual_embeddings=[np.ones(4, dtype=np.float32)])

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == []
    assert summary.skipped == ["alice"]
    assert len(db.get_all_users()) == 1


def test_seed_users_replace_removes_users_but_keeps_access_logs(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "001_alice.JPG").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")
    user_id = db.add_user("old", np.ones(4, dtype=np.float32), nim="999", individual_embeddings=[np.ones(4, dtype=np.float32)])
    db.add_access_log(user_id, "old", "ALLOWED", 0.9)

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        replace_users=True,
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == ["alice"]
    assert [user["name"] for user in db.get_all_users()] == ["alice"]
    assert db.count_access_logs() == 1
    logs = db.get_access_logs(limit=10)
    assert logs[0]["user_id"] is None


def test_seed_users_requires_nim_name_label(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "alice.JPG").write_bytes(b"image")
    db = Database(tmp_path / "palmprint.db")

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == []
    assert "alice" in summary.failed
    assert "nim_name" in summary.failed["alice"]


def test_seed_script_can_be_executed_directly():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "seed_users.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "Seed PalmGate users" in result.stdout


def test_seed_script_loads_hand_model_for_runtime_preprocessing():
    source = (ROOT / "scripts" / "seed_users.py").read_text()

    assert "PalmProcessor()" in source
    assert "NotebookPreprocessor" not in source


def test_seed_users_plain_folders_fail_without_auto_demo_nim(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "Dataset_Webcam"
    seed_dir.mkdir()
    person_dir = seed_dir / "Afrizal"
    person_dir.mkdir()
    (person_dir / "capture_0.jpg").write_bytes(b"image")
    db = Database(tmp_path / "palmprint.db")

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        read_image=lambda path: np.full((80, 80, 3), 10, dtype=np.uint8),
    )

    assert summary.created == []
    assert "Afrizal" in summary.failed
    assert "nim_name" in summary.failed["Afrizal"]


def test_seed_users_plain_folders_use_stable_demo_nims_when_enabled(tmp_path):
    from app.services.seed_users import seed_users_from_directory

    seed_dir = tmp_path / "Dataset_Webcam"
    seed_dir.mkdir()
    for name in ["Afrizal", "Naufal", "Reza", "Rizky"]:
        person_dir = seed_dir / name
        person_dir.mkdir()
        for index in range(2):
            (person_dir / f"capture_{index}.jpg").write_bytes(b"image")

    db = Database(tmp_path / "palmprint.db")

    summary = seed_users_from_directory(
        seed_dir,
        db,
        FakePalmProcessor(),
        auto_demo_nim=True,
        read_image=lambda path: np.full((80, 80, 3), 10 + len(path.parent.name), dtype=np.uint8),
    )

    assert summary.created == ["Afrizal", "Naufal", "Reza", "Rizky"]
    assert summary.failed == {}
    assert [user["nim"] for user in db.get_all_users()] == ["SEED-001", "SEED-002", "SEED-003", "SEED-004"]
    assert [user["name"] for user in db.get_all_users()] == ["Afrizal", "Naufal", "Reza", "Rizky"]
    assert [entry["hand"] for entry in db.get_all_embeddings()] == ["unknown", "unknown", "unknown", "unknown"]


def test_seed_script_exposes_auto_demo_nim_flag():
    source = (ROOT / "scripts" / "seed_users.py").read_text()

    assert "--auto-demo-nim" in source
    assert "auto_demo_nim=args.auto_demo_nim" in source
