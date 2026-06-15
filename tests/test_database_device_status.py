import os
import tempfile

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



def test_device_status_round_trip(db):
    db.upsert_device_status(
        worker_state="running",
        camera_connected=True,
        last_error=None,
        fps=4.0,
        last_inference_ms=320.5,
    )

    row = db.get_device_status()

    assert row["worker_state"] == "running"
    assert row["camera_connected"] == 1
    assert row["fps"] == 4.0
    assert row["last_inference_ms"] == 320.5
