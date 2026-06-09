from fastapi.testclient import TestClient

from app.main import app


def test_start_device_registration_requires_runtime(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "device_runtime", None)
    client = TestClient(app)

    response = client.post("/api/device-registration/start", json={"name": "Alice"})

    assert response.status_code == 409


def test_start_device_registration_returns_session(monkeypatch):
    import app.main as main

    class FakeSession:
        id = "session-1"
        name = "Alice"
        current_sample_index = 0
        captured_samples = []

    class FakeRuntime:
        def start_registration(self, name):
            return FakeSession()

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.post("/api/device-registration/start", json={"name": "Alice"})

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-1"
    assert data["total_required"] == 10
    assert data["required_per_hand"] == 5
    assert data["current_hand"] == "left"
    assert data["left_count"] == 0
    assert data["right_count"] == 0


def test_device_registration_status_returns_session(monkeypatch):
    import app.main as main

    class FakeSession:
        id = "session-1"
        name = "Alice"
        current_sample_index = 2
        captured_samples = [{"hand": "left"}, {"hand": "left"}]
        last_guidance = {"acceptable": True}

    class FakeRuntime:
        worker_state = "registration_active"
        registration_session = FakeSession()

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.get("/api/device-registration/status")

    assert response.status_code == 200
    data = response.json()
    assert data["captured_count"] == 2
    assert data["total_required"] == 10
    assert data["required_per_hand"] == 5
    assert data["current_hand"] == "left"
    assert data["left_count"] == 2
    assert data["right_count"] == 0
    assert data["guidance"]["acceptable"] is True


def test_capture_endpoint_returns_sample(monkeypatch):
    import app.main as main

    class FakeRuntime:
        def capture_registration_sample(self):
            return {"sample_index": 0, "quality_score": 0.95}

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.post("/api/device-registration/capture")

    assert response.status_code == 200
    assert response.json()["sample_index"] == 0


def test_finalize_endpoint_returns_user(monkeypatch):
    import app.main as main

    class FakeRuntime:
        def finalize_registration(self):
            return {"user_id": 10, "name": "Alice", "stored_embeddings": 5}

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.post("/api/device-registration/finalize")

    assert response.status_code == 200
    assert response.json()["user_id"] == 10


def test_usb_preview_endpoint_returns_latest_frame(monkeypatch):
    import app.main as main

    class FakeRuntime:
        def get_latest_frame_jpeg(self):
            return b"\xff\xd8jpeg-data"

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.get("/api/device-registration/preview.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_usb_preview_stream_endpoint_returns_mjpeg_response(monkeypatch):
    import asyncio
    import app.main as main
    from app.routes.device_registration import preview_stream

    class FakeRuntime:
        def get_latest_frame_jpeg(self):
            return b"\xff\xd8jpeg-data"

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())

    response = asyncio.run(preview_stream())

    assert response.status_code == 200
    assert response.media_type.startswith("multipart/x-mixed-replace")
    assert response.headers["cache-control"] == "no-store"


def test_mjpeg_frames_yields_latest_frame():
    from app.routes.device_registration import mjpeg_frames

    class FakeRuntime:
        def get_latest_frame_jpeg(self):
            return b"\xff\xd8jpeg-data"

    chunk = next(mjpeg_frames(FakeRuntime()))

    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"\xff\xd8jpeg-data" in chunk


def test_usb_preview_endpoint_returns_503_without_frame(monkeypatch):
    import app.main as main

    class FakeRuntime:
        def get_latest_frame_jpeg(self):
            return None

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.get("/api/device-registration/preview.jpg")

    assert response.status_code == 503


def test_cancel_endpoint_cancels_session(monkeypatch):
    import app.main as main

    class FakeRuntime:
        def __init__(self):
            self.cancelled = False

        def cancel_registration(self):
            self.cancelled = True

    runtime = FakeRuntime()
    monkeypatch.setattr(main, "device_runtime", runtime)
    client = TestClient(app)

    response = client.post("/api/device-registration/cancel")

    assert response.status_code == 200
    assert runtime.cancelled is True
