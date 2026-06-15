from fastapi.testclient import TestClient

from app.main import app


def test_start_device_registration_requires_runtime(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "device_runtime", None)
    client = TestClient(app)

    response = client.post("/api/device-registration/start", json={"nim": "12345", "name": "Alice"})

    assert response.status_code == 409


def test_start_registration_requires_nim(monkeypatch):
    import app.main as main

    class FakeRuntime:
        pass

    monkeypatch.setattr(main, "device_runtime", FakeRuntime())
    client = TestClient(app)

    response = client.post("/api/device-registration/start", json={"name": "Alice"})

    assert response.status_code == 400
    assert "NIM is required" in response.json()["detail"]


def test_start_device_registration_returns_session(monkeypatch):
    import app.main as main

    class FakeSession:
        id = "session-1"
        nim = "12345"
        name = "Alice"
        current_sample_index = 0
        captured_samples = []

    class FakeRuntime:
        def __init__(self):
            self.started = None

        def start_registration(self, nim, name):
            self.started = (nim, name)
            return FakeSession()

    runtime = FakeRuntime()
    monkeypatch.setattr(main, "device_runtime", runtime)
    client = TestClient(app)

    response = client.post("/api/device-registration/start", json={"nim": "12345", "name": "Alice"})

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-1"
    assert data["nim"] == "12345"
    assert runtime.started == ("12345", "Alice")
    assert data["total_required"] == 10
    assert data["required_per_hand"] == 5
    assert data["current_hand"] == "left"
    assert data["left_count"] == 0
    assert data["right_count"] == 0


def test_device_registration_status_returns_session(monkeypatch):
    import app.main as main

    class FakeSession:
        id = "session-1"
        nim = "12345"
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
    import asyncio
    from app.routes.device_registration import mjpeg_frames

    class FakeRuntime:
        def get_latest_frame_jpeg(self):
            return b"\xff\xd8jpeg-data"

    chunk = asyncio.run(anext(mjpeg_frames(FakeRuntime())))

    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"\xff\xd8jpeg-data" in chunk


def test_mjpeg_frames_uses_runtime_preview_interval(monkeypatch):
    import asyncio
    import pytest
    from app.routes import device_registration
    from app.routes.device_registration import mjpeg_frames

    class FakeRuntime:
        preview_frame_interval_ms = 25

        def get_latest_frame_jpeg(self):
            return b"\xff\xd8jpeg-data"

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        raise RuntimeError("stop")

    async def run_stream():
        frames = mjpeg_frames(FakeRuntime())
        await anext(frames)
        await anext(frames)

    monkeypatch.setattr(device_registration.asyncio, "sleep", fake_sleep)
    with pytest.raises(RuntimeError):
        asyncio.run(run_stream())

    assert sleeps == [0.025]


def test_scan_event_stream_formats_events():
    import asyncio
    import json
    from app.routes.device_registration import scan_event_stream

    class FakeSubscriber:
        def get(self, block=True, timeout=None):
            return {"stage": "recognized", "name": "Alice"}

    class FakeBroadcaster:
        def __init__(self):
            self.unsubscribed = False

        def subscribe(self):
            return FakeSubscriber()

        def unsubscribe(self, subscriber):
            self.unsubscribed = True

    class FakeRuntime:
        scan_broadcaster = FakeBroadcaster()

    async def first_event():
        stream = scan_event_stream(FakeRuntime())
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    chunk = asyncio.run(first_event())

    assert chunk.startswith("data: ")
    assert json.loads(chunk.removeprefix("data: ").strip()) == {"stage": "recognized", "name": "Alice"}


def test_scan_event_stream_yields_keepalive_on_empty_queue():
    import asyncio
    import queue
    from app.routes.device_registration import scan_event_stream

    class FakeSubscriber:
        def get(self, block=True, timeout=None):
            raise queue.Empty

    class FakeBroadcaster:
        def subscribe(self):
            return FakeSubscriber()

        def unsubscribe(self, subscriber):
            pass

    class FakeRuntime:
        scan_broadcaster = FakeBroadcaster()

    async def first_event():
        stream = scan_event_stream(FakeRuntime())
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    assert asyncio.run(first_event()) == ": keepalive\n\n"


def test_scan_event_stream_unsubscribes_on_close():
    import asyncio
    from app.routes.device_registration import scan_event_stream

    class FakeSubscriber:
        def get(self, block=True, timeout=None):
            return {"stage": "recognized"}

    class FakeBroadcaster:
        def __init__(self):
            self.subscriber = FakeSubscriber()
            self.unsubscribed = False

        def subscribe(self):
            return self.subscriber

        def unsubscribe(self, subscriber):
            self.unsubscribed = subscriber is self.subscriber

    class FakeRuntime:
        def __init__(self):
            self.scan_broadcaster = FakeBroadcaster()

    runtime = FakeRuntime()

    async def consume_and_close():
        stream = scan_event_stream(runtime)
        await anext(stream)
        await stream.aclose()

    asyncio.run(consume_and_close())

    assert runtime.scan_broadcaster.unsubscribed is True


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
