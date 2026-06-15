# Orange Pi Device-First Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt this repo into a single project that supports (1) Orange Pi-hosted admin/dashboard access with iPhone browser camera testing now and (2) 24x7 USB-camera recognition later.

**Architecture:** Keep FastAPI, SQLite, and the existing palm-processing pipeline as the shared core. Add a small shared recognition service, a DB-backed device status layer, and a background USB-camera runtime that writes heartbeats and recognition events into the same database while the existing web UI remains the admin dashboard.

**Tech Stack:** FastAPI, SQLite, OpenCV, MediaPipe Tasks, TFLite Runtime, HTML/CSS/JavaScript, pytest, systemd.

---

## Context

The current app is browser-driven: the camera loop lives in `app/static/app.js`, while recognition, registration, and persistence already live server-side in `app/palm_processor.py`, `app/routes/*.py`, and `app/database.py`. That makes this repo a good base for Orange Pi.

The user wants one project, not a second repo. Near term, the Orange Pi should host the app and accept an iPhone browser as a temporary camera client over Wi-Fi. Long term, the Orange Pi should run 24x7 with an always-active USB camera and still expose an admin dashboard for registration, users, logs, and device status.

The recommended approach is to keep the current browser scan/register flow for testing and enrollment, while adding a separate same-repo USB-camera worker for always-on recognition. Use SQLite as the shared integration point between the web app and the device worker.

## Existing Code To Reuse

- `app/palm_processor.py:24-248`
  - Reuse `PalmProcessor.get_embedding()` for full-frame device recognition.
  - Reuse `PalmProcessor.get_embedding_from_roi()` for browser-supplied ROI flows.
  - Reuse `PalmProcessor.compute_similarity()` as the single matching implementation.
- `app/routes/recognize.py:27-71`
  - Reuse `decode_base64_image()`.
  - Keep `/api/recognize` for browser-based testing and diagnostics.
- `app/routes/register.py:24-75`
  - Keep `/api/register` as the admin enrollment path from phone browser.
- `app/database.py:47-146`
  - Reuse `add_user()`, `get_all_users()`, `get_all_embeddings()`, `delete_user()`, `add_access_log()`, `get_access_logs()`, `count_access_logs()`.
- `app/static/index.html:37-360` and `app/static/app.js:367-790`
  - Reuse the existing Scan/Register/Log dashboard structure.
  - Extend it with device status rather than replacing it.
- `app/main.py:22-59`
  - Reuse FastAPI lifespan startup/shutdown and router composition.

## Implementation Notes

- Use the current `palm_embedding.tflite` runtime model and keep browser-based testing available.
- Do not remove browser-based scanning; keep it as the easiest test path from iPhone.
- Favor `tflite_runtime` on Orange Pi. The existing fallback in `app/palm_processor.py:54-66` already supports that direction.
- Target a USB camera first (`/dev/video0` via OpenCV). Do not introduce GPIO/door-relay work in this change.

---

### Task 1: Extend configuration for browser mode + device mode

**Files:**
- Modify: `app/config.py:1-19`
- Modify: `requirements.txt:1-10`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
import importlib
import os


def test_device_runtime_env_overrides(monkeypatch):
    monkeypatch.setenv("DEVICE_RUNTIME_ENABLED", "1")
    monkeypatch.setenv("CAMERA_SOURCE", "usb")
    monkeypatch.setenv("CAMERA_DEVICE_INDEX", "0")
    monkeypatch.setenv("APP_HOST", "0.0.0.0")

    import app.config as config
    importlib.reload(config)

    assert config.DEVICE_RUNTIME_ENABLED is True
    assert config.CAMERA_SOURCE == "usb"
    assert config.CAMERA_DEVICE_INDEX == 0
    assert config.APP_HOST == "0.0.0.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_device_runtime_env_overrides -v`
Expected: FAIL because the new config constants do not exist yet.

**Step 3: Write minimal implementation**

Add env-driven settings to `app/config.py` without introducing a new config framework:

```python
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEVICE_RUNTIME_ENABLED = os.getenv("DEVICE_RUNTIME_ENABLED", "0") == "1"
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "browser")
CAMERA_DEVICE_INDEX = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
DEVICE_FRAME_INTERVAL_MS = int(os.getenv("DEVICE_FRAME_INTERVAL_MS", "250"))
DEVICE_HOLD_MS = int(os.getenv("DEVICE_HOLD_MS", "1200"))
DEVICE_COOLDOWN_MS = int(os.getenv("DEVICE_COOLDOWN_MS", "3000"))
DEVICE_STATUS_HEARTBEAT_MS = int(os.getenv("DEVICE_STATUS_HEARTBEAT_MS", "1000"))
```

Update `requirements.txt` so Orange Pi can prefer `tflite-runtime` while still keeping current development dependencies explicit.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_device_runtime_env_overrides -v`
Expected: PASS

**Step 5: Run the current fast unit suite**

Run: `pytest tests/test_config.py tests/test_database.py tests/test_palm_processor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/config.py requirements.txt tests/test_config.py
git commit -m "feat: add config for browser and device runtime"
```

---

### Task 2: Add DB-backed device status so web and worker can share runtime state

**Files:**
- Modify: `app/database.py:16-149`
- Create: `tests/test_database_device_status.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_database_device_status.py::test_device_status_round_trip -v`
Expected: FAIL because the table and methods do not exist.

**Step 3: Write minimal implementation**

Extend `_create_tables()` with a single-row table:

```sql
CREATE TABLE IF NOT EXISTS device_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    worker_state TEXT NOT NULL,
    camera_connected INTEGER NOT NULL,
    last_error TEXT,
    fps REAL,
    last_inference_ms REAL,
    last_recognition_at TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Add methods:
- `upsert_device_status(...)`
- `get_device_status()`

Keep this table tiny and overwrite the same row instead of logging every frame.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_database_device_status.py::test_device_status_round_trip -v`
Expected: PASS

**Step 5: Run database tests together**

Run: `pytest tests/test_database.py tests/test_database_device_status.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/database.py tests/test_database_device_status.py
git commit -m "feat: persist shared device status in sqlite"
```

---

### Task 3: Extract shared recognition logic for HTTP and device runtime reuse

**Files:**
- Create: `app/services/recognition_service.py`
- Modify: `app/routes/recognize.py:27-71`
- Create: `tests/test_recognition_service.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_match_and_log_allowed_result():
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
        def add_access_log(self, user_id, matched_name, status, similarity):
            self.logged = (user_id, matched_name, status, similarity)

    from app.services.recognition_service import match_embedding_and_log

    db = FakeDB()
    result = match_embedding_and_log(FakeProcessor(), db, np.ones(4, dtype=np.float32), 0.75)

    assert result["status"] == "ALLOWED"
    assert db.logged == (1, "Naufal", "ALLOWED", 0.91)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_recognition_service.py::test_match_and_log_allowed_result -v`
Expected: FAIL because the service module does not exist.

**Step 3: Write minimal implementation**

Create a shared helper:

```python
def match_embedding_and_log(palm_processor, db, embedding, threshold):
    stored = db.get_all_embeddings()
    result = palm_processor.compute_similarity(embedding, stored, threshold)
    db.add_access_log(
        user_id=result["user_id"],
        matched_name=result["name"] if result["status"] == "ALLOWED" else "Unknown",
        status=result["status"],
        similarity=result["similarity"],
    )
    return result
```

Refactor `app/routes/recognize.py` to call the service instead of duplicating the match/log block.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_recognition_service.py::test_match_and_log_allowed_result -v`
Expected: PASS

**Step 5: Run related tests**

Run: `pytest tests/test_recognition_service.py tests/test_database.py tests/test_palm_processor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/services/recognition_service.py app/routes/recognize.py tests/test_recognition_service.py
git commit -m "refactor: share recognition match and log flow"
```

---

### Task 4: Add device status/health endpoints for the admin dashboard

**Files:**
- Create: `app/routes/status.py`
- Modify: `app/main.py:8-38`
- Create: `tests/test_status_routes.py`

**Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient
from app.main import app


def test_status_endpoint_returns_device_status():
    client = TestClient(app)
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "device" in data
    assert "database" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_status_routes.py::test_status_endpoint_returns_device_status -v`
Expected: FAIL because the route does not exist.

**Step 3: Write minimal implementation**

Add `app/routes/status.py` with a route like:

```python
@router.get("/api/status")
async def status():
    from app.main import db
    row = db.get_device_status()
    return {
        "app": {"mode": "hybrid", "version": "local"},
        "database": {"path": str(DB_PATH)},
        "device": row or {
            "worker_state": "disabled",
            "camera_connected": 0,
            "last_error": None,
            "fps": None,
            "last_inference_ms": None,
            "last_recognition_at": None,
        },
    }
```

Register the router in `app/main.py`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_status_routes.py::test_status_endpoint_returns_device_status -v`
Expected: PASS

**Step 5: Run route tests**

Run: `pytest tests/test_status_routes.py tests/test_database_device_status.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/routes/status.py app/main.py tests/test_status_routes.py
git commit -m "feat: expose app and device status api"
```

---

### Task 5: Add USB-camera device runtime with hold/cooldown state machine

**Files:**
- Create: `app/camera.py`
- Create: `app/device_runtime.py`
- Modify: `app/main.py:22-29`
- Create: `tests/test_device_runtime.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_runtime_recognizes_after_hold_threshold():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self): self.now_ms = 0
        def now(self): return self.now_ms

    class FakeCamera:
        def read(self): return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_embedding(self, frame): return np.ones(4, dtype=np.float32)

    class FakeDB:
        def __init__(self): self.logged = []
        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]
        def add_access_log(self, user_id, matched_name, status, similarity):
            self.logged.append((user_id, matched_name, status, similarity))
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
        cooldown_ms=3000,
    )

    runtime.clock.now_ms = 0
    runtime.tick()
    runtime.clock.now_ms = 1200
    runtime.tick()

    assert runtime.db.logged[0][2] == "ALLOWED"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_device_runtime.py::test_runtime_recognizes_after_hold_threshold -v`
Expected: FAIL because the runtime module does not exist.

**Step 3: Write minimal implementation**

Create:
- `app/camera.py` with a tiny `OpenCVCameraSource` wrapper around `cv2.VideoCapture`
- `app/device_runtime.py` with:
  - `start()` / `stop()` methods
  - `tick()` for testable one-iteration behavior
  - hold timer
  - cooldown timer
  - status heartbeat updates via `db.upsert_device_status()`
  - reuse `match_embedding_and_log()` from `app/services/recognition_service.py`

Minimal runtime logic:

```python
if now_ms < self.cooldown_until_ms:
    return

frame = self.camera.read()
embedding = self.palm_processor.get_embedding(frame)
if embedding is None:
    self.hand_seen_since_ms = None
    return

if self.hand_seen_since_ms is None:
    self.hand_seen_since_ms = now_ms
    return

if now_ms - self.hand_seen_since_ms >= self.hold_ms:
    match_embedding_and_log(...)
    self.cooldown_until_ms = now_ms + self.cooldown_ms
    self.hand_seen_since_ms = None
```

Hook worker startup into FastAPI lifespan only when `DEVICE_RUNTIME_ENABLED` and `CAMERA_SOURCE == "usb"`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_device_runtime.py::test_runtime_recognizes_after_hold_threshold -v`
Expected: PASS

**Step 5: Run focused verification**

Run: `pytest tests/test_device_runtime.py tests/test_recognition_service.py tests/test_database_device_status.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/camera.py app/device_runtime.py app/main.py tests/test_device_runtime.py
git commit -m "feat: add usb camera device runtime"
```

---

### Task 6: Extend the web UI into an admin dashboard with device status

**Files:**
- Modify: `app/static/index.html:16-360`
- Modify: `app/static/app.js:22-790`
- Modify: `app/static/style.css`

**Step 1: Add the failing UI contract test notes**

Create a manual verification checklist in the plan execution notes first, then automate later if the repo adds browser tests. The required UI states are:
- status badge shows online/offline worker state
- scan tab still works with browser camera
- register tab still works with browser camera
- log tab still shows access history
- new device status card shows camera, worker, and last recognition fields from `/api/status`

**Step 2: Implement minimal HTML changes**

Add a compact device status section without rebuilding the whole UI:

```html
<div class="device-status-card" id="deviceStatusCard">
  <div><span>Worker</span><strong id="deviceWorkerState">disabled</strong></div>
  <div><span>Camera</span><strong id="deviceCameraState">offline</strong></div>
  <div><span>FPS</span><strong id="deviceFps">—</strong></div>
  <div><span>Last recognition</span><strong id="deviceLastRecognition">—</strong></div>
</div>
```

Place it in the existing Scan panel result column so the dashboard stays simple and mobile-friendly.

**Step 3: Implement minimal JS changes**

Add polling in `app/static/app.js`:

```javascript
async function loadStatus() {
  const data = await fetch('/api/status').then((r) => r.json());
  $('deviceWorkerState').textContent = data.device.worker_state ?? 'disabled';
  $('deviceCameraState').textContent = data.device.camera_connected ? 'connected' : 'offline';
  $('deviceFps').textContent = data.device.fps ?? '—';
  $('deviceLastRecognition').textContent = data.device.last_recognition_at ?? '—';
}
```

Call `loadStatus()` at startup and on a timer.

**Step 4: Verify the browser dashboard manually**

Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
Expected:
- dashboard loads
- existing tabs still work
- status card renders even when the device worker is disabled

**Step 5: Verify from iPhone on the LAN**

Open the Orange Pi-hosted page from Safari on the same Wi-Fi.
Expected:
- page loads
- camera permission can be requested for browser scan/register mode
- register and scan still work against the Orange Pi backend

**Step 6: Commit**

```bash
git add app/static/index.html app/static/app.js app/static/style.css
git commit -m "feat: add device status to admin dashboard"
```

---

### Task 7: Add Orange Pi deployment assets and document the operator flow

**Files:**
- Create: `deploy/orangepi/palmgate-api.service`
- Create: `deploy/orangepi/palmgate-device.service`
- Modify: `README.md:1-74`

**Step 1: Write the deployment notes section**

Document two runtime roles:
- `palmgate-api.service` → FastAPI + dashboard
- `palmgate-device.service` → USB camera worker

**Step 2: Add systemd service files**

`deploy/orangepi/palmgate-api.service`

```ini
[Unit]
Description=PalmGate API
After=network.target

[Service]
WorkingDirectory=/opt/palmgate
Environment=MODEL_PATH=/opt/palmgate/palm_embedding.tflite
Environment=MODEL_METADATA_PATH=/opt/palmgate/model_metadata.json
ExecStart=/opt/palmgate/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

`deploy/orangepi/palmgate-device.service`

```ini
[Unit]
Description=PalmGate Device Runtime
After=network.target

[Service]
WorkingDirectory=/opt/palmgate
Environment=DEVICE_RUNTIME_ENABLED=1
Environment=CAMERA_SOURCE=usb
Environment=MODEL_PATH=/opt/palmgate/palm_embedding.tflite
Environment=MODEL_METADATA_PATH=/opt/palmgate/model_metadata.json
ExecStart=/opt/palmgate/.venv/bin/python -m app.device_runtime
Restart=always

[Install]
WantedBy=multi-user.target
```

Adjust interpreter paths for the actual Orange Pi install path during execution.

**Step 3: Update README with the operator workflow**

Add sections for:
- iPhone test mode over Wi-Fi
- USB camera mode
- required model files in the repo root
- how to check `/api/status`
- how to restart services

**Step 4: Verify docs and service files**

Run: `pytest tests/ -v`
Expected: PASS

Then verify the README instructions against the actual repo structure before finishing.

**Step 5: Commit**

```bash
git add deploy/orangepi/palmgate-api.service deploy/orangepi/palmgate-device.service README.md
git commit -m "docs: add orangepi service deployment guide"
```

---

## End-to-End Verification

### Local/Desktop verification

Run:
- `pytest tests/test_config.py tests/test_database.py tests/test_database_device_status.py tests/test_recognition_service.py tests/test_device_runtime.py tests/test_status_routes.py tests/test_palm_processor.py -v`
- `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Verify:
- `/api/status` returns app/database/device keys
- browser scan still works on desktop
- registration still stores 5-capture users
- log tab still paginates correctly

### Orange Pi + iPhone verification

Verify on the Orange Pi:
- model files exist in the project root
- app binds to `0.0.0.0`
- iPhone on the same Wi-Fi can open the dashboard
- browser scan/register can still call `/api/recognize` and `/api/register`
- status card shows worker disabled when USB runtime is off

### Orange Pi + USB camera verification

Verify:
- `/api/status` reports `worker_state=running`
- camera can disconnect and reconnect without crashing the API
- a steady hand triggers one recognition, then cooldown blocks duplicates
- each device recognition writes to `access_logs`
- admin dashboard reflects recent recognitions and current worker state

## Risks To Watch

- `mediapipe` on Orange Pi ARM can still be the hardest dependency even after moving off TensorFlow.
- The current preprocessing path may perform differently between iPhone browser captures and USB camera captures; validate cross-device recognition early.
- SQLite is fine for this scope, but do not write status every frame; heartbeat only once per second or slower.
- Keep the browser scan route for testing even after USB mode works; it is the fastest way to debug the model remotely.

## Definition of Done

This feature is done when:
- the same repo supports iPhone browser testing and USB-camera device runtime
- the admin dashboard still supports register/users/logs
- `/api/status` exposes useful runtime health
- the USB camera worker can run continuously and log recognitions into SQLite
- Orange Pi deployment instructions are documented and reproducible
