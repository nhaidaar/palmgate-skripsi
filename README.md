# PALMGATE — Palmprint Recognition Preview

Web-based palmprint recognition app that now supports two operating modes:
- **Browser mode** — use a phone or desktop browser camera for testing and registration
- **Device mode** — run a 24x7 USB-camera recognition worker on an Orange Pi

## Requirements

- Python 3.10+
- `palm_embedding.tflite` in the project root
- `model_metadata.json` in the project root if available
- `hand_landmarker.task` in the project root

## Setup

```bash
pip install -r requirements.txt
```

If `hand_landmarker.task` is missing:

```python
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'hand_landmarker.task'
)
```

## Run locally

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Current features

- **Scan Palm** — browser-camera recognition with ALLOWED / DENIED result
- **Register** — USB-camera registration that captures 5 left-hand and 5 right-hand samples and stores per-hand templates
- **Access Log** — timestamped history of recognition attempts
- **Device Status** — shows worker state, camera state, FPS, registration state, and last recognition
- **USB Runtime** — optional always-on worker for Orange Pi with a USB camera

## Where to see recognition status

### 1. Admin dashboard
Use the **Access Log** tab to see:
- timestamp
- matched name
- `ALLOWED` / `DENIED`
- similarity score

### 2. Running shell / service logs
Recognition events are also printed to the running shell from the shared recognition path.
Examples:

```text
ALLOWED | user=Naufal | similarity=0.9100
DENIED | user=Unknown | similarity=0.4200
```

If you run with `systemd`, you can inspect these with:

```bash
journalctl -u palmgate-api -f
journalctl -u palmgate-device -f
```

## `/api/status`

The app exposes runtime status at:

```text
/api/status
```

Example response:

```json
{
  "app": {"mode": "hybrid", "version": "local"},
  "database": {"path": "/opt/palmgate/palmprint.db"},
  "device": {
    "worker_state": "disabled",
    "camera_connected": 0,
    "last_error": null,
    "fps": null,
    "last_inference_ms": null,
    "last_recognition_at": null
  }
}
```

## Official USB registration workflow

Production registration uses the Orange Pi USB camera, not the browser camera. The app captures 5 left-hand and 5 right-hand samples and stores one template per hand.

Sample sequence for each hand:
1. Center palm
2. Move closer
3. Move farther
4. Rotate left
5. Rotate right

Browser registration is kept only for local testing and may be less reliable on the USB device.

## Seed initial users from photos

You can bootstrap temporary users from named full-hand images in `seeds/`:

```bash
python scripts/seed_users.py seeds
```

Each image or folder label must use `nim_name` format, for example `12345_Naufal.jpg`. The script uses the same MediaPipe embedding path as runtime and stores a template for test seeding. Existing users are skipped by default and access logs are preserved. To replace registered users while keeping logs:

```bash
python scripts/seed_users.py seeds --replace-users
```

Seeded users are for initial testing only. Re-register users with the USB 7-sample flow before using the system for a real door lock.

## Orange Pi workflow

### Phase 1 — iPhone test mode over Wi-Fi
This is the easiest way to test before attaching a USB camera.

1. Run the API on the Orange Pi:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Put the Orange Pi and iPhone on the same Wi-Fi network
3. Open `http://<orange-pi-ip>:8000` on the iPhone
4. Use the phone camera for:
   - **Scan** tab testing
   - **Register** tab enrollment
5. Use **Log** and the new status card to monitor results

### Phase 2 — USB camera mode
When the USB camera is connected to the Orange Pi, run the device worker:

```bash
DEVICE_RUNTIME_ENABLED=1 CAMERA_SOURCE=usb CAMERA_DEVICE_INDEX=0 python -m app.device_runtime
```

The worker will:
- capture frames from the USB camera
- wait for the palm hold threshold
- run recognition
- write recognition attempts to the same SQLite database
- update `/api/status`

## Orange Pi systemd services

Service files are included in:
- `deploy/orangepi/palmgate-api.service`
- `deploy/orangepi/palmgate-device.service`

Expected install layout:
- project root: `/opt/palmgate`
- virtualenv python: `/opt/palmgate/.venv/bin/python`

### API service
Runs the FastAPI dashboard and browser-testing endpoints.

```bash
sudo cp deploy/orangepi/palmgate-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now palmgate-api
```

### Device service
Runs the USB camera worker continuously.

```bash
sudo cp deploy/orangepi/palmgate-device.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now palmgate-device
```

## Service management

```bash
sudo systemctl status palmgate-api
sudo systemctl status palmgate-device

sudo systemctl restart palmgate-api
sudo systemctl restart palmgate-device
```

Live logs:

```bash
journalctl -u palmgate-api -f
journalctl -u palmgate-device -f
```

## How recognition works

1. Frames use MediaPipe hand landmarks to crop the palm ROI.
2. The ROI is converted to grayscale, enhanced with CLAHE, converted back to RGB, resized to `224×224`, and kept as `0–255` float32 input.
3. `palm_embedding.tflite` outputs a 128-d L2-normalized embedding directly.
4. Cosine similarity compares the query embedding against stored per-hand templates.
5. Result is recorded as `ALLOWED` or `DENIED`.

## Cross-device / cross-brightness reliability

USB registration captures 5 samples per hand, averages each hand into a normalized template, and compares recognition queries against those templates. Combined with brightness normalization, this improves accuracy when enrollment and recognition cameras differ.

## Database migration notice

> **Breaking change** — the preprocessing pipeline was updated. Embeddings generated by an older version of PalmGate are **not compatible** with the current version.
>
> **Action required after upgrading:** delete `palmprint.db` and re-register users.
>
> ```bash
> rm palmprint.db
> ```

## Tests

```bash
python -m pytest tests/ -v
```
