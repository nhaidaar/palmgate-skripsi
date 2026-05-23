# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PalmGate is a palmprint recognition system for smart door locks. It runs in two modes:
- **Browser mode**: Web UI with phone/desktop camera for testing and registration
- **Device mode**: 24/7 USB camera worker on Orange Pi for production use

## Commands

```bash
# Run development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Run tests
python -m pytest tests/ -v

# Run single test
python -m pytest tests/test_notebook_preprocessing.py -v

# Run device worker (Orange Pi with USB camera)
DEVICE_RUNTIME_ENABLED=1 CAMERA_SOURCE=usb CAMERA_DEVICE_INDEX=0 python -m app.device_runtime

# Seed users from images
python scripts/seed_users.py seeds
python scripts/seed_users.py seeds --replace-users  # replace existing
```

## Architecture

### Processing Pipeline

Two preprocessing paths exist:

1. **MediaPipe path** (`palm_processor.py:extract_palm_roi`): Real-time hand detection using landmarks for browser-based scanning. Faster but less accurate.

2. **Notebook path** (`notebook_preprocessing.py`): Background removal (rembg) → mask thresholding → contour extraction → FFT valley detection → palm ROI crop → CLAHE → 224x224 resize. Used for USB registration and all recognition embeddings.

The TFLite model (`palm_recognition.tflite`) extracts 1280-dim embeddings from the GlobalAveragePooling2D layer. Recognition uses cosine similarity against stored embeddings with a 0.75 threshold.

### Multi-Embedding Storage

USB registration captures 7 guided samples, stores the best 5 individual embeddings per user. Recognition matches against every stored capture (not just the averaged embedding) for better cross-device accuracy.

### Key Components

- `app/main.py`: FastAPI entry point, lifespan management
- `app/device_runtime.py`: USB camera worker with hold-to-scan and registration state machine
- `app/palm_processor.py`: MediaPipe detection, CLAHE enhancement, TFLite inference
- `app/notebook_preprocessing.py`: Production preprocessing (rembg + FFT valley detection)
- `app/services/registration_quality.py`: 7-sample guidance targets (center, closer, farther, rotate, shift)
- `app/database.py`: SQLite with users, user_embeddings, access_logs, device_status tables

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEVICE_RUNTIME_ENABLED` | `0` | Enable USB camera worker |
| `CAMERA_SOURCE` | `browser` | `browser` or `usb` |
| `CAMERA_DEVICE_INDEX` | `0` | USB camera index |
| `CAMERA_DEVICE_PATH` | - | Override camera path (e.g., `/dev/video0`) |
| `DB_PATH` | `palmprint.db` | SQLite database location |
| `NOTEBOOK_REMBG_ENABLED` | `1` | Enable background removal |

## Required Model Files

Place in project root:
- `palm_recognition.tflite` - EfficientNetB0 embedding model
- `hand_landmarker.task` - MediaPipe hand detection model

## Database Migration

Embeddings are incompatible across preprocessing pipeline changes. After upgrading, delete `palmprint.db` and re-register users.

## Orange Pi Deployment

Service files in `deploy/orangepi/`:
- `palmgate-api.service` - FastAPI dashboard
- `palmgate-device.service` - USB camera worker

Expected layout: `/opt/palmgate` with `.venv/bin/python`
