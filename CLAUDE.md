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
DEVICE_RUNTIME_ENABLED=1 CAMERA_SOURCE=usb CAMERA_DEVICE_PATH=/dev/video0 python -m app.device_runtime

# Seed users from images
python scripts/seed_users.py seeds
python scripts/seed_users.py seeds --replace-users  # replace existing
```

## Architecture

### Processing Pipeline

The active runtime preprocessing path is **MediaPipe ROI** (`palm_processor.py:extract_palm_roi`): hand landmarks → palm ROI crop → grayscale → CLAHE → RGB → 224x224 resize. This must match `Palm Embedding.ipynb`.

`notebook_preprocessing.py` contains the old rembg/FFT ROI path for reference only; it is not the active registration or recognition path.

The TFLite model (`palm_embedding.tflite`) outputs a 128-dim L2-normalized embedding directly. Recognition uses cosine similarity against stored per-hand templates with the threshold from `model_metadata.json` or the default notebook operating threshold.

### Multi-Embedding Storage

USB registration captures 5 samples per hand, stores one normalized template per hand, and recognition matches against the best hand template.

### Key Components

- `app/main.py`: FastAPI entry point, lifespan management
- `app/device_runtime.py`: USB camera worker with hold-to-scan and registration state machine
- `app/palm_processor.py`: MediaPipe detection, CLAHE enhancement, TFLite inference
- `app/notebook_preprocessing.py`: Legacy rembg/FFT ROI reference path (not active runtime)
- `app/services/registration_quality.py`: 5 guidance targets per hand (center, closer, farther, rotate, shift)
- `app/database.py`: SQLite with users, user_embeddings, access_logs, device_status tables

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEVICE_RUNTIME_ENABLED` | `0` | Enable USB camera worker |
| `CAMERA_SOURCE` | `browser` | `browser` or `usb` |
| `CAMERA_DEVICE_PATH` | `/dev/video0` | Camera device path (e.g., `/dev/video0`, `/dev/video1`) |
| `DB_PATH` | `palmprint.db` | SQLite database location |
| `MODEL_VERSION` | `embedding_new_roi_v2` | Use `models/<version>/model.tflite` and `models/<version>/model_metadata.json` |
| `MODEL_PATH` | `models/<MODEL_VERSION>/model.tflite` | Explicit model path; overrides `MODEL_VERSION` |
| `MODEL_METADATA_PATH` | `models/<MODEL_VERSION>/model_metadata.json` | Explicit metadata path; defaults to `models/<MODEL_VERSION>/model_metadata.json` |
| `NOTEBOOK_REMBG_ENABLED` | `1` | Legacy notebook preprocessing only; inactive runtime path |

## Required Model Files

Place model files at either:
- default versioned path: `models/embedding_new_roi_v2/model.tflite`
- custom versioned path with `MODEL_VERSION=<version>`: `models/<version>/model.tflite`
- optional metadata: `models/<version>/model_metadata.json` or `MODEL_METADATA_PATH`
- `hand_landmarker.task` in project root - MediaPipe hand detection model

## Database Migration

Embeddings are incompatible across preprocessing/model changes. After upgrading to `palm_embedding.tflite`, delete `palmprint.db` and re-register users.

## Orange Pi Deployment

Service files in `deploy/orangepi/`:
- `palmgate-api.service` - FastAPI dashboard
- `palmgate-device.service` - USB camera worker

Expected layout: `/opt/palmgate` with `.venv/bin/python`
