import importlib


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


def test_usb_preview_interval_defaults_to_realtime(monkeypatch):
    monkeypatch.delenv("DEVICE_PREVIEW_FRAME_INTERVAL_MS", raising=False)

    import app.config as config
    importlib.reload(config)

    assert config.DEVICE_PREVIEW_FRAME_INTERVAL_MS == 33


def test_notebook_rembg_env_override(monkeypatch):
    monkeypatch.setenv("NOTEBOOK_REMBG_ENABLED", "0")

    import app.config as config
    importlib.reload(config)

    assert config.NOTEBOOK_REMBG_ENABLED is False


def test_embedding_model_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_METADATA_PATH", raising=False)
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    import app.config as config
    importlib.reload(config)

    assert config.MODEL_PATH == config.BASE_DIR / "models" / "embedding" / "palm_embedding.tflite"
    assert config.MODEL_METADATA_PATH == config.BASE_DIR / "models" / "embedding" / "model_metadata.json"
    assert config.EMBEDDING_DIM == 128
    assert config.SIMILARITY_THRESHOLD == 0.745932400226593
    assert config.TTA_ROTATIONS == (0.0, -6.0, 6.0)
    assert config.ENROLLMENT_TTA_ENABLED is True
    assert config.RECOGNITION_TTA_ENABLED is False


def test_model_metadata_overrides_threshold_and_dim(tmp_path, monkeypatch):
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(
        '{"embedding_dim": 64, "operating_threshold": 0.8123, "tta_rotations": [0, -3, 3]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_METADATA_PATH", str(metadata))
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    import app.config as config
    importlib.reload(config)

    assert config.EMBEDDING_DIM == 64
    assert config.SIMILARITY_THRESHOLD == 0.8123
    assert config.TTA_ROTATIONS == (0.0, -3.0, 3.0)
