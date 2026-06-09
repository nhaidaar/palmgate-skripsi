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
