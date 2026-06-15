from pathlib import Path


def test_docker_requirements_use_active_runtime_dependencies():
    requirements = Path("requirements.docker.txt").read_text()

    assert "mediapipe" in requirements
    assert "opencv-python-headless" in requirements
    assert "tflite-runtime" in requirements
    assert "rembg" not in requirements
    assert "onnxruntime" not in requirements


def test_compose_does_not_configure_old_notebook_rembg_path():
    compose = Path("docker-compose.yml").read_text()

    assert "NOTEBOOK_REMBG" not in compose


def test_usb_compose_uses_configurable_camera_device_path():
    compose = Path("docker-compose.yml").read_text()

    assert "CAMERA_SOURCE=usb" in compose
    assert "CAMERA_DEVICE_PATH=/dev/video1" in compose
    assert "${PALMGATE_CAMERA_DEVICE:-/dev/video0}:/dev/video1" in compose
    assert "usb-046d_C270_HD_WEBCAM" not in compose


def test_usb_compose_uses_separate_preview_and_processing_intervals():
    compose = Path("docker-compose.yml").read_text()

    assert "DEVICE_PREVIEW_FRAME_INTERVAL_MS=33" in compose
    assert "DEVICE_FRAME_INTERVAL_MS=250" in compose
    assert "DEVICE_FRAME_INTERVAL_MS=1000" not in compose


def test_compose_mounts_embedding_model():
    compose = Path("docker-compose.yml").read_text()

    assert "palm_embedding.tflite" in compose
    assert "palm_recognition.tflite:/app/palm_recognition.tflite" not in compose
