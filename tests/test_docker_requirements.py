from pathlib import Path


def test_docker_requirements_include_notebook_preprocessing_dependencies():
    requirements = Path("requirements.docker.txt").read_text()

    assert "rembg" in requirements
    assert "onnxruntime" in requirements


def test_browser_compose_disables_rembg_for_sbc_cpu_runtime():
    compose = Path("docker-compose.yml").read_text()

    assert "NOTEBOOK_REMBG_ENABLED=0" in compose


def test_usb_compose_uses_logitech_camera_device():
    compose = Path("docker-compose.yml").read_text()

    assert "CAMERA_SOURCE=usb" in compose
    assert "CAMERA_DEVICE_INDEX=1" in compose
    assert "NOTEBOOK_REMBG_ENABLED=0" in compose
    assert "/dev/video1:/dev/video1" in compose


def test_usb_compose_uses_separate_preview_and_processing_intervals():
    compose = Path("docker-compose.yml").read_text()

    assert "DEVICE_PREVIEW_FRAME_INTERVAL_MS=100" in compose
    assert "DEVICE_FRAME_INTERVAL_MS=500" in compose
    assert "DEVICE_FRAME_INTERVAL_MS=1000" not in compose
