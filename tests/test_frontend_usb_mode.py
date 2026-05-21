from pathlib import Path


def test_frontend_checks_status_before_starting_browser_camera():
    source = Path("app/static/app.js").read_text()
    init_block = source[source.index("Init") :]

    assert "await loadStatus()" in init_block
    assert "if (!state.usbDeviceMode)" in init_block
    assert init_block.index("await loadStatus()") < init_block.index("startCamera()")


def test_frontend_tracks_usb_device_mode_from_status():
    source = Path("app/static/app.js").read_text()

    assert "usbDeviceMode" in source
    assert "data.app?.camera_source === 'usb'" in source


def test_frontend_shows_usb_preview_without_browser_camera():
    source = Path("app/static/app.js").read_text()
    init_block = source[source.index("Init") :]

    assert "startUsbPreview()" in init_block
    assert "document.createElement('img')" in source
    assert "preview.src = '/api/device-registration/preview.mjpg'" in source
    assert "setInterval(refreshPreview" not in source
    assert init_block.index("startUsbPreview()") < init_block.index("setAutoMode(false)")


def test_usb_registration_panel_has_camera_preview():
    html = Path("app/static/index.html").read_text()
    source = Path("app/static/app.js").read_text()

    assert "usbRegistrationCameraFrame" in html
    assert "usbRegistrationPreview" in html
    assert "usbRegistrationPreview" in source
    assert "usbRegistrationPreview.src = '/api/device-registration/preview.mjpg'" in source


def test_usb_quality_ui_distinguishes_required_and_guidance_items():
    source = Path("app/static/app.js").read_text()

    assert "const blockers = new Set(guidance.blockers || [])" in source
    assert "Required" in source
    assert "Guide" in source
    assert "Adjust" in source
