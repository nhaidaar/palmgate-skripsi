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


def test_frontend_streams_usb_preview_without_browser_camera():
    source = Path("app/static/app.js").read_text()
    init_block = source[source.index("Init") :]

    assert "startUsbPreview()" in init_block
    assert "document.createElement('img')" in source
    assert "const USB_PREVIEW_STREAM_URL = '/api/device-registration/preview.mjpg'" in source
    assert "/api/device-registration/preview.jpg?t=" not in source
    assert "URL.createObjectURL" not in source
    assert "setInterval(updatePreview" not in source
    assert init_block.index("startUsbPreview()") < init_block.index("setAutoMode(false)")


def test_usb_registration_panel_has_camera_preview():
    html = Path("app/static/index.html").read_text()
    source = Path("app/static/app.js").read_text()

    assert "regCameraFrame" in html
    assert "usbRegistrationPreview" in html
    assert "usbRegistrationPreview" in source
    assert "syncUsbPreviewTarget()" in source
    assert "setUsbPreviewStream(usbRegistrationPreview" in source


def test_frontend_keeps_only_active_usb_preview_stream_connected():
    source = Path("app/static/app.js").read_text()
    switch_tab_block = source[source.index("function switchTab") : source.index("btnMode.addEventListener")]

    assert "syncUsbPreviewTarget()" in source
    assert "state.currentTab === 'scan'" in source
    assert "state.currentTab === 'register'" in source
    assert "img.removeAttribute('src')" in source
    assert "syncUsbPreviewTarget();" in switch_tab_block


def test_usb_quality_ui_distinguishes_required_and_guidance_items():
    source = Path("app/static/app.js").read_text()

    assert "const blockers = new Set(guidance.blockers || [])" in source
    assert "Required" in source
    assert "Guide" in source
    assert "Adjust" in source


def test_registration_ui_uses_two_hand_flow_copy():
    html = Path("app/static/index.html").read_text()
    source = Path("app/static/app.js").read_text()

    assert "Follow 7 guided poses" not in html
    assert "Sample 1/7" not in html
    assert "0 / 7" not in html
    assert "5 left-hand and 5 right-hand" in html
    assert "REGISTRATION_CAPTURES_PER_HAND = 5" in source
    assert "currentSampleIndex % SAMPLE_TARGETS.length" in source


def test_browser_registration_sends_hand_labels():
    source = Path("app/static/app.js").read_text()

    assert "hands: state.capturedSamples.map((c) => c.hand)" in source
    assert "leftCount === REGISTRATION_CAPTURES_PER_HAND" in source
    assert "rightCount === REGISTRATION_CAPTURES_PER_HAND" in source
    assert "getCurrentRegistrationHand()" in source
