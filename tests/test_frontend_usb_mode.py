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


def test_registration_ui_requires_and_sends_nim():
    html = Path("app/static/index.html").read_text()
    source = Path("app/static/app.js").read_text()

    assert "id=\"userNim\"" in html
    assert "const userNim" in source
    assert "body: JSON.stringify({ nim, name })" in source
    assert "hasNim" in source


def test_registration_ui_has_camera_upload_mode_tabs():
    html = Path("app/static/index.html").read_text()

    assert "id=\"registrationModeTabs\"" in html
    assert "id=\"cameraRegistrationTab\"" in html
    assert "id=\"uploadRegistrationTab\"" in html
    assert "data-registration-mode=\"camera\"" in html
    assert "data-registration-mode=\"upload\"" in html
    assert "Camera capture" in html
    assert "Upload images" in html
    assert "id=\"cameraRegistrationPanel\"" in html
    assert "aria-labelledby=\"cameraRegistrationTab\"" in html
    assert "id=\"uploadRegistrationPanel\"" in html
    assert "aria-labelledby=\"uploadRegistrationTab\"" in html


def test_upload_registration_has_separate_left_right_pickers():
    html = Path("app/static/index.html").read_text()

    assert "id=\"uploadLeftFiles\"" in html
    assert "id=\"uploadRightFiles\"" in html
    assert "Left hand photos" in html
    assert "Right hand photos" in html
    assert "Select exactly 5 full-hand photos" in html
    assert "id=\"btnUploadRegister\"" in html
    assert "id=\"btnClearUploadFiles\"" in html


def test_upload_registration_sends_full_photo_payload():
    source = Path("app/static/app.js").read_text()

    assert "async function finalizeUploadRegistration()" in source
    assert "function fileToDataUrl(file)" in source
    assert "uploadLeftFiles.files.length === REGISTRATION_CAPTURES_PER_HAND" in source
    assert "uploadRightFiles.files.length === REGISTRATION_CAPTURES_PER_HAND" in source
    assert "images: [...leftImages, ...rightImages]" in source
    assert "hands: [...Array(REGISTRATION_CAPTURES_PER_HAND).fill('left'), ...Array(REGISTRATION_CAPTURES_PER_HAND).fill('right')]" in source
    assert "is_roi: false" in source


def test_upload_busy_disables_registration_controls():
    source = Path("app/static/app.js").read_text()
    mode_block = source[source.index("function setRegistrationMode") : source.index("function uploadFiles")]
    ui_block = source[source.index("function updateRegistrationUI") : source.index("function renderQualityList")]

    assert "if (state.registrationActive || state.uploadBusy) return;" in mode_block
    assert "const busy = state.uploadBusy;" in ui_block
    assert "tab.disabled = active || busy;" in ui_block
    assert "btnStartRegistration.disabled = active || busy || !hasNim || !hasName" in ui_block
    assert "btnCaptureSample.disabled = !active || busy || !(state.lastGuidance?.acceptable)" in ui_block
    assert "btnFinalizeRegistration.disabled = !active || busy || !isRegistrationComplete()" in ui_block
    assert "btnCancelRegistration.disabled = !active || busy" in ui_block


def test_browser_camera_sends_full_frames_for_server_roi():
    source = Path("app/static/app.js").read_text()
    scan_block = source[source.index("async function triggerScan") : source.index("function showScanning")]
    capture_block = source[source.index("function captureBrowserSample") : source.index("async function finalizeBrowserRegistration")]
    finalize_block = source[source.index("async function finalizeBrowserRegistration") : source.index("function updateRegistrationUI")]

    assert "extractClientROI" not in source
    assert "b64 = captureFrame(video);" in scan_block
    assert "body: JSON.stringify({ image: b64, is_roi: false })" in scan_block
    assert "b64 = captureFrame(videoReg);" in capture_block
    assert "is_roi: false" in finalize_block


def test_frontend_displays_nim_with_user_name():
    source = Path("app/static/app.js").read_text()

    assert "${esc(u.nim)}" in source
    assert "${esc(u.name)}" in source
