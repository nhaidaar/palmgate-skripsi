/* ================================================================
   Palm Access — Biometric identification
   Browser-side: MediaPipe hand detection + client ROI crop
================================================================ */

// MediaPipe imports - loaded dynamically to avoid blocking if CDN fails
let HandLandmarker, FilesetResolver, DrawingUtils;

// ── Timing constants ─────────────────────────────────────────────
const SCAN_HOLD_MS     = 800;    // hold steady before auto-scan triggers
const REG_HOLD_MS      = 1000;   // hold steady before auto-capture
const REG_COOLDOWN_MS  = 1500;   // gap between auto-captures
const SCAN_COOLDOWN_MS = 3000;   // cooldown after scan result
const USB_PREVIEW_STREAM_URL = '/api/device-registration/preview.mjpg';

const REGISTRATION_HANDS = ['left', 'right'];
const REGISTRATION_CAPTURES_PER_HAND = 5;
const REGISTRATION_TOTAL_CAPTURES = REGISTRATION_HANDS.length * REGISTRATION_CAPTURES_PER_HAND;
const REGISTRATION_POSES = [
  { key: 'center', label: 'Center palm', desc: 'Palm facing camera, fingers up, wrist visible.' },
  { key: 'closer', label: 'Move closer', desc: 'Move closer while keeping the full hand visible.' },
  { key: 'farther', label: 'Move farther', desc: 'Move farther while keeping palm lines visible.' },
  { key: 'rotate_left', label: 'Rotate slightly left', desc: 'Tilt the palm slightly left on screen.' },
  { key: 'rotate_right', label: 'Rotate slightly right', desc: 'Tilt the palm slightly right on screen.' },
];

// Ring circumference for r=42: 2π×42 ≈ 263.9
const RING_C = 2 * Math.PI * 42;

// MediaPipe landmark indices (must match server)
const WRIST = 0, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;

// ── State ────────────────────────────────────────────────────────
const state = {
  stream: null,
  currentTab: 'scan',
  autoMode: true,
  usbDeviceMode: false,
  usbScanEventSource: null,
  // Registration state
  registrationActive: false,
  registrationStatusTimer: null,
  capturedSamples: [],
  registrationCounts: { left: 0, right: 0 },
  currentSampleIndex: 0,
  lastGuidance: null,
  // Scan state
  handSeenMs: 0,
  lastFrameTs: null,
  lastLandmarks: null,
  scanBusy: false,
  scanCooldownUntil: 0,
  scanStats: { total: 0, allowed: 0, denied: 0, users: 0 },
};

// ── DOM refs ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const video            = $('video');
const videoReg         = $('videoReg');
const canvas           = $('canvas');
const overlayCanvas    = $('overlayCanvas');
const overlayCanvasReg = $('overlayCanvasReg');
const btnScan          = $('btnScan');
const btnMode          = $('btnMode');
const btnRefresh       = $('btnRefresh');
const userName         = $('userName');
const usbRegistrationPreview = $('usbRegistrationPreview');
// Unified registration buttons
const btnStartRegistration = $('btnStartRegistration');
const btnCaptureSample = $('btnCaptureSample');
const btnFinalizeRegistration = $('btnFinalizeRegistration');
const btnCancelRegistration = $('btnCancelRegistration');

// ── MediaPipe init ───────────────────────────────────────────────
let handLandmarker = null;
let drawUtils      = null;

async function initMediaPipe() {
  try {
    // Dynamic import to avoid blocking page load if CDN fails
    const mediapipe = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    HandLandmarker = mediapipe.HandLandmarker;
    FilesetResolver = mediapipe.FilesetResolver;
    DrawingUtils = mediapipe.DrawingUtils;

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    });

    drawUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
    console.log('[PalmAccess] MediaPipe HandLandmarker ready');
    const cameraStatus = $('cameraStatus');
    if (cameraStatus) cameraStatus.innerHTML = '<span class="cam-dot"></span>Ready';
    startDetectLoop();
  } catch (err) {
    console.warn('[PalmAccess] MediaPipe failed — falling back to manual mode', err);
    setAutoMode(false);
    const cameraStatus = $('cameraStatus');
    if (cameraStatus) cameraStatus.textContent = 'Manual mode';
  }
}

// ── Detection loop ───────────────────────────────────────────────
function startDetectLoop() {
  requestAnimationFrame(detectLoop);
}

function detectLoop(ts) {
  requestAnimationFrame(detectLoop);
  if (!handLandmarker) return;

  const activeVideo = state.currentTab === 'scan' ? video : videoReg;
  if (activeVideo.readyState < 2 || activeVideo.paused) return;

  const result = handLandmarker.detectForVideo(activeVideo, ts);

  const activeCanvas = state.currentTab === 'scan' ? overlayCanvas : overlayCanvasReg;
  syncCanvasSize(activeCanvas, activeVideo);
  drawLandmarks(result, activeCanvas);

  const idleCanvas = state.currentTab === 'scan' ? overlayCanvasReg : overlayCanvas;
  idleCanvas.getContext('2d').clearRect(0, 0, idleCanvas.width, idleCanvas.height);

  const handFound = result.landmarks && result.landmarks.length > 0;
  const dt = state.lastFrameTs != null ? ts - state.lastFrameTs : 0;
  state.lastFrameTs = ts;

  if (handFound) {
    state.lastLandmarks = result.landmarks;
    state.handSeenMs = Math.min(state.handSeenMs + dt, Math.max(SCAN_HOLD_MS, REG_HOLD_MS) + 50);
    setCameraHandState(true);
    updateRingProgress();
    runAutoLogic();
    // Show brightness quality indicator
    if (state.currentTab === 'scan') {
      updateBrightnessBadge(video, result.landmarks[0], 'brightnessBadge');
    } else {
      updateBrightnessBadge(videoReg, result.landmarks[0], 'brightnessBadgeReg');
      // Update hand guide overlay in real-time during browser registration
      if (state.registrationActive && !state.usbDeviceMode) {
        const metrics = computeBrowserMetrics();
        state.lastGuidance = evaluateBrowserGuidance(metrics);
        updateHandGuideOverlay(metrics);
        updateRegistrationUI();
      }
    }
  } else {
    state.handSeenMs = Math.max(0, state.handSeenMs - dt * 2.5);
    if (state.handSeenMs <= 0) {
      setCameraHandState(false);
      state.lastLandmarks = null;
      $('brightnessBadge').style.display    = 'none';
      $('brightnessBadgeReg').style.display = 'none';
      // Clear hand guide overlay when no hand detected
      if (state.currentTab === 'register' && state.registrationActive && !state.usbDeviceMode) {
        updateHandGuideOverlay(null);
      }
    }
    updateRingProgress();
  }
}

function syncCanvasSize(cvs, vid) {
  if (cvs.width !== vid.videoWidth || cvs.height !== vid.videoHeight) {
    cvs.width  = vid.videoWidth  || 640;
    cvs.height = vid.videoHeight || 480;
  }
}

function drawLandmarks(result, cvs) {
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);

  if (!result.landmarks || !result.landmarks.length) return;
  if (!HandLandmarker || !DrawingUtils) return;

  const du = new DrawingUtils(ctx);
  for (const lm of result.landmarks) {
    du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
      color: '#6b7cf9',
      lineWidth: 2,
    });
    du.drawLandmarks(lm, {
      color: '#6b7cf9',
      fillColor: '#6b7cf9',
      lineWidth: 1,
      radius: 2,
    });
  }
}

// ── Brightness feedback ───────────────────────────────────────────
// Reads the mean luminance of the palm ROI canvas and updates a small
// badge so users know whether lighting conditions are suitable.
function updateBrightnessBadge(videoEl, landmarks, badgeId) {
  const badge = $(badgeId);
  if (!badge) return;
  if (!landmarks) { badge.style.display = 'none'; return; }

  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;
  const wrist     = landmarks[WRIST];
  const indexMcp  = landmarks[INDEX_MCP];
  const middleMcp = landmarks[MIDDLE_MCP];
  const pinkyMcp  = landmarks[PINKY_MCP];

  const cx = Math.round(middleMcp.x * w);
  const cy = Math.round(((middleMcp.y + wrist.y) / 2) * h);
  const palmWidth = Math.abs(Math.round((indexMcp.x - pinkyMcp.x) * w));
  const roiSize = Math.max(Math.round(palmWidth * 1.5), 60);
  const half = Math.round(roiSize / 2);
  const x1 = Math.max(0, cx - half);
  const y1 = Math.max(0, cy - half);
  const cropW = Math.min(w, cx + half) - x1;
  const cropH = Math.min(h, cy + half) - y1;
  if (cropW <= 0 || cropH <= 0) { badge.style.display = 'none'; return; }

  // Sample into a tiny 32×32 canvas to keep this cheap
  const tmp = document.createElement('canvas');
  tmp.width = 32; tmp.height = 32;
  tmp.getContext('2d').drawImage(videoEl, x1, y1, cropW, cropH, 0, 0, 32, 32);
  const pixels = tmp.getContext('2d').getImageData(0, 0, 32, 32).data;

  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    // Rec.709 luminance weights
    sum += pixels[i] * 0.2126 + pixels[i+1] * 0.7152 + pixels[i+2] * 0.0722;
  }
  const mean = sum / (pixels.length / 4);

  let label, cls;
  if (mean < 55) {
    label = 'Too dark'; cls = 'bri-dark';
  } else if (mean > 200) {
    label = 'Too bright'; cls = 'bri-bright';
  } else {
    label = 'Good light'; cls = 'bri-good';
  }

  badge.textContent = label;
  badge.className = `brightness-badge ${cls}`;
  badge.style.display = 'block';
}

// ── Client-side ROI extraction ───────────────────────────────────
// Mirrors the server's extract_palm_roi() logic, using landmarks already
// computed by the browser's MediaPipe instance. Sends a small JPEG crop
// instead of a full-resolution PNG, eliminating server-side detection.
//
// Also mirrors the notebook's calculate_roi rotation step: the knuckle line
// (index-MCP → pinky-MCP) is rotated to horizontal before cropping so the
// crop matches the training data distribution.
//
// Returns { data: base64string, rotationAngle: degrees }
function extractClientROI(videoEl, landmarks) {
  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;

  const wrist     = landmarks[WRIST];
  const indexMcp  = landmarks[INDEX_MCP];
  const middleMcp = landmarks[MIDDLE_MCP];
  const pinkyMcp  = landmarks[PINKY_MCP];

  // Knuckle-line rotation angle (same logic as calculate_roi in the notebook)
  const dx = (pinkyMcp.x - indexMcp.x) * w;
  const dy = (pinkyMcp.y - indexMcp.y) * h;
  const rotationAngle = Math.atan2(dy, dx) * (180 / Math.PI);

  // Rotate the video frame to align the knuckle line to horizontal
  const knuckleCx = (indexMcp.x + pinkyMcp.x) / 2 * w;
  const knuckleCy = (indexMcp.y + pinkyMcp.y) / 2 * h;
  const rad = rotationAngle * (Math.PI / 180);
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  // Rotate a point around the knuckle midpoint
  function rotPt(px, py) {
    const rx = cosA * (px - knuckleCx) + sinA * (py - knuckleCy) + knuckleCx;
    const ry = -sinA * (px - knuckleCx) + cosA * (py - knuckleCy) + knuckleCy;
    return [rx, ry];
  }

  const [midRx, midRy]   = rotPt(middleMcp.x * w, middleMcp.y * h);
  const [wristRx, wristRy] = rotPt(wrist.x * w, wrist.y * h);
  const [idxRx]           = rotPt(indexMcp.x * w, indexMcp.y * h);
  const [pnkRx]           = rotPt(pinkyMcp.x * w, pinkyMcp.y * h);

  const cx = Math.round(midRx);
  const cy = Math.round((midRy + wristRy) / 2);
  const palmWidth = Math.abs(Math.round(idxRx - pnkRx));
  const roiSize = Math.max(Math.round(palmWidth * 1.5), 60);
  const half = Math.round(roiSize / 2);

  const x1 = Math.max(0, cx - half);
  const y1 = Math.max(0, cy - half);
  const cropW = Math.min(w, cx + half) - x1;
  const cropH = Math.min(h, cy + half) - y1;

  // Draw the rotated video frame into an intermediate canvas, then crop
  const rotCanvas = document.createElement('canvas');
  rotCanvas.width  = w;
  rotCanvas.height = h;
  const rctx = rotCanvas.getContext('2d');
  rctx.save();
  rctx.translate(knuckleCx, knuckleCy);
  rctx.rotate(-rad);
  rctx.translate(-knuckleCx, -knuckleCy);
  rctx.drawImage(videoEl, 0, 0, w, h);
  rctx.restore();

  const roiCanvas = document.createElement('canvas');
  roiCanvas.width  = cropW;
  roiCanvas.height = cropH;
  roiCanvas.getContext('2d').drawImage(rotCanvas, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

  return { data: roiCanvas.toDataURL('image/jpeg', 0.9), rotationAngle };
}

// ── Ring progress ────────────────────────────────────────────────
function updateRingProgress() {
  const tab = state.currentTab;

  if (tab === 'scan') {
    const holdMs = SCAN_HOLD_MS;
    const ring   = $('autoscanRing');
    const fill   = $('ringFill');
    const label  = $('ringLabel');
    const pct    = Math.min(state.handSeenMs / holdMs, 1);

    if (state.handSeenMs > 20) {
      ring.style.display = 'block';
      fill.style.strokeDashoffset = RING_C * (1 - pct);
      const remaining = Math.ceil((holdMs - state.handSeenMs) / 1000);
      label.textContent = pct >= 1 ? '✓' : (remaining > 0 ? remaining + 's' : 'Hold');
    } else {
      ring.style.display = 'none';
    }
  }

  if (tab === 'register') {
    // Registration ring is handled by the registration status polling
    // Hide the ring when not in browser mode with active registration
    $('autoscanRingReg').style.display = 'none';
  }
}

// ── Auto-logic trigger ───────────────────────────────────────────
function runAutoLogic() {
  if (!state.autoMode) return;

  const now = Date.now();

  if (state.currentTab === 'scan') {
    if (state.handSeenMs >= SCAN_HOLD_MS && !state.scanBusy && now >= state.scanCooldownUntil) {
      triggerScan();
    }
  }

  // Registration auto-capture is handled by the registration status polling
}

// ── Camera hand-detected visual state ────────────────────────────
function setCameraHandState(detected) {
  const frame = state.currentTab === 'scan' ? $('cameraFrame') : $('regCameraFrame');
  if (!frame) return;
  frame.classList.toggle('hand-detected', detected);

  if (state.currentTab === 'scan') {
    $('palmGuide').style.opacity = detected ? '0' : '1';
  } else {
    $('palmGuideReg').style.opacity = detected ? '0' : '1';
  }
}

// ── Webcam ───────────────────────────────────────────────────────
async function startCamera() {
  if (state.stream) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject    = state.stream;
    videoReg.srcObject = state.stream;
    $('cameraStatus').innerHTML = '<span class="cam-dot"></span>Loading detector…';
  } catch (err) {
    $('cameraStatus').textContent = 'Camera error';
    console.error('Camera error:', err);
  }
}

function ensureUsbScanPreview() {
  video.style.display = 'none';
  let preview = $('usbPreview');
  if (!preview) {
    preview = document.createElement('img');
    preview.id = 'usbPreview';
    preview.className = 'usb-preview';
    $('cameraFrame').prepend(preview);
  }
  return preview;
}

function setUsbPreviewStream(img, active) {
  if (!img) return;
  if (active) {
    if (!img.src.includes(USB_PREVIEW_STREAM_URL)) {
      img.src = `${USB_PREVIEW_STREAM_URL}?t=${Date.now()}`;
    }
  } else {
    img.removeAttribute('src');
  }
}

function syncUsbPreviewTarget() {
  if (!state.usbDeviceMode) return;
  const scanPreview = ensureUsbScanPreview();
  setUsbPreviewStream(scanPreview, state.currentTab === 'scan');
  setUsbPreviewStream(usbRegistrationPreview, state.currentTab === 'register');
}

function startUsbPreview() {
  ensureUsbScanPreview();
  syncUsbPreviewTarget();
  $('cameraStatus').innerHTML = '<span class="cam-dot"></span>USB camera active';
}

function startUsbScanEvents() {
  if (state.usbScanEventSource) return;
  state.usbScanEventSource = new EventSource('/api/device-registration/scan-events');
  state.usbScanEventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.stage === 'recognized' && data.result) {
        showResult(data.result, null);
        updateStats(data.result.status);
      }
    } catch (err) {
      console.error('[PalmAccess] SSE parse error:', err);
    }
  };
  state.usbScanEventSource.onerror = () => {
    console.warn('[PalmAccess] SSE connection lost, reconnecting...');
  };
}

function captureFrame(videoEl) {
  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;
  canvas.width  = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(videoEl, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.9);
}

function triggerFlash(flashId) {
  const el = $(flashId);
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

// ── Tab navigation ───────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  state.currentTab = tab;
  state.handSeenMs = 0;

  document.querySelectorAll('.nav-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.tab === tab)
  );
  document.querySelectorAll('.panel').forEach((p) =>
    p.classList.toggle('active', p.id === `panel-${tab}`)
  );

  if (tab === 'log') {
    logPagState.page = 0;
    loadLogs();
    loadUsers();
  }

  if (state.usbDeviceMode) {
    syncUsbPreviewTarget();
  }
}

// ── Auto / Manual mode toggle ────────────────────────────────────
btnMode.addEventListener('click', () => setAutoMode(!state.autoMode));

function setAutoMode(on) {
  state.autoMode = on;
  btnMode.textContent = on ? 'Auto' : 'Manual';
  btnMode.classList.toggle('manual', !on);
  $('idleText').innerHTML = on
    ? 'Hold your open palm<br/>in front of the camera'
    : 'Press <strong>Scan now</strong><br/>to identify your palm';
  $('idleHint') && ($('idleHint').textContent = on ? 'Auto-detect on' : 'Manual mode');
}

// ── Scan Palm ────────────────────────────────────────────────────
btnScan.addEventListener('click', () => {
  if (!state.scanBusy) triggerScan();
});

async function triggerScan() {
  if (state.scanBusy) return;
  state.scanBusy = true;
  state.handSeenMs = 0;
  $('autoscanRing').style.display = 'none';

  triggerFlash('captureFlash');
  showScanning();

  // Use client-side ROI extraction if landmarks available (matches registration pipeline)
  let b64, isRoi = false, rotationAngle = 0;
  if (state.lastLandmarks && state.lastLandmarks.length > 0) {
    const roi = extractClientROI(video, state.lastLandmarks[0]);
    b64 = roi.data;
    rotationAngle = roi.rotationAngle;
    isRoi = true;
  } else {
    b64 = captureFrame(video);
  }

  const scanStart = performance.now();

  try {
    const res = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, is_roi: isRoi, rotation_angle: rotationAngle }),
    });
    const elapsed = Math.round(performance.now() - scanStart);

    if (res.status === 422) {
      showNoHand('No hand detected — adjust position and try again');
    } else if (!res.ok) {
      showNoHand('Server error — please try again');
    } else {
      const data = await res.json();
      showResult(data, elapsed);
      updateStats(data.status);
    }
  } catch (err) {
    showNoHand('Network error');
    console.error(err);
  }

  state.scanCooldownUntil = Date.now() + SCAN_COOLDOWN_MS;
  state.scanBusy = false;
}

function showScanning() {
  $('resultDisplay').style.display  = 'none';
  $('resultIdle').style.display     = 'none';
  $('resultScanning').style.display = 'flex';
  $('resultCard').className = 'result-card';
}

function showNoHand(msg) {
  $('resultScanning').style.display = 'none';
  $('resultDisplay').style.display  = 'none';
  $('resultIdle').style.display     = 'flex';
  $('resultIdle').querySelector('.idle-text').innerHTML =
    msg + '<br/><small style="opacity:.6;font-size:.85em">Adjust and try again</small>';
  $('resultCard').className = 'result-card';
}

function showResult(data, elapsedMs) {
  $('resultScanning').style.display = 'none';
  $('resultIdle').style.display     = 'none';
  $('resultDisplay').style.display  = 'flex';

  const ok = data.status === 'ALLOWED';
  $('resultCard').className = `result-card ${ok ? 'allowed' : 'denied'}`;
  $('badgeIcon').textContent   = ok ? '✓' : '✕';
  $('badgeStatus').textContent = ok ? 'Allowed' : 'Denied';
  $('badgeStatus').className   = `badge-status ${ok ? 'allowed' : 'denied'}`;
  $('resultName').textContent  = ok ? data.name : 'Unrecognized';
  $('resultName').className    = `result-name ${ok ? 'allowed' : 'denied'}`;
  $('resultSimilarity').textContent =
    data.similarity != null ? (data.similarity * 100).toFixed(1) + '%' : '—';

  const closestRow = $('closestRow');
  if (!ok && data.closest_match) {
    closestRow.style.display = 'flex';
    $('resultClosest').textContent =
      data.closest_match + ' (' + (data.similarity * 100).toFixed(1) + '%)';
  } else {
    closestRow.style.display = 'none';
  }

  $('resultTimestamp').textContent = new Date().toLocaleTimeString();

  const timingRow = $('timingRow');
  if (elapsedMs != null) {
    $('resultTiming').textContent = elapsedMs + ' ms';
    timingRow.style.display = 'flex';
  } else {
    timingRow.style.display = 'none';
  }
}

function updateStats(status) {
  state.scanStats.total++;
  if (status === 'ALLOWED') state.scanStats.allowed++;
  else state.scanStats.denied++;
  $('statTotal').textContent   = state.scanStats.total;
  $('statAllowed').textContent = state.scanStats.allowed;
  $('statDenied').textContent  = state.scanStats.denied;
}

// ── Unified Registration ─────────────────────────────────────────
// Sample targets matching server-side SAMPLE_TARGETS
const SAMPLE_TARGETS = [
  { key: 'center', label: 'Center palm', minHeight: 0.50, maxHeight: 0.60, minRot: -5, maxRot: 5, minCx: 0.40, maxCx: 0.60 },
  { key: 'closer', label: 'Move closer', minHeight: 0.65, maxHeight: 0.75, minRot: -5, maxRot: 5, minCx: 0.40, maxCx: 0.60 },
  { key: 'farther', label: 'Move farther', minHeight: 0.38, maxHeight: 0.48, minRot: -5, maxRot: 5, minCx: 0.40, maxCx: 0.60 },
  { key: 'rotate_left', label: 'Rotate left', minHeight: 0.50, maxHeight: 0.60, minRot: -15, maxRot: -8, minCx: 0.40, maxCx: 0.60 },
  { key: 'rotate_right', label: 'Rotate right', minHeight: 0.50, maxHeight: 0.60, minRot: 8, maxRot: 15, minCx: 0.40, maxCx: 0.60 },
];

function getCurrentRegistrationHand() {
  const index = Math.min(state.currentSampleIndex, REGISTRATION_TOTAL_CAPTURES - 1);
  return REGISTRATION_HANDS[Math.floor(index / REGISTRATION_CAPTURES_PER_HAND)];
}

function getCurrentPoseIndex() {
  const currentSampleIndex = Math.min(state.currentSampleIndex, REGISTRATION_TOTAL_CAPTURES - 1);
  return currentSampleIndex % SAMPLE_TARGETS.length;
}

function getRegistrationCounts() {
  if (state.usbDeviceMode) return state.registrationCounts;
  return {
    left: state.capturedSamples.filter((sample) => sample.hand === 'left').length,
    right: state.capturedSamples.filter((sample) => sample.hand === 'right').length,
  };
}

function isRegistrationComplete() {
  const { left: leftCount, right: rightCount } = getRegistrationCounts();
  return leftCount === REGISTRATION_CAPTURES_PER_HAND && rightCount === REGISTRATION_CAPTURES_PER_HAND;
}

function getCurrentSamplePrompt() {
  const hand = getCurrentRegistrationHand();
  const poseIndex = getCurrentPoseIndex();
  const pose = REGISTRATION_POSES[poseIndex];
  const handLabel = hand[0].toUpperCase() + hand.slice(1);
  return {
    title: `${handLabel} hand sample ${poseIndex + 1}/${REGISTRATION_CAPTURES_PER_HAND}: ${pose.label}`,
    desc: `Use your actual ${hand} hand. ${pose.desc} Keep the inside palm facing the camera.`,
  };
}

btnStartRegistration?.addEventListener('click', async () => {
  const name = userName.value.trim();
  if (!name) return setFeedback('Name is required', 'error');

  if (state.usbDeviceMode) {
    const result = await apiStartRegistration(name);
    if (result.detail) return setFeedback(result.detail, 'error');
  }

  state.registrationActive = true;
  state.capturedSamples = [];
  state.registrationCounts = { left: 0, right: 0 };
  state.currentSampleIndex = 0;
  setFeedback('Registration started. Capture 5 left-hand and 5 right-hand samples.', 'success');
  startRegistrationStatusPolling();
  updateRegistrationUI();
});

btnCaptureSample?.addEventListener('click', async () => {
  if (!state.registrationActive) return;

  if (state.usbDeviceMode) {
    const result = await apiCaptureSample();
    if (result.detail) return setFeedback(result.detail, 'error');
    triggerFlash('captureFlashReg');
    setFeedback(`Captured sample ${result.sample_index + 1}.`, 'success');
    await refreshRegistrationStatus();
  } else {
    captureBrowserSample();
  }
});

btnFinalizeRegistration?.addEventListener('click', async () => {
  if (state.usbDeviceMode) {
    const result = await apiFinalizeRegistration();
    if (result.detail) return setFeedback(result.detail, 'error');
    setFeedback(`✓ ${result.name} enrolled successfully`, 'success');
  } else {
    await finalizeBrowserRegistration();
  }
  resetRegistration();
  await loadUsers();
  await loadStats();
});

btnCancelRegistration?.addEventListener('click', async () => {
  if (state.usbDeviceMode) {
    await apiCancelRegistration();
  }
  resetRegistration();
  setFeedback('Registration cancelled.', '');
});

// API calls for USB mode
async function apiStartRegistration(name) {
  const res = await fetch('/api/device-registration/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  return await res.json();
}

async function apiGetRegistrationStatus() {
  const res = await fetch('/api/device-registration/status');
  return await res.json();
}

async function apiCaptureSample() {
  const res = await fetch('/api/device-registration/capture', { method: 'POST' });
  return await res.json();
}

async function apiFinalizeRegistration() {
  const res = await fetch('/api/device-registration/finalize', { method: 'POST' });
  return await res.json();
}

async function apiCancelRegistration() {
  const res = await fetch('/api/device-registration/cancel', { method: 'POST' });
  return await res.json();
}

function startRegistrationStatusPolling() {
  stopRegistrationStatusPolling();
  state.registrationStatusTimer = setInterval(refreshRegistrationStatus, 500);
}

function stopRegistrationStatusPolling() {
  if (state.registrationStatusTimer) clearInterval(state.registrationStatusTimer);
  state.registrationStatusTimer = null;
}

async function refreshRegistrationStatus() {
  if (state.usbDeviceMode) {
    const status = await apiGetRegistrationStatus();
    state.currentSampleIndex = status.current_sample_index || 0;
    state.capturedSamples = new Array(status.captured_count || 0);
    state.registrationCounts = { left: status.left_count || 0, right: status.right_count || 0 };
    state.lastGuidance = status.guidance;
    updateRegistrationUI();
    updateHandGuideOverlay(status.guidance?.metrics);
  } else {
    const metrics = computeBrowserMetrics();
    state.lastGuidance = evaluateBrowserGuidance(metrics);
    updateRegistrationUI();
    updateHandGuideOverlay(metrics);
  }
}

// Browser mode: compute metrics from MediaPipe landmarks
function computeBrowserMetrics() {
  if (!state.lastLandmarks || !state.lastLandmarks.length) {
    return { hand_detected: false };
  }
  const lm = state.lastLandmarks[0];
  const w = videoReg.videoWidth || 640;
  const h = videoReg.videoHeight || 480;

  const wrist = lm[WRIST];
  const indexMcp = lm[INDEX_MCP];
  const middleMcp = lm[MIDDLE_MCP];
  const pinkyMcp = lm[PINKY_MCP];

  // Height ratio (hand size relative to frame)
  const minY = Math.min(...lm.map(p => p.y));
  const maxY = Math.max(...lm.map(p => p.y));
  const heightRatio = maxY - minY;

  // Rotation (knuckle line angle)
  const dx = (pinkyMcp.x - indexMcp.x) * w;
  const dy = (pinkyMcp.y - indexMcp.y) * h;
  const rotationDegrees = Math.atan2(dy, dx) * (180 / Math.PI);

  // Center X position
  const centerX = (indexMcp.x + pinkyMcp.x) / 2;

  // Check if hand is clipped (any landmark near edge)
  const margin = 0.05;
  const handClipped = lm.some(p => p.x < margin || p.x > 1 - margin || p.y < margin || p.y > 1 - margin);

  return {
    hand_detected: true,
    height_ratio: heightRatio,
    rotation_degrees: rotationDegrees,
    center_x_ratio: centerX,
    hand_clipped: handClipped,
    brightness: 128,
    blur_score: 100,
    steady: state.handSeenMs > 300,
  };
}

function evaluateBrowserGuidance(metrics) {
  const target = SAMPLE_TARGETS[getCurrentPoseIndex()];
  const failures = [];
  const blockers = [];

  if (!metrics.hand_detected) {
    failures.push('hand');
    blockers.push('hand');
  } else {
    if (metrics.hand_clipped) failures.push('clipping');
    if (!(target.minHeight <= metrics.height_ratio && metrics.height_ratio <= target.maxHeight)) failures.push('size');
    if (!(target.minRot <= metrics.rotation_degrees && metrics.rotation_degrees <= target.maxRot)) failures.push('rotation');
    if (!(target.minCx <= metrics.center_x_ratio && metrics.center_x_ratio <= target.maxCx)) failures.push('position');
    if (!metrics.steady) failures.push('steady');
  }

  return {
    acceptable: blockers.length === 0 && failures.length <= 2,
    failures,
    blockers,
    target: target.key,
    label: target.label,
  };
}

// ── Browser registration fallback ─────────────────────────────────
function captureBrowserSample() {
  if (state.capturedSamples.length >= REGISTRATION_TOTAL_CAPTURES) return;
  if (!state.lastGuidance?.acceptable) {
    setFeedback('Adjust hand position before capturing', 'error');
    return;
  }

  const hand = getCurrentRegistrationHand();
  let b64, rotationAngle = 0;
  if (state.lastLandmarks && state.lastLandmarks.length > 0) {
    const roi = extractClientROI(videoReg, state.lastLandmarks[0]);
    b64 = roi.data;
    rotationAngle = roi.rotationAngle;
  } else {
    b64 = captureFrame(videoReg);
  }

  triggerFlash('captureFlashReg');
  state.capturedSamples.push({ data: b64, rotationAngle, hand });
  state.currentSampleIndex = state.capturedSamples.length;
  const counts = getRegistrationCounts();
  state.registrationCounts = counts;
  setFeedback(`Captured ${hand} hand sample ${counts[hand]}/${REGISTRATION_CAPTURES_PER_HAND}.`, 'success');
  updateRegistrationUI();
}

async function finalizeBrowserRegistration() {
  const name = userName.value.trim();
  const { left: leftCount, right: rightCount } = getRegistrationCounts();
  if (!name || !(leftCount === REGISTRATION_CAPTURES_PER_HAND && rightCount === REGISTRATION_CAPTURES_PER_HAND)) return;

  setFeedback('Registering…', '');
  const avgRotation = state.capturedSamples.reduce((s, c) => s + (c.rotationAngle || 0), 0) / state.capturedSamples.length;

  try {
    const res = await fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        images: state.capturedSamples.map((c) => c.data),
        hands: state.capturedSamples.map((c) => c.hand),
        is_roi: true,
        rotation_angle: avgRotation,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Registration failed');
    }
    setFeedback(`✓ ${data.name} enrolled successfully`, 'success');
  } catch (err) {
    setFeedback(err.message + ' — please try again.', 'error');
    throw err;
  }
}

function updateRegistrationUI() {
  const sample = getCurrentSamplePrompt();
  const { left: leftCount, right: rightCount } = getRegistrationCounts();

  const sampleTitle = $('regSampleTitle');
  const sampleDesc = $('regSampleDesc');
  const captureCounter = $('captureCounter');

  if (sampleTitle) sampleTitle.textContent = sample.title;
  if (sampleDesc) sampleDesc.textContent = sample.desc;
  if (captureCounter) {
    captureCounter.textContent = `Left ${leftCount}/${REGISTRATION_CAPTURES_PER_HAND} · Right ${rightCount}/${REGISTRATION_CAPTURES_PER_HAND}`;
  }

  document.querySelectorAll('#captureDots .dot').forEach((dot) => {
    const hand = dot.dataset.hand;
    const i = Number(dot.dataset.i || 0);
    dot.classList.toggle('filled', i < (hand === 'left' ? leftCount : rightCount));
  });

  renderQualityList(state.lastGuidance);

  const active = state.registrationActive;
  const hasName = userName?.value?.trim()?.length > 0;
  if (btnStartRegistration) btnStartRegistration.disabled = active || !hasName;
  if (btnCaptureSample) btnCaptureSample.disabled = !active || !(state.lastGuidance?.acceptable);
  if (btnFinalizeRegistration) btnFinalizeRegistration.disabled = !active || !isRegistrationComplete();
  if (btnCancelRegistration) btnCancelRegistration.disabled = !active;
}

function renderQualityList(guidance) {
  const list = $('qualityList');
  if (!list) return;

  if (!state.registrationActive) {
    list.innerHTML = '<li><span>Status</span><strong class="neutral">Enter name to start</strong></li>';
    return;
  }
  if (!guidance) {
    list.innerHTML = '<li><span>Status</span><strong class="bad">Waiting for hand</strong></li>';
    return;
  }
  const failures = new Set(guidance.failures || []);
  const blockers = new Set(guidance.blockers || []);
  const rows = [
    ['hand', 'Hand detected', 'Required'],
    ['brightness', 'Lighting', 'Required'],
    ['sharpness', 'Sharpness', 'Required'],
    ['clipping', 'Full hand visible', 'Guide'],
    ['size', 'Target size', 'Guide'],
    ['rotation', 'Target rotation', 'Guide'],
    ['position', 'Target position', 'Guide'],
    ['steady', 'Steady frame', 'Guide'],
  ];
  list.innerHTML = rows.map(([key, label, type]) => {
    const ok = !failures.has(key);
    const blocking = blockers.has(key);
    const status = ok ? 'OK' : (blocking ? 'Fix' : 'Adjust');
    const cls = ok ? 'ok' : (blocking ? 'bad' : 'warn');
    let detail = '';
    if (key === 'sharpness' && guidance.metrics?.blur_score != null) {
      detail = ` (${guidance.metrics.blur_score.toFixed(0)})`;
    } else if (key === 'brightness' && guidance.metrics?.brightness != null) {
      detail = ` (${guidance.metrics.brightness.toFixed(0)})`;
    }
    return `<li><span>${label}${detail} <em>${type}</em></span><strong class="${cls}">${status}</strong></li>`;
  }).join('');
}

function resetRegistration() {
  state.registrationActive = false;
  state.capturedSamples = [];
  state.registrationCounts = { left: 0, right: 0 };
  state.currentSampleIndex = 0;
  state.lastGuidance = null;
  stopRegistrationStatusPolling();
  userName.value = '';
  updateRegistrationUI();
  updateHandGuideOverlay(null);
}

function setFeedback(msg, type) {
  const el = $('registerFeedback');
  if (!el) return;
  el.textContent = msg;
  el.className = `register-feedback ${type}`;
}

// Hand guide overlay
function updateHandGuideOverlay(metrics) {
  const overlay = $('handGuideOverlay');
  const sizeRing = $('guideSizeRing');
  const crossH = $('guideCrossH');
  const crossV = $('guideCrossV');
  const rotArc = $('guideRotationArc');
  const palmGuide = $('palmGuideReg');
  const metricsDisplay = $('guidanceMetrics');

  // Early return if elements don't exist
  if (!overlay || !sizeRing) return;

  if (!metrics || !metrics.hand_detected) {
    overlay.style.opacity = '0.3';
    sizeRing.setAttribute('stroke', 'var(--text-muted)');
    if (palmGuide) palmGuide.style.opacity = '1';
    if (metricsDisplay) {
      const metricSize = $('metricSize');
      const metricRotation = $('metricRotation');
      const metricPosition = $('metricPosition');
      if (metricSize) metricSize.querySelector('strong').textContent = '--';
      if (metricRotation) metricRotation.querySelector('strong').textContent = '--';
      if (metricPosition) metricPosition.querySelector('strong').textContent = '--';
    }
    return;
  }

  overlay.style.opacity = '1';
  if (palmGuide) palmGuide.style.opacity = '0';

  const target = SAMPLE_TARGETS[Math.min(state.currentSampleIndex, SAMPLE_TARGETS.length - 1)];

  // Size indicator
  const sizeOk = target.minHeight <= metrics.height_ratio && metrics.height_ratio <= target.maxHeight;
  const sizeColor = sizeOk ? 'var(--success)' : 'var(--warning)';
  sizeRing.setAttribute('stroke', sizeColor);
  const targetSize = (target.minHeight + target.maxHeight) / 2;
  const sizeRadius = 20 + targetSize * 40;
  sizeRing.setAttribute('r', sizeRadius.toFixed(1));

  // Position indicator (crosshair)
  if (crossH && crossV) {
    const posOk = target.minCx <= metrics.center_x_ratio && metrics.center_x_ratio <= target.maxCx;
    const posColor = posOk ? 'var(--success)' : 'var(--warning)';
    const targetCx = (target.minCx + target.maxCx) / 2 * 100;
    crossH.setAttribute('x1', (targetCx - 5).toFixed(1));
    crossH.setAttribute('x2', (targetCx + 5).toFixed(1));
    crossH.setAttribute('y1', '50');
    crossH.setAttribute('y2', '50');
    crossV.setAttribute('x1', targetCx.toFixed(1));
    crossV.setAttribute('x2', targetCx.toFixed(1));
    crossH.setAttribute('stroke', posColor);
    crossV.setAttribute('stroke', posColor);
  }

  // Rotation indicator
  if (rotArc) {
    const rotOk = target.minRot <= metrics.rotation_degrees && metrics.rotation_degrees <= target.maxRot;
    const rotColor = rotOk ? 'var(--success)' : 'var(--warning)';
    const targetRot = (target.minRot + target.maxRot) / 2;
    const arcRadius = 45;
    const startAngle = (targetRot - 10) * Math.PI / 180;
    const endAngle = (targetRot + 10) * Math.PI / 180;
    const x1 = 50 + arcRadius * Math.cos(startAngle);
    const y1 = 50 + arcRadius * Math.sin(startAngle);
    const x2 = 50 + arcRadius * Math.cos(endAngle);
    const y2 = 50 + arcRadius * Math.sin(endAngle);
    rotArc.setAttribute('d', `M ${x1} ${y1} A ${arcRadius} ${arcRadius} 0 0 1 ${x2} ${y2}`);
    rotArc.setAttribute('stroke', rotColor);
  }

  // Update metrics display
  if (metricsDisplay) {
    const sizeOk = target.minHeight <= metrics.height_ratio && metrics.height_ratio <= target.maxHeight;
    const rotOk = target.minRot <= metrics.rotation_degrees && metrics.rotation_degrees <= target.maxRot;
    const posOk = target.minCx <= metrics.center_x_ratio && metrics.center_x_ratio <= target.maxCx;
    const sizePercent = (metrics.height_ratio * 100).toFixed(0);
    const rotDeg = metrics.rotation_degrees.toFixed(1);
    const posPercent = (metrics.center_x_ratio * 100).toFixed(0);
    const metricSize = $('metricSize');
    const metricRotation = $('metricRotation');
    const metricPosition = $('metricPosition');
    if (metricSize) {
      metricSize.querySelector('strong').textContent = `${sizePercent}%`;
      metricSize.querySelector('strong').className = sizeOk ? 'ok' : 'warn';
    }
    if (metricRotation) {
      metricRotation.querySelector('strong').textContent = `${rotDeg}°`;
      metricRotation.querySelector('strong').className = rotOk ? 'ok' : 'warn';
    }
    if (metricPosition) {
      metricPosition.querySelector('strong').textContent = `${posPercent}%`;
      metricPosition.querySelector('strong').className = posOk ? 'ok' : 'warn';
    }
  }
}

userName?.addEventListener('input', () => {
  const hasName = userName.value.trim().length > 0;
  if (btnStartRegistration) btnStartRegistration.disabled = state.registrationActive || !hasName;
});

// ── Access Log ───────────────────────────────────────────────────
// ── Access Log Pagination ─────────────────────────────────────────
const LOG_PAGE_SIZE = 10;
const logPagState = { page: 0, total: 0 };

btnRefresh?.addEventListener('click', () => {
  logPagState.page = 0;
  loadLogs();
  loadUsers();
});

$('btnLogPrev')?.addEventListener('click', () => {
  if (logPagState.page > 0) { logPagState.page--; loadLogs(); }
});

$('btnLogNext')?.addEventListener('click', () => {
  const totalPages = Math.ceil(logPagState.total / LOG_PAGE_SIZE);
  if (logPagState.page < totalPages - 1) { logPagState.page++; loadLogs(); }
});

async function loadLogs() {
  try {
    const [countRes, logs] = await Promise.all([
      fetch('/api/logs/count').then((r) => r.json()),
      fetch(`/api/logs?limit=${LOG_PAGE_SIZE}&offset=${logPagState.page * LOG_PAGE_SIZE}`).then((r) => r.json()),
    ]);
    logPagState.total = countRes.count ?? 0;
    renderLogs(logs);
    updateLogPagination();
  } catch (err) { console.error(err); }
}

function updateLogPagination() {
  const totalPages = Math.max(1, Math.ceil(logPagState.total / LOG_PAGE_SIZE));
  const pagInfo = $('pagInfo');
  if (pagInfo) pagInfo.textContent = `Page ${logPagState.page + 1} of ${totalPages}`;
  const prev = $('btnLogPrev');
  const next = $('btnLogNext');
  if (prev) prev.disabled = logPagState.page === 0;
  if (next) next.disabled = logPagState.page >= totalPages - 1;
}

function renderLogs(logs) {
  const tbody = $('logTableBody');
  if (!logs.length) {
    tbody.innerHTML = `<tr class="log-empty-row"><td colspan="6"><div class="log-empty">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
        <path d="M8 12h24M8 20h16M8 28h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg><span>No access attempts recorded yet</span></div></td></tr>`;
    return;
  }
  tbody.innerHTML = logs.map((log) => {
    const ts = new Date(log.timestamp);
    const ok = log.status === 'ALLOWED';
    return `<tr>
      <td>${isNaN(ts) ? log.timestamp : ts.toLocaleString()}</td>
      <td>${esc(log.matched_name)}</td>
      <td><span class="log-status ${ok ? 'allowed' : 'denied'}">${ok ? 'Allowed' : 'Denied'}</span></td>
      <td>${log.similarity != null ? (log.similarity * 100).toFixed(1) + '%' : '—'}</td>
      <td>${log.duration_ms != null ? log.duration_ms + ' ms' : '—'}</td>
      <td>${log.description ? esc(log.description) : '—'}</td>
    </tr>`;
  }).join('');
}

async function loadUsers() {
  try {
    const users = await fetch('/api/users').then((r) => r.json());
    renderUsers(users);
    state.scanStats.users = users.length;
    $('statUsers').textContent = users.length;
  } catch (err) { console.error(err); }
}

function renderUsers(users) {
  const grid = $('usersGrid');
  if (!users.length) {
    grid.innerHTML = '<div class="users-empty">No users enrolled yet.</div>';
    return;
  }
  grid.innerHTML = users.map((u) => `
    <div class="user-chip" id="chip-${u.id}">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="4.5" r="2.5" stroke="currentColor" stroke-width="1.2"/>
        <path d="M2 13c0-2.761 2.239-4 5-4s5 1.239 5 4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      <span class="user-chip-name">${esc(u.name)}</span>
      <button class="user-chip-delete" onclick="window.deleteUser(${u.id})" title="Remove">×</button>
    </div>`).join('');
}

window.deleteUser = async (id) => {
  if (!confirm('Remove this user?')) return;
  await fetch(`/api/users/${id}`, { method: 'DELETE' });
  loadUsers();
  loadLogs();
};

async function loadStats() {
  try {
    const users = await fetch('/api/users').then((r) => r.json());
    $('statUsers').textContent = users.length;
  } catch (_) { /* silent */ }
}

async function loadStatus() {
  try {
    const data = await fetch('/api/status').then((r) => r.json());
    const device = data.device || {};
    state.usbDeviceMode = data.app?.camera_source === 'usb' && data.app?.device_runtime_enabled === true;
    const workerState = device.worker_state ?? 'disabled';
    const cameraConnected = !!device.camera_connected;

    $('deviceWorkerState').textContent = workerState;
    $('deviceCameraState').textContent = cameraConnected ? 'connected' : 'offline';
    $('deviceFps').textContent = device.fps != null ? String(device.fps) : '—';
    $('deviceLastRecognition').textContent = device.last_recognition_at ?? '—';

    $('systemStatus').classList.toggle('offline', workerState !== 'running');
    $('systemStatusLabel').textContent = workerState === 'running' ? 'Online' : 'Idle';
  } catch (_) {
    $('deviceWorkerState').textContent = 'unreachable';
    $('deviceCameraState').textContent = 'offline';
    $('deviceFps').textContent = '—';
    $('deviceLastRecognition').textContent = '—';
    $('systemStatus').classList.add('offline');
    $('systemStatusLabel').textContent = 'Offline';
  }
}

const esc = (s) =>
  String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── Init ─────────────────────────────────────────────────────────
(async () => {
  try {
    loadStats();
    await loadStatus();
    setInterval(loadStatus, 5000);

    if (!state.usbDeviceMode) {
      // Browser mode: use webcam for both scan and registration
      await startCamera();
      video.addEventListener('loadeddata', () => initMediaPipe(), { once: true });
      if (video.readyState >= 2) initMediaPipe();
      // Show browser video in registration, hide USB preview
      if (videoReg) videoReg.style.display = 'block';
      if (usbRegistrationPreview) usbRegistrationPreview.style.display = 'none';
    } else {
      // USB mode: use MJPEG stream for both scan and registration
      startUsbPreview();
      startUsbScanEvents();
      setAutoMode(false);
      // Show USB preview in registration, hide browser video
      if (videoReg) videoReg.style.display = 'none';
      if (usbRegistrationPreview) {
        usbRegistrationPreview.style.display = 'block';
      }
    }

    // Initialize registration UI
    updateRegistrationUI();
  } catch (err) {
    console.error('[PalmAccess] Init error:', err);
  }
})();
