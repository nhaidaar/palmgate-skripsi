import json
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import REGISTRATION_CAPTURES_PER_HAND, REGISTRATION_HANDS, REGISTRATION_TOTAL_CAPTURES

router = APIRouter(prefix="/api/device-registration")


class StartRegistrationRequest(BaseModel):
    name: str


class StartRegistrationResponse(BaseModel):
    session_id: str
    name: str
    current_sample_index: int
    captured_count: int
    required_per_hand: int
    total_required: int
    current_hand: str
    left_count: int
    right_count: int


def _runtime():
    from app.main import device_runtime

    if device_runtime is None:
        raise HTTPException(status_code=409, detail="USB device runtime is not enabled")
    return device_runtime


def _hand_for_sample_index(index: int) -> str:
    hand_index = index // REGISTRATION_CAPTURES_PER_HAND
    return REGISTRATION_HANDS[min(hand_index, len(REGISTRATION_HANDS) - 1)]


def _registration_progress(session) -> dict:
    counts = {hand: 0 for hand in REGISTRATION_HANDS}
    for i, sample in enumerate(session.captured_samples):
        hand = sample.get("hand", _hand_for_sample_index(i))
        if hand in counts:
            counts[hand] += 1
    return {
        "required_per_hand": REGISTRATION_CAPTURES_PER_HAND,
        "total_required": REGISTRATION_TOTAL_CAPTURES,
        "current_hand": _hand_for_sample_index(session.current_sample_index),
        "left_count": counts.get("left", 0),
        "right_count": counts.get("right", 0),
    }


@router.post("/start", response_model=StartRegistrationResponse)
async def start_registration(req: StartRegistrationRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    try:
        session = _runtime().start_registration(req.name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return StartRegistrationResponse(
        session_id=session.id,
        name=session.name,
        current_sample_index=session.current_sample_index,
        captured_count=len(session.captured_samples),
        **_registration_progress(session),
    )


@router.get("/status")
async def registration_status():
    runtime = _runtime()
    session = runtime.registration_session
    if session is None:
        return {"active": False, "worker_state": runtime.worker_state}
    return {
        "active": True,
        "worker_state": runtime.worker_state,
        "session_id": session.id,
        "name": session.name,
        "current_sample_index": session.current_sample_index,
        "captured_count": len(session.captured_samples),
        "guidance": session.last_guidance,
        **_registration_progress(session),
    }


@router.get("/preview.jpg")
async def preview_frame():
    frame = _runtime().get_latest_frame_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail="USB preview frame is not ready")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def mjpeg_frames(runtime):
    while True:
        frame = runtime.get_latest_frame_jpeg()
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.1)


@router.get("/preview.mjpg")
async def preview_stream():
    return StreamingResponse(
        mjpeg_frames(_runtime()),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/capture")
async def capture_registration_sample():
    try:
        sample = _runtime().capture_registration_sample()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "sample_index": sample["sample_index"],
        "quality_score": sample["quality_score"],
    }


@router.post("/finalize")
async def finalize_registration():
    try:
        return _runtime().finalize_registration()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/cancel")
async def cancel_registration():
    _runtime().cancel_registration()
    return {"success": True}


def scan_event_stream(runtime):
    """SSE generator that yields scan events as they occur."""
    subscriber = runtime.scan_broadcaster.subscribe()
    try:
        while True:
            try:
                event = subscriber.get(timeout=30)
                data = json.dumps(event)
                yield f"data: {data}\n\n"
            except Exception:
                yield ": keepalive\n\n"
    finally:
        runtime.scan_broadcaster.unsubscribe(subscriber)


@router.get("/scan-events")
async def scan_events():
    """SSE endpoint for real-time scan result notifications."""
    return StreamingResponse(
        scan_event_stream(_runtime()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
