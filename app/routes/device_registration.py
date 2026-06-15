import asyncio
import json
import queue

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/device-registration")


class StartRegistrationRequest(BaseModel):
    nim: str = ""
    name: str


class StartRegistrationResponse(BaseModel):
    session_id: str
    nim: str
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


@router.post("/start", response_model=StartRegistrationResponse)
async def start_registration(req: StartRegistrationRequest):
    nim = req.nim.strip()
    name = req.name.strip()
    if not nim:
        raise HTTPException(status_code=400, detail="NIM is required")
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    runtime = _runtime()
    try:
        runtime.start_registration(nim, name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    status = runtime.get_registration_status()
    return StartRegistrationResponse(
        session_id=status["session_id"],
        nim=status["nim"],
        name=status["name"],
        current_sample_index=status["current_sample_index"],
        captured_count=status["captured_count"],
        required_per_hand=status["required_per_hand"],
        total_required=status["total_required"],
        current_hand=status["current_hand"],
        left_count=status["left_count"],
        right_count=status["right_count"],
    )


@router.get("/status")
async def registration_status():
    return _runtime().get_registration_status()


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


async def mjpeg_frames(runtime):
    while True:
        frame = runtime.get_latest_frame_jpeg()
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        interval_seconds = getattr(runtime, "preview_frame_interval_ms", 100) / 1000
        await asyncio.sleep(max(interval_seconds, 0.001))


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
    try:
        _runtime().cancel_registration()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True}


async def scan_event_stream(runtime):
    """SSE generator that yields scan events as they occur."""
    subscriber = runtime.scan_broadcaster.subscribe()
    try:
        while True:
            try:
                event = await asyncio.to_thread(subscriber.get, True, 30)
                data = json.dumps(event)
                yield f"data: {data}\n\n"
            except queue.Empty:
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
