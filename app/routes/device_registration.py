from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/device-registration")


class StartRegistrationRequest(BaseModel):
    name: str


class StartRegistrationResponse(BaseModel):
    session_id: str
    name: str
    current_sample_index: int
    captured_count: int


def _runtime():
    from app.main import device_runtime

    if device_runtime is None:
        raise HTTPException(status_code=409, detail="USB device runtime is not enabled")
    return device_runtime


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
    }


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
