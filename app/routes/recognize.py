import base64
import logging
import time
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import RECOGNITION_TTA_ENABLED, SIMILARITY_THRESHOLD
from app.services.recognition_service import match_embedding_and_log

log = logging.getLogger("palmgate")
router = APIRouter()


class RecognizeRequest(BaseModel):
    image: str
    is_roi: bool = False          # True when the browser has pre-cropped the palm ROI
    rotation_angle: float = 0.0   # Knuckle-line tilt (deg) from index-MCP→pinky-MCP vector


class RecognizeResponse(BaseModel):
    status: str
    name: str
    similarity: float
    closest_match: "str | None" = None


def decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@router.post("/api/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest):
    from app.main import palm_processor, db

    started_at = time.perf_counter()
    try:
        frame = decode_base64_image(req.image)
        log.debug("RECOGNIZE | decoded image  shape=%s  dtype=%s  payload_len=%d",
                  frame.shape, frame.dtype, len(req.image))
    except Exception as e:
        log.error("RECOGNIZE | image decode failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image data")

    if req.is_roi:
        log.debug("RECOGNIZE | using pre-cropped client ROI — skipping server detection")
        embedding = palm_processor.get_embedding_from_roi(
            frame,
            req.rotation_angle,
            tta_enabled=RECOGNITION_TTA_ENABLED,
        )
    else:
        embedding = palm_processor.get_embedding_from_notebook_frame(
            frame,
            tta_enabled=RECOGNITION_TTA_ENABLED,
        )

    if embedding is None:
        log.warning("RECOGNIZE | returning 422 — no hand detected in frame")
        raise HTTPException(status_code=422, detail="No hand detected")

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    result = match_embedding_and_log(palm_processor, db, embedding, SIMILARITY_THRESHOLD, duration_ms=duration_ms)
    return RecognizeResponse(**result)
