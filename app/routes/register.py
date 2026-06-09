import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import (
    DUPLICATE_THRESHOLD,
    REGISTRATION_CAPTURES_PER_HAND,
    REGISTRATION_HANDS,
    REGISTRATION_TOTAL_CAPTURES,
)
from app.routes.recognize import decode_base64_image

router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    images: list
    hands: list[str] = []
    is_roi: bool = False          # True when the browser pre-cropped all palm ROIs
    rotation_angle: float = 0.0   # Knuckle-line tilt (deg) from index-MCP→pinky-MCP vector


class RegisterResponse(BaseModel):
    success: bool
    user_id: int
    name: str


@router.post("/api/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    from app.main import palm_processor, db

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    required_detail = (
        f"Need exactly {REGISTRATION_CAPTURES_PER_HAND} left-hand and "
        f"{REGISTRATION_CAPTURES_PER_HAND} right-hand palm images"
    )
    hands = [hand.lower() for hand in req.hands]
    if len(req.images) != REGISTRATION_TOTAL_CAPTURES or len(hands) != len(req.images):
        raise HTTPException(status_code=400, detail=required_detail)
    if any(hand not in REGISTRATION_HANDS for hand in hands):
        raise HTTPException(status_code=400, detail=required_detail)
    if any(hands.count(hand) != REGISTRATION_CAPTURES_PER_HAND for hand in REGISTRATION_HANDS):
        raise HTTPException(status_code=400, detail=required_detail)

    embeddings = []
    for i, img_b64 in enumerate(req.images):
        try:
            frame = decode_base64_image(img_b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image at index {i}")

        if req.is_roi:
            emb = palm_processor.get_embedding_from_roi(frame, req.rotation_angle)
        else:
            emb = palm_processor.get_embedding(frame)

        if emb is None:
            raise HTTPException(
                status_code=422,
                detail=f"No hand detected in image {i + 1}",
            )
        embeddings.append(emb)

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

    stored = db.get_all_embeddings()
    for emb in embeddings:
        dupe = palm_processor.compute_similarity(emb, stored, DUPLICATE_THRESHOLD)
        if dupe["status"] == "ALLOWED":
            raise HTTPException(
                status_code=409,
                detail=f"This palm is already registered as '{dupe['name']}' "
                       f"(similarity {dupe['similarity'] * 100:.0f}%). "
                       "Use a different palm or remove the existing user first.",
            )

    user_id = db.add_user(
        req.name.strip(),
        avg_embedding,
        individual_embeddings=embeddings,
        embedding_hands=hands,
    )
    return RegisterResponse(success=True, user_id=user_id, name=req.name.strip())
