import binascii
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import (
    DUPLICATE_THRESHOLD,
    ENROLLMENT_TTA_ENABLED,
    REGISTRATION_CAPTURES_PER_HAND,
    REGISTRATION_HANDS,
    REGISTRATION_MIN_VALID_PER_HAND,
    REGISTRATION_TOTAL_CAPTURES,
)
from app.routes.recognize import decode_base64_image
from app.services.embedding_templates import build_hand_templates, overall_template

router = APIRouter()


class RegisterRequest(BaseModel):
    nim: str = ""
    name: str
    images: list
    hands: list[str] = []
    is_roi: bool = False          # True when the browser pre-cropped all palm ROIs
    rotation_angle: float = 0.0   # Kept for backward compatibility; new ROIs are already aligned


class RegisterResponse(BaseModel):
    success: bool
    user_id: int
    name: str


@router.post("/api/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    from app.main import palm_processor, db

    nim = req.nim.strip()
    if not nim:
        raise HTTPException(status_code=400, detail="NIM is required")
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

    session_id = str(uuid.uuid4())
    save_dir = os.path.join("data", "captures", f"{nim}_{session_id}")
    os.makedirs(save_dir, exist_ok=True)
    import cv2

    samples = []
    for i, img_b64 in enumerate(req.images):
        try:
            frame = decode_base64_image(img_b64)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image at index {i}") from exc

        cv2.imwrite(os.path.join(save_dir, f"{hands[i]}_{i}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if req.is_roi:
            emb = palm_processor.get_embedding_from_roi(
                frame,
                req.rotation_angle,
                tta_enabled=ENROLLMENT_TTA_ENABLED,
            )
        else:
            emb = palm_processor.get_embedding(frame, tta_enabled=ENROLLMENT_TTA_ENABLED)

        if emb is None:
            raise HTTPException(
                status_code=422,
                detail=f"No hand detected in image {i + 1}",
            )
        samples.append({"hand": hands[i], "embedding": emb})

    try:
        templates = build_hand_templates(
            samples,
            required_hands=REGISTRATION_HANDS,
            min_per_hand=REGISTRATION_MIN_VALID_PER_HAND,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    embedding_hands = list(templates.keys())
    template_embeddings = [templates[hand] for hand in embedding_hands]
    avg_embedding = overall_template(templates)

    stored = db.get_all_embeddings()
    for emb in template_embeddings:
        dupe = palm_processor.compute_similarity(emb, stored, DUPLICATE_THRESHOLD)
        if dupe["status"] == "ALLOWED":
            raise HTTPException(
                status_code=409,
                detail=f"This palm is already registered as '{dupe['name']}' "
                       f"(similarity {dupe['similarity'] * 100:.0f}%). "
                       "Use a different palm or remove the existing user first.",
            )

    try:
        user_id = db.add_user(
            req.name.strip(),
            avg_embedding,
            nim=nim,
            individual_embeddings=template_embeddings,
            embedding_hands=embedding_hands,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return RegisterResponse(success=True, user_id=user_id, name=req.name.strip())
