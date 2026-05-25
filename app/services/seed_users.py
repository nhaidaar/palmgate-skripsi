from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import SIMILARITY_THRESHOLD, USB_REGISTRATION_STORE_EMBEDDINGS
from app.services.registration_ranking import RegistrationSample, rank_registration_samples

SEED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ROTATION_DEGREES = 15.0  # Only accept images with rotation within ±15°


@dataclass(frozen=True)
class SeedVariant:
    name: str
    roi: np.ndarray


@dataclass(frozen=True)
class SeedEmbeddingResult:
    embedding: np.ndarray
    individual_embeddings: list[np.ndarray]
    variant_count: int
    selected_count: int


@dataclass(frozen=True)
class SeedUsersSummary:
    created: list[str]
    skipped: list[str]
    failed: dict[str, str]


def _rotate(roi: np.ndarray, angle_degrees: float) -> np.ndarray:
    h, w = roi.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_degrees, 1.0)
    return cv2.warpAffine(roi, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _zoom(roi: np.ndarray, scale: float) -> np.ndarray:
    h, w = roi.shape[:2]
    resized = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]

    if scale >= 1.0:
        y1 = max(0, (rh - h) // 2)
        x1 = max(0, (rw - w) // 2)
        return resized[y1:y1 + h, x1:x1 + w]

    result = np.zeros_like(roi)
    y1 = (h - rh) // 2
    x1 = (w - rw) // 2
    result[y1:y1 + rh, x1:x1 + rw] = resized
    return result


def _shift(roi: np.ndarray, x_ratio: float) -> np.ndarray:
    h, w = roi.shape[:2]
    matrix = np.float32([[1, 0, w * x_ratio], [0, 1, 0]])
    return cv2.warpAffine(roi, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def create_seed_variants(roi: np.ndarray) -> list[SeedVariant]:
    return [
        SeedVariant("original", roi.copy()),
        SeedVariant("rotate_left", _rotate(roi, -8.0)),
        SeedVariant("rotate_right", _rotate(roi, 8.0)),
        SeedVariant("zoom_in", _zoom(roi, 1.06)),
        SeedVariant("zoom_out", _zoom(roi, 0.94)),
        SeedVariant("shift_left", _shift(roi, -0.05)),
        SeedVariant("shift_right", _shift(roi, 0.05)),
    ]


def _select_seed_embeddings(samples: list[RegistrationSample], source_name: str) -> SeedEmbeddingResult:
    selected = rank_registration_samples(
        samples,
        keep=USB_REGISTRATION_STORE_EMBEDDINGS,
        min_similarity=SIMILARITY_THRESHOLD,
    )
    if len(selected) < USB_REGISTRATION_STORE_EMBEDDINGS:
        raise RuntimeError(f"Not enough consistent seed {source_name}")

    embeddings = [sample.embedding.astype(np.float32) for sample in selected]
    average = np.mean(embeddings, axis=0).astype(np.float32)
    return SeedEmbeddingResult(
        embedding=average,
        individual_embeddings=embeddings,
        variant_count=len(samples),
        selected_count=len(selected),
    )


def build_seed_embedding(frame_rgb: np.ndarray, palm_processor, preprocessor) -> SeedEmbeddingResult:
    extracted = preprocessor.extract_full_hand_roi(frame_rgb)
    if extracted is None:
        raise RuntimeError("Notebook preprocessing failed")

    variants = create_seed_variants(extracted.roi)
    samples = []
    for index, variant in enumerate(variants):
        processed = preprocessor.preprocess_roi_to_model_input(variant.roi)
        embedding = palm_processor._run_inference(processed).astype(np.float32)
        samples.append(RegistrationSample(index, 1.0, embedding))

    result = _select_seed_embeddings(samples, "augmentations")
    return SeedEmbeddingResult(
        embedding=result.embedding,
        individual_embeddings=[result.embedding],
        variant_count=result.variant_count,
        selected_count=result.selected_count,
    )


def build_seed_embedding_from_frames(
    frames_rgb: list[np.ndarray],
    palm_processor,
    preprocessor,
    max_rotation: float = MAX_ROTATION_DEGREES,
) -> SeedEmbeddingResult:
    """Build embedding from multiple frames, filtering by rotation consistency.

    Only accepts frames where the detected palm rotation is within ±max_rotation degrees.
    This ensures consistent ROI extraction across enrollment images.
    """
    valid_extractions = []
    for index, frame_rgb in enumerate(frames_rgb):
        extracted = preprocessor.extract_full_hand_roi(frame_rgb)
        if extracted is None:
            continue
        if abs(extracted.rotation_degrees) > max_rotation:
            continue
        valid_extractions.append((index, extracted))

    if not valid_extractions:
        raise RuntimeError(
            f"No valid seed captures with rotation within ±{max_rotation}°. "
            "Ensure enrollment images have palm roughly vertical."
        )

    samples = []
    for index, extracted in valid_extractions:
        processed = preprocessor.preprocess_roi_to_model_input(extracted.roi)
        embedding = palm_processor._run_inference(processed).astype(np.float32)
        samples.append(RegistrationSample(index, 1.0, embedding))

    embeddings = [sample.embedding.astype(np.float32) for sample in samples]
    average = np.mean(embeddings, axis=0).astype(np.float32)
    return SeedEmbeddingResult(
        embedding=average,
        individual_embeddings=embeddings,
        variant_count=len(frames_rgb),
        selected_count=len(samples),
    )


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Image could not be read")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _seed_image_paths(seed_dir: Path) -> list[Path]:
    return sorted(
        path for path in seed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SEED_IMAGE_EXTENSIONS
    )


def _seed_person_dirs(seed_dir: Path) -> list[Path]:
    return sorted(
        path for path in seed_dir.iterdir()
        if path.is_dir() and _seed_image_paths(path)
    )


def _existing_user_names(db) -> set[str]:
    return {user["name"] for user in db.get_all_users()}


def _replace_users(db):
    for user in db.get_all_users():
        db.delete_user(user["id"])


def seed_users_from_directory(
    seed_dir: str | Path,
    db,
    palm_processor,
    preprocessor,
    *,
    replace_users: bool = False,
    read_image=read_image_rgb,
) -> SeedUsersSummary:
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        raise RuntimeError(f"Seed directory does not exist: {seed_dir}")

    if replace_users:
        _replace_users(db)

    existing_names = _existing_user_names(db)
    created = []
    skipped = []
    failed = {}

    person_dirs = _seed_person_dirs(seed_dir)
    if person_dirs:
        for person_dir in person_dirs:
            name = person_dir.name
            if name in existing_names:
                skipped.append(name)
                continue

            try:
                frames = [read_image(path) for path in _seed_image_paths(person_dir)]
                result = build_seed_embedding_from_frames(frames, palm_processor, preprocessor)
                db.add_user(name, result.embedding, individual_embeddings=result.individual_embeddings)
                created.append(name)
                existing_names.add(name)
            except Exception as exc:
                failed[name] = str(exc)

        return SeedUsersSummary(created=created, skipped=skipped, failed=failed)

    for path in _seed_image_paths(seed_dir):
        name = path.stem
        if name in existing_names:
            skipped.append(name)
            continue

        try:
            frame = read_image(path)
            result = build_seed_embedding(frame, palm_processor, preprocessor)
            db.add_user(name, result.embedding, individual_embeddings=result.individual_embeddings)
            created.append(name)
            existing_names.add(name)
        except Exception as exc:
            failed[name] = str(exc)

    return SeedUsersSummary(created=created, skipped=skipped, failed=failed)
