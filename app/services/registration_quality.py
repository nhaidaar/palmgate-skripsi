from dataclasses import dataclass

from app.config import (
    USB_REGISTRATION_MAX_BRIGHTNESS,
    USB_REGISTRATION_MIN_BLUR,
    USB_REGISTRATION_MIN_BRIGHTNESS,
)


@dataclass(frozen=True)
class SampleTarget:
    key: str
    label: str
    min_height_ratio: float
    max_height_ratio: float
    min_rotation: float
    max_rotation: float
    min_center_x: float
    max_center_x: float


@dataclass(frozen=True)
class GuidanceResult:
    acceptable: bool
    failures: list[str]
    blockers: list[str]
    score: float


SAMPLE_TARGETS = [
    SampleTarget("center", "Center palm", 0.50, 0.60, -5.0, 5.0, 0.40, 0.60),
    SampleTarget("closer", "Move closer", 0.65, 0.75, -5.0, 5.0, 0.40, 0.60),
    SampleTarget("farther", "Move farther", 0.38, 0.48, -5.0, 5.0, 0.40, 0.60),
    SampleTarget("rotate_left", "Rotate left", 0.50, 0.60, -15.0, -8.0, 0.40, 0.60),
    SampleTarget("rotate_right", "Rotate right", 0.50, 0.60, 8.0, 15.0, 0.40, 0.60),
]


def evaluate_guidance(sample_index: int, metrics: dict) -> GuidanceResult:
    target = SAMPLE_TARGETS[sample_index]
    failures = []
    blockers = []

    if not metrics.get("hand_detected", False):
        failures.append("hand")
        blockers.append("hand")
    if metrics.get("hand_clipped", True):
        failures.append("clipping")
    if not target.min_height_ratio <= metrics.get("height_ratio", 0.0) <= target.max_height_ratio:
        failures.append("size")
    if not target.min_rotation <= metrics.get("rotation_degrees", 999.0) <= target.max_rotation:
        failures.append("rotation")
    if not target.min_center_x <= metrics.get("center_x_ratio", 0.0) <= target.max_center_x:
        failures.append("position")
    if not USB_REGISTRATION_MIN_BRIGHTNESS <= metrics.get("brightness", 0.0) <= USB_REGISTRATION_MAX_BRIGHTNESS:
        failures.append("brightness")
        blockers.append("brightness")
    if metrics.get("blur_score", 0.0) < USB_REGISTRATION_MIN_BLUR:
        failures.append("sharpness")
        blockers.append("sharpness")
    if not metrics.get("steady", False):
        failures.append("steady")

    score = max(0.0, 1.0 - (len(failures) * 0.10) - (len(blockers) * 0.20))
    return GuidanceResult(acceptable=not blockers, failures=failures, blockers=blockers, score=score)
