import numpy as np


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)


def mean_template(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one embedding is required")
    normalized = [l2_normalize(embedding) for embedding in embeddings]
    return l2_normalize(np.mean(normalized, axis=0))


def build_hand_templates(
    samples: list[dict],
    *,
    required_hands: tuple,
    min_per_hand: int,
) -> dict[str, np.ndarray]:
    templates = {}
    for hand in required_hands:
        embeddings = [sample["embedding"] for sample in samples if sample.get("hand") == hand]
        if len(embeddings) < min_per_hand:
            raise ValueError(f"Not enough valid {hand} samples")
        templates[hand] = mean_template(embeddings)
    return templates


def overall_template(templates: dict[str, np.ndarray]) -> np.ndarray:
    if not templates:
        raise ValueError("At least one hand template is required")
    return mean_template(list(templates.values()))
