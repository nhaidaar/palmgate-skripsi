from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegistrationSample:
    sample_index: int
    quality_score: float
    embedding: np.ndarray


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_registration_samples(
    samples: list[RegistrationSample],
    keep: int,
    min_similarity: float,
) -> list[RegistrationSample]:
    if not samples:
        return []

    scored = []
    for sample in samples:
        others = [other for other in samples if other is not sample]
        if not others:
            mean_similarity = 1.0
        else:
            mean_similarity = float(
                np.mean([_cosine(sample.embedding, other.embedding) for other in others])
            )
        if mean_similarity >= min_similarity:
            scored.append((sample.quality_score + mean_similarity, sample))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [sample for _, sample in scored[:keep]]
