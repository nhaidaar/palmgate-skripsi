import numpy as np

from app.services.registration_ranking import RegistrationSample, rank_registration_samples


def sample(index, quality, embedding):
    return RegistrationSample(
        sample_index=index,
        quality_score=quality,
        embedding=np.array(embedding, dtype=np.float32),
    )


def test_ranking_keeps_best_five_by_quality_when_embeddings_agree():
    samples = [sample(i, 1.0 - i * 0.1, [1, 0, 0, 0]) for i in range(7)]

    ranked = rank_registration_samples(samples, keep=5, min_similarity=0.75)

    assert [s.sample_index for s in ranked] == [0, 1, 2, 3, 4]


def test_ranking_removes_embedding_outlier():
    samples = [sample(i, 0.9, [1, 0, 0, 0]) for i in range(6)]
    samples.append(sample(6, 1.0, [0, 1, 0, 0]))

    ranked = rank_registration_samples(samples, keep=5, min_similarity=0.75)

    assert all(s.sample_index != 6 for s in ranked)
    assert len(ranked) == 5
