import numpy as np

from app.services.embedding_templates import build_hand_templates, l2_normalize, mean_template, overall_template


def test_l2_normalize_returns_unit_vector():
    result = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(result, np.array([0.6, 0.8], dtype=np.float32), rtol=1e-6)


def test_mean_template_normalizes_average():
    result = mean_template([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])

    np.testing.assert_allclose(result, np.array([0.70710677, 0.70710677], dtype=np.float32), rtol=1e-6)


def test_build_hand_templates_requires_each_hand():
    samples = [
        {"hand": "left", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
        {"hand": "right", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
    ]

    templates = build_hand_templates(samples, required_hands=("left", "right"), min_per_hand=1)

    assert set(templates) == {"left", "right"}
    np.testing.assert_allclose(templates["left"], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(templates["right"], np.array([0.0, 1.0], dtype=np.float32))


def test_build_hand_templates_rejects_missing_hand():
    samples = [{"hand": "left", "embedding": np.array([1.0, 0.0], dtype=np.float32)}]

    try:
        build_hand_templates(samples, required_hands=("left", "right"), min_per_hand=1)
    except ValueError as exc:
        assert "Not enough valid right samples" in str(exc)
    else:
        raise AssertionError("Expected missing right hand samples to fail")


def test_overall_template_averages_hand_templates():
    result = overall_template({
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, 1.0], dtype=np.float32),
    })

    np.testing.assert_allclose(result, np.array([0.70710677, 0.70710677], dtype=np.float32), rtol=1e-6)
