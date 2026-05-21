from app.services.registration_quality import SAMPLE_TARGETS, evaluate_guidance


def test_has_seven_registration_targets():
    assert [target.key for target in SAMPLE_TARGETS] == [
        "center",
        "closer",
        "farther",
        "rotate_left",
        "rotate_right",
        "shift_left",
        "shift_right",
    ]


def test_center_target_accepts_good_metrics():
    result = evaluate_guidance(
        sample_index=0,
        metrics={
            "hand_detected": True,
            "hand_clipped": False,
            "height_ratio": 0.55,
            "rotation_degrees": 0.0,
            "center_x_ratio": 0.50,
            "brightness": 120.0,
            "blur_score": 150.0,
            "steady": True,
        },
    )

    assert result.acceptable is True
    assert result.failures == []


def test_pose_mismatch_is_guidance_not_capture_blocker():
    result = evaluate_guidance(
        sample_index=3,
        metrics={
            "hand_detected": True,
            "hand_clipped": False,
            "height_ratio": 0.55,
            "rotation_degrees": 0.0,
            "center_x_ratio": 0.50,
            "brightness": 120.0,
            "blur_score": 150.0,
            "steady": True,
        },
    )

    assert result.acceptable is True
    assert "rotation" in result.failures
    assert result.blockers == []


def test_missing_hand_still_blocks_capture():
    result = evaluate_guidance(
        sample_index=0,
        metrics={
            "hand_detected": False,
            "hand_clipped": True,
            "height_ratio": 0.0,
            "rotation_degrees": 999.0,
            "center_x_ratio": 0.0,
            "brightness": 120.0,
            "blur_score": 150.0,
            "steady": True,
        },
    )

    assert result.acceptable is False
    assert "hand" in result.blockers


def test_blurry_frame_still_blocks_capture():
    result = evaluate_guidance(
        sample_index=0,
        metrics={
            "hand_detected": True,
            "hand_clipped": False,
            "height_ratio": 0.55,
            "rotation_degrees": 0.0,
            "center_x_ratio": 0.50,
            "brightness": 120.0,
            "blur_score": 20.0,
            "steady": True,
        },
    )

    assert result.acceptable is False
    assert "sharpness" in result.blockers
