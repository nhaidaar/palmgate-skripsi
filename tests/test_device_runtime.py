import numpy as np



def test_runtime_recognizes_after_hold_threshold():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {
                "hand_detected": True,
                "hand_clipped": False,
                "height_ratio": 0.55,
                "rotation_degrees": 0.0,
                "center_x_ratio": 0.5,
                "brightness": 120.0,
                "blur_score": 150.0,
                "steady": True,
            }

        def get_embedding_from_notebook_frame(self, frame):
            return np.ones(4, dtype=np.float32)

        def get_embedding(self, frame):
            raise AssertionError("USB runtime must use notebook preprocessing")

        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "ALLOWED",
                "name": "Naufal",
                "similarity": 0.91,
                "closest_match": "Naufal",
                "user_id": 1,
            }

    class FakeDB:
        def __init__(self):
            self.logged = []

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            self.logged.append((user_id, matched_name, status, similarity, duration_ms, description))

        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
        cooldown_ms=3000,
    )

    runtime.clock.now_ms = 0
    runtime.tick()
    runtime.clock.now_ms = 1200
    runtime.tick()

    assert runtime.db.logged[0][2] == "ALLOWED"
    assert runtime.db.logged[0][4] == 1200


def test_runtime_stores_latest_camera_frame_for_preview():
    from app.device_runtime import DeviceRuntime

    frame = np.full((240, 320, 3), 127, dtype=np.uint8)

    class FakeClock:
        def now(self):
            return 0

    class FakeCamera:
        def read(self):
            return frame

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": False, "hand_clipped": True}

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(FakeCamera(), FakeProcessor(), FakeDB(), clock=FakeClock())

    runtime.tick()

    np.testing.assert_array_equal(runtime.latest_frame, frame)
    assert runtime.get_latest_frame_jpeg().startswith(b"\xff\xd8")


def test_runtime_preview_capture_is_independent_from_scan_processing():
    from app.device_runtime import DeviceRuntime

    frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
    ]

    class FakeCamera:
        def __init__(self):
            self.read_count = 0

        def read(self):
            frame = frames[min(self.read_count, len(frames) - 1)]
            self.read_count += 1
            return frame

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=None,
        db=None,
        frame_interval_ms=1000,
        preview_frame_interval_ms=100,
    )

    runtime.capture_preview_frame()
    runtime.capture_preview_frame()

    np.testing.assert_array_equal(runtime.latest_frame, frames[1])
    assert runtime.camera.read_count == 2
    assert runtime.preview_frame_interval_ms == 100


def test_runtime_tracks_scan_state_when_no_hand_detected():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def now(self):
            return 0

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": False, "brightness": 80.0, "blur_score": 120.0}

        def get_embedding_from_notebook_frame(self, frame):
            raise AssertionError("No-hand frames must not be embedded")

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(FakeCamera(), FakeProcessor(), FakeDB(), clock=FakeClock())

    runtime.tick()

    assert runtime.scan_state["stage"] == "waiting_for_hand"
    assert runtime.scan_state["metrics"]["hand_detected"] is False


def test_runtime_tracks_scan_state_while_holding_detected_hand():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": True, "hand_clipped": True}

        def get_embedding_from_notebook_frame(self, frame):
            return np.ones(4, dtype=np.float32)

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
    )

    runtime.tick()

    assert runtime.scan_state["stage"] == "holding"
    assert runtime.scan_state["hold_elapsed_ms"] == 0
    assert runtime.scan_state["hold_required_ms"] == 1000


def test_runtime_does_not_embed_until_hold_window_completes():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def __init__(self):
            self.embedding_calls = 0

        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": True, "hand_clipped": False}

        def get_embedding_from_notebook_frame(self, frame):
            self.embedding_calls += 1
            return np.ones(4, dtype=np.float32)

        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.5,
                "closest_match": "Naufal",
                "user_id": None,
            }

    class FakeDB:
        def __init__(self):
            self.logged = []

        def get_all_embeddings(self):
            return []

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            self.logged.append((user_id, matched_name, status, similarity, duration_ms, description))

        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    processor = FakeProcessor()
    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=processor,
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
    )

    runtime.tick()
    runtime.clock.now_ms = 500
    runtime.tick()
    assert processor.embedding_calls == 0
    assert runtime.db.logged == []

    runtime.clock.now_ms = 1000
    runtime.tick()
    assert processor.embedding_calls == 1
    assert runtime.db.logged[0][5] == "similar to Naufal"


def test_runtime_scans_again_after_cooldown_with_hand_still_present():
    """After recognition, system scans again once cooldown expires, even if hand never left."""
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def __init__(self):
            self.hand_detected = True
            self.embedding_calls = 0

        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": self.hand_detected, "hand_clipped": False}

        def get_embedding_from_notebook_frame(self, frame):
            self.embedding_calls += 1
            return np.ones(4, dtype=np.float32)

        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "ALLOWED",
                "name": "Naufal",
                "similarity": 0.91,
                "closest_match": "Naufal",
                "user_id": 1,
            }

    class FakeDB:
        def __init__(self):
            self.logged = []

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity, duration_ms=None, description=None):
            self.logged.append((user_id, matched_name, status, similarity, duration_ms, description))

        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    processor = FakeProcessor()
    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=processor,
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
        cooldown_ms=3000,
    )

    # First scan: hold for 1000ms
    runtime.tick()
    runtime.clock.now_ms = 1000
    runtime.tick()
    assert processor.embedding_calls == 1
    assert len(runtime.db.logged) == 1

    # During cooldown (3000ms), no new scan even with hand present
    runtime.clock.now_ms = 2000
    runtime.tick()
    assert processor.embedding_calls == 1
    assert runtime.scan_state["stage"] == "cooldown"

    # After cooldown expires, hand still present — should start new hold
    runtime.clock.now_ms = 5000
    runtime.tick()
    assert runtime.scan_state["stage"] == "holding"

    # Complete second hold — should scan again
    runtime.clock.now_ms = 6000
    runtime.tick()
    assert processor.embedding_calls == 2
    assert len(runtime.db.logged) == 2


def test_runtime_starts_hold_for_detected_clipped_hand():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": True, "hand_clipped": True}

        def get_embedding_from_notebook_frame(self, frame):
            return np.ones(4, dtype=np.float32)

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
    )

    assert runtime.tick() is None
    assert runtime.hand_seen_since_ms == 0


def test_runtime_does_not_recognize_without_detected_hand():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {"hand_detected": False, "hand_clipped": True}

        def get_embedding_from_notebook_frame(self, frame):
            raise AssertionError("background must not be embedded without a detected hand")

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
    )

    runtime.clock.now_ms = 0
    assert runtime.tick() is None
    runtime.clock.now_ms = 1200
    assert runtime.tick() is None
    assert runtime.hand_seen_since_ms is None


def test_start_registration_pauses_recognition():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def now(self):
            return 0

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {
                "hand_detected": False,
                "hand_clipped": True,
                "height_ratio": 0.0,
                "rotation_degrees": 999.0,
                "center_x_ratio": 0.0,
                "brightness": 0.0,
                "blur_score": 0.0,
                "steady": False,
            }

        def get_embedding_from_notebook_frame(self, frame):
            raise AssertionError("recognition should be paused during registration")

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(FakeCamera(), FakeProcessor(), FakeDB(), clock=FakeClock())

    runtime.start_registration("Alice")
    result = runtime.tick()

    assert result is None
    assert runtime.registration_session.name == "Alice"
    assert runtime.worker_state == "registration_active"


def test_cancel_registration_returns_to_running_state():
    from app.device_runtime import DeviceRuntime

    runtime = DeviceRuntime(camera=None, palm_processor=None, db=None)

    runtime.start_registration("Alice")
    runtime.cancel_registration()

    assert runtime.registration_session is None
    assert runtime.worker_state == "running"


def test_capture_registration_sample_requires_active_session():
    from app.device_runtime import DeviceRuntime

    runtime = DeviceRuntime(camera=None, palm_processor=None, db=None)

    try:
        runtime.capture_registration_sample()
    except RuntimeError as exc:
        assert "No registration active" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_capture_registration_sample_uses_guidance_score():
    from app.device_runtime import DeviceRuntime

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_embedding_from_notebook_frame(self, frame):
            return np.ones(4, dtype=np.float32)

    runtime = DeviceRuntime(camera=FakeCamera(), palm_processor=FakeProcessor(), db=None)
    runtime.start_registration("Alice")
    runtime.registration_session.last_guidance = {"acceptable": True, "score": 0.85}

    sample = runtime.capture_registration_sample()

    assert sample["sample_index"] == 0
    assert sample["hand"] == "left"
    assert sample["quality_score"] == 0.85
    np.testing.assert_array_equal(sample["embedding"], np.ones(4, dtype=np.float32))


def test_capture_registration_sample_tags_first_five_left_next_five_right():
    from app.device_runtime import DeviceRuntime

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_embedding_from_notebook_frame(self, frame):
            return np.ones(4, dtype=np.float32)

    runtime = DeviceRuntime(camera=FakeCamera(), palm_processor=FakeProcessor(), db=None)
    runtime.start_registration("Alice")
    runtime.registration_session.last_guidance = {"acceptable": True, "score": 1.0}

    samples = [runtime.capture_registration_sample() for _ in range(10)]

    assert [sample["hand"] for sample in samples] == ["left"] * 5 + ["right"] * 5


def test_finalize_registration_stores_five_embeddings_per_hand():
    from app.device_runtime import DeviceRuntime

    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {"status": "DENIED", "name": "Unknown", "similarity": 0.1}

    class FakeDB:
        def __init__(self):
            self.added = None

        def get_all_embeddings(self):
            return []

        def add_user(self, name, embedding, individual_embeddings=None, embedding_hands=None):
            self.added = (name, embedding, individual_embeddings, embedding_hands)
            return 123

    runtime = DeviceRuntime(camera=None, palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.captured_samples = [
        {
            "sample_index": i,
            "hand": "left" if i < 5 else "right",
            "quality_score": 1.0,
            "embedding": np.ones(4, dtype=np.float32),
        }
        for i in range(10)
    ]

    result = runtime.finalize_registration()

    assert result["user_id"] == 123
    assert result["stored_embeddings"] == 10
    assert result["hands"] == {"left": 5, "right": 5}
    assert runtime.db.added[0] == "Alice"
    assert len(runtime.db.added[2]) == 10
    assert runtime.db.added[3] == ["left"] * 5 + ["right"] * 5
    assert runtime.registration_session is None
    assert runtime.worker_state == "running"


def test_finalize_registration_requires_five_valid_samples_per_hand():
    from app.device_runtime import DeviceRuntime

    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {"status": "DENIED", "name": "Unknown", "similarity": 0.1}

    class FakeDB:
        def get_all_embeddings(self):
            return []

        def add_user(self, *args, **kwargs):
            raise AssertionError("Incomplete registration should not be stored")

    runtime = DeviceRuntime(camera=None, palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.captured_samples = [
        {
            "sample_index": i,
            "hand": "left" if i < 5 else "right",
            "quality_score": 1.0,
            "embedding": np.ones(4, dtype=np.float32),
        }
        for i in range(9)
    ]

    try:
        runtime.finalize_registration()
    except RuntimeError as exc:
        assert "Not enough valid registration samples" in str(exc)
    else:
        raise AssertionError("Expected incomplete registration rejection")


def test_finalize_registration_rejects_duplicate_palm():
    from app.device_runtime import DeviceRuntime

    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {"status": "ALLOWED", "name": "Existing", "similarity": 0.9}

    class FakeDB:
        def get_all_embeddings(self):
            return [{"id": 1, "name": "Existing", "embedding": np.ones(4, dtype=np.float32)}]

        def add_user(self, *args, **kwargs):
            raise AssertionError("Duplicate should not be stored")

    runtime = DeviceRuntime(camera=None, palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.captured_samples = [
        {
            "sample_index": i,
            "hand": "left" if i < 5 else "right",
            "quality_score": 1.0,
            "embedding": np.ones(4, dtype=np.float32),
        }
        for i in range(10)
    ]

    try:
        runtime.finalize_registration()
    except RuntimeError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate rejection")


def test_capture_requires_acceptable_guidance():
    from app.device_runtime import DeviceRuntime

    runtime = DeviceRuntime(camera=None, palm_processor=None, db=None)
    runtime.start_registration("Alice")
    runtime.registration_session.last_guidance = {"acceptable": False, "failures": ["size"]}

    try:
        runtime.capture_registration_sample()
    except RuntimeError as exc:
        assert "Frame does not satisfy guidance" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_capture_registration_sample_stops_after_ten_samples():
    from app.device_runtime import DeviceRuntime

    runtime = DeviceRuntime(camera=None, palm_processor=None, db=None)
    runtime.start_registration("Alice")
    runtime.registration_session.current_sample_index = 10
    runtime.registration_session.last_guidance = {"acceptable": True, "score": 1.0}

    try:
        runtime.capture_registration_sample()
    except RuntimeError as exc:
        assert "All registration samples captured" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_registration_tick_reuses_five_guidance_targets_per_hand():
    from app.device_runtime import DeviceRuntime

    class FakeCamera:
        def read(self):
            return np.full((240, 320, 3), 128, dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {
                "hand_detected": True,
                "hand_clipped": False,
                "height_ratio": 0.55,
                "rotation_degrees": 0.0,
                "center_x_ratio": 0.5,
                "brightness": 120.0,
                "blur_score": 150.0,
                "steady": True,
            }

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            pass

    runtime = DeviceRuntime(FakeCamera(), palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.current_sample_index = 5

    runtime.tick()

    assert runtime.registration_session.last_guidance["target"] == "center"


def test_registration_tick_after_final_sample_keeps_last_target():
    from app.device_runtime import DeviceRuntime

    class FakeCamera:
        def read(self):
            return np.full((240, 320, 3), 128, dtype=np.uint8)

    class FakeProcessor:
        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            return {
                "hand_detected": True,
                "hand_clipped": False,
                "height_ratio": 0.55,
                "rotation_degrees": 10.0,
                "center_x_ratio": 0.5,
                "brightness": 120.0,
                "blur_score": 150.0,
                "steady": True,
            }

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            pass

    runtime = DeviceRuntime(FakeCamera(), palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.current_sample_index = 10

    runtime.tick()

    assert runtime.registration_session.last_guidance["target"] == "rotate_right"


def test_registration_tick_updates_real_guidance_from_processor():
    from app.device_runtime import DeviceRuntime

    class FakeCamera:
        def read(self):
            return np.full((240, 320, 3), 128, dtype=np.uint8)

    class FakeProcessor:
        def __init__(self):
            self.called = False

        def get_registration_guidance_metrics(self, frame, previous_metrics=None):
            self.called = True
            return {
                "hand_detected": True,
                "hand_clipped": False,
                "height_ratio": 0.55,
                "rotation_degrees": 0.0,
                "center_x_ratio": 0.5,
                "brightness": 120.0,
                "blur_score": 150.0,
                "steady": True,
            }

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            pass

    processor = FakeProcessor()
    runtime = DeviceRuntime(FakeCamera(), palm_processor=processor, db=FakeDB())
    runtime.start_registration("Alice")

    runtime.tick()

    assert processor.called is True
    assert runtime.registration_session.last_guidance["acceptable"] is True
    assert runtime.registration_session.last_guidance["target"] == "center"
