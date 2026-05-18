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

        def add_access_log(self, user_id, matched_name, status, similarity):
            self.logged.append((user_id, matched_name, status, similarity))

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
    assert sample["quality_score"] == 0.85
    np.testing.assert_array_equal(sample["embedding"], np.ones(4, dtype=np.float32))


def test_finalize_registration_stores_best_samples():
    from app.device_runtime import DeviceRuntime

    class FakeProcessor:
        def compute_similarity(self, embedding, stored, threshold):
            return {"status": "DENIED", "name": "Unknown", "similarity": 0.1}

    class FakeDB:
        def __init__(self):
            self.added = None

        def get_all_embeddings(self):
            return []

        def add_user(self, name, embedding, individual_embeddings=None):
            self.added = (name, embedding, individual_embeddings)
            return 123

    runtime = DeviceRuntime(camera=None, palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.captured_samples = [
        {"sample_index": i, "quality_score": 1.0, "embedding": np.ones(4, dtype=np.float32)}
        for i in range(7)
    ]

    result = runtime.finalize_registration()

    assert result["user_id"] == 123
    assert runtime.db.added[0] == "Alice"
    assert len(runtime.db.added[2]) == 5
    assert runtime.registration_session is None
    assert runtime.worker_state == "running"


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
        {"sample_index": i, "quality_score": 1.0, "embedding": np.ones(4, dtype=np.float32)}
        for i in range(7)
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


def test_capture_registration_sample_stops_after_seven_samples():
    from app.device_runtime import DeviceRuntime

    runtime = DeviceRuntime(camera=None, palm_processor=None, db=None)
    runtime.start_registration("Alice")
    runtime.registration_session.current_sample_index = 7
    runtime.registration_session.last_guidance = {"acceptable": True, "score": 1.0}

    try:
        runtime.capture_registration_sample()
    except RuntimeError as exc:
        assert "All registration samples captured" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


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
                "rotation_degrees": 0.0,
                "center_x_ratio": 0.65,
                "brightness": 120.0,
                "blur_score": 150.0,
                "steady": True,
            }

    class FakeDB:
        def upsert_device_status(self, **kwargs):
            pass

    runtime = DeviceRuntime(FakeCamera(), palm_processor=FakeProcessor(), db=FakeDB())
    runtime.start_registration("Alice")
    runtime.registration_session.current_sample_index = 7

    runtime.tick()

    assert runtime.registration_session.last_guidance["target"] == "shift_right"


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
