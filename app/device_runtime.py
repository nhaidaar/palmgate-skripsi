from dataclasses import dataclass, field
import logging
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from app.camera import OpenCVCameraSource
from app.config import (
    CAMERA_DEVICE_PATH,
    DEVICE_COOLDOWN_MS,
    DEVICE_FRAME_INTERVAL_MS,
    DEVICE_PREVIEW_FRAME_INTERVAL_MS,
    DEVICE_HOLD_MS,
    DEVICE_STATUS_HEARTBEAT_MS,
    DUPLICATE_THRESHOLD,
    ENROLLMENT_TTA_ENABLED,
    RECOGNITION_TTA_ENABLED,
    REGISTRATION_CAPTURES_PER_HAND,
    REGISTRATION_HANDS,
    REGISTRATION_MIN_VALID_PER_HAND,
    SIMILARITY_THRESHOLD,
)
from app.lock_controller import NoopLockController, build_lock_controller
from app.services.embedding_templates import build_hand_templates, overall_template
from app.services.recognition_service import match_embedding_and_log
from app.services.registration_quality import SAMPLE_TARGETS, evaluate_guidance


log = logging.getLogger("palmgate.device_runtime")

@dataclass
class DeviceRegistrationSession:
    id: str
    nim: str
    name: str
    hands: tuple[str, ...] = REGISTRATION_HANDS
    current_sample_index: int = 0
    captured_samples: list = field(default_factory=list)
    last_guidance: dict | None = None


class SystemClock:
    def now(self):
        return int(time.time() * 1000)


class ScanEventBroadcaster:
    """Manages SSE subscribers for scan state changes."""

    def __init__(self):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, event: dict):
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass


class DeviceRuntime:
    def __init__(
        self,
        camera,
        palm_processor,
        db,
        clock=None,
        hold_ms: int = DEVICE_HOLD_MS,
        cooldown_ms: int = DEVICE_COOLDOWN_MS,
        frame_interval_ms: int = DEVICE_FRAME_INTERVAL_MS,
        preview_frame_interval_ms: int = DEVICE_PREVIEW_FRAME_INTERVAL_MS,
        heartbeat_ms: int = DEVICE_STATUS_HEARTBEAT_MS,
        threshold: float = SIMILARITY_THRESHOLD,
        lock_controller=None,
    ):
        self.camera = camera
        self.palm_processor = palm_processor
        self.db = db
        self.clock = clock or SystemClock()
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.frame_interval_ms = frame_interval_ms
        self.preview_frame_interval_ms = preview_frame_interval_ms
        self.heartbeat_ms = heartbeat_ms
        self.threshold = threshold
        self.lock_controller = lock_controller or NoopLockController()
        self.hand_seen_since_ms = None
        self.cooldown_until_ms = 0
        self.last_heartbeat_ms = None
        self.last_recognition_at = None
        self.worker_state = "running"
        self.registration_session = None
        self._registration_lock = threading.Lock()
        self.scan_state = {"stage": "starting", "metrics": None}
        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._preview_thread = None
        self._stop_event = threading.Event()
        self.scan_broadcaster = ScanEventBroadcaster()

    def capture_preview_frame(self):
        frame = self.camera.read()
        with self._frame_lock:
            self.latest_frame = frame.copy()
        return frame

    def _latest_frame_copy(self):
        with self._frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def _read_frame(self):
        frame = self._latest_frame_copy()
        if frame is None:
            frame = self.capture_preview_frame()
        return frame

    def _selected_registration_hands(self, hands=None) -> tuple[str, ...]:
        if hands is None:
            return REGISTRATION_HANDS
        selected = []
        for hand in hands:
            clean = str(hand).lower()
            if clean not in REGISTRATION_HANDS:
                raise RuntimeError("Select left, right, or both hands")
            if clean not in selected:
                selected.append(clean)
        if not selected:
            raise RuntimeError("Select at least one hand to register")
        return tuple(hand for hand in REGISTRATION_HANDS if hand in selected)

    def _registration_total_captures(self, hands=None) -> int:
        return REGISTRATION_CAPTURES_PER_HAND * len(hands or REGISTRATION_HANDS)

    def _hand_for_sample_index(self, index: int, hands=None) -> str:
        selected_hands = hands or REGISTRATION_HANDS
        hand_index = index // REGISTRATION_CAPTURES_PER_HAND
        return selected_hands[min(hand_index, len(selected_hands) - 1)]

    def _target_index_for_sample_index(self, index: int) -> int:
        return index % REGISTRATION_CAPTURES_PER_HAND

    def get_registration_status(self) -> dict:
        with self._registration_lock:
            session = self.registration_session
            if session is None:
                return {"active": False, "worker_state": self.worker_state}

            counts = {hand: 0 for hand in REGISTRATION_HANDS}
            for i, sample in enumerate(session.captured_samples):
                hand = sample.get("hand", self._hand_for_sample_index(i, session.hands))
                if hand in counts:
                    counts[hand] += 1

            return {
                "active": True,
                "worker_state": self.worker_state,
                "session_id": session.id,
                "nim": session.nim,
                "name": session.name,
                "current_sample_index": session.current_sample_index,
                "captured_count": len(session.captured_samples),
                "guidance": session.last_guidance,
                "required_per_hand": REGISTRATION_CAPTURES_PER_HAND,
                "total_required": self._registration_total_captures(session.hands),
                "current_hand": self._hand_for_sample_index(session.current_sample_index, session.hands),
                "left_count": counts.get("left", 0),
                "right_count": counts.get("right", 0),
            }

    def get_latest_frame_jpeg(self):
        frame = self._latest_frame_copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        if not ok:
            return None
        return encoded.tobytes()

    def start_registration(self, nim: str, name: str, hands=None):
        clean_nim = nim.strip()
        clean_name = name.strip()
        selected_hands = self._selected_registration_hands(hands)
        if not clean_nim:
            raise RuntimeError("NIM is required")
        if not clean_name:
            raise RuntimeError("Name is required")
        with self._registration_lock:
            if self.registration_session is not None:
                raise RuntimeError("Registration already active")
            self.registration_session = DeviceRegistrationSession(
                id=str(uuid.uuid4()),
                nim=clean_nim,
                name=clean_name,
                hands=selected_hands,
            )
            self.worker_state = "registration_active"
            self.hand_seen_since_ms = None
            self.cooldown_until_ms = 0
            return self.registration_session

    def cancel_registration(self):
        with self._registration_lock:
            self.registration_session = None
            self.worker_state = "running"

    def capture_registration_sample(self):
        with self._registration_lock:
            if self.registration_session is None:
                raise RuntimeError("No registration active")
            if self.registration_session.current_sample_index >= self._registration_total_captures(self.registration_session.hands):
                raise RuntimeError("All registration samples captured")
            guidance = self.registration_session.last_guidance
            if not guidance or not guidance.get("acceptable", False):
                raise RuntimeError("Frame does not satisfy guidance")
            sample_index = self.registration_session.current_sample_index

            frame = self._read_frame()

            import os
            import cv2
            base_data_dir = "/data" if os.path.exists("/data") else "data"
            save_dir = os.path.join(base_data_dir, "captures", f"{self.registration_session.nim}_{self.registration_session.id}")
            os.makedirs(save_dir, exist_ok=True)
            hand = self._hand_for_sample_index(sample_index, self.registration_session.hands)
            cv2.imwrite(os.path.join(save_dir, f"{hand}_{sample_index}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            embedding = self.palm_processor.get_embedding_from_notebook_frame(
                frame,
                tta_enabled=ENROLLMENT_TTA_ENABLED,
            )
            if embedding is None:
                raise RuntimeError("Palm preprocessing failed")
            sample = {
                "sample_index": sample_index,
                "hand": hand,
                "quality_score": float(guidance.get("score", 1.0)),
                "embedding": embedding,
            }
            self.registration_session.captured_samples.append(sample)
            self.registration_session.current_sample_index += 1
            self.registration_session.last_guidance = None
            return sample

    def finalize_registration(self):
        with self._registration_lock:
            if self.registration_session is None:
                raise RuntimeError("No registration active")

            samples = []
            for item in self.registration_session.captured_samples:
                samples.append({
                    "hand": item.get("hand", self._hand_for_sample_index(item["sample_index"], self.registration_session.hands)),
                    "embedding": item["embedding"],
                })

            try:
                templates = build_hand_templates(
                    samples,
                    required_hands=self.registration_session.hands,
                    min_per_hand=REGISTRATION_MIN_VALID_PER_HAND,
                )
            except ValueError as exc:
                raise RuntimeError("Not enough valid registration samples") from exc

            embedding_hands = list(templates.keys())
            embeddings = [templates[hand].astype(np.float32) for hand in embedding_hands]
            avg_embedding = overall_template(templates)

            stored = self.db.get_all_embeddings()
            for embedding in embeddings:
                duplicate = self.palm_processor.compute_similarity(embedding, stored, DUPLICATE_THRESHOLD)
                if duplicate["status"] == "ALLOWED":
                    raise RuntimeError(f"This palm is already registered as '{duplicate['name']}'")

            try:
                user_id = self.db.add_user(
                    self.registration_session.name,
                    avg_embedding,
                    nim=self.registration_session.nim,
                    individual_embeddings=embeddings,
                    embedding_hands=embedding_hands,
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc

            nim = self.registration_session.nim
            name = self.registration_session.name
            self.registration_session = None
            self.worker_state = "running"
            return {
                "user_id": user_id,
                "nim": nim,
                "name": name,
                "stored_embeddings": len(embeddings),
                "hands": {hand: embedding_hands.count(hand) for hand in REGISTRATION_HANDS},
            }

    def tick(self):
        now_ms = self.clock.now()

        if self.last_heartbeat_ms is None or now_ms - self.last_heartbeat_ms >= self.heartbeat_ms:
            self.db.upsert_device_status(
                worker_state=self.worker_state,
                camera_connected=True,
                last_error=None,
                fps=(1000 / self.frame_interval_ms) if self.frame_interval_ms > 0 else None,
                last_inference_ms=None,
                last_recognition_at=self.last_recognition_at,
            )
            self.last_heartbeat_ms = now_ms

        with self._registration_lock:
            if self.registration_session is not None:
                frame = self._read_frame()
                previous_metrics = None
                if self.registration_session.last_guidance:
                    previous_metrics = self.registration_session.last_guidance.get("metrics")
                metrics = self.palm_processor.get_registration_guidance_metrics(frame, previous_metrics)
                total = self._registration_total_captures(self.registration_session.hands)
                sample_index = min(self.registration_session.current_sample_index, total - 1)
                target_index = self._target_index_for_sample_index(sample_index)
                guidance = evaluate_guidance(target_index, metrics)
                target = SAMPLE_TARGETS[target_index]
                self.registration_session.last_guidance = {
                    "target": target.key,
                    "label": target.label,
                    "acceptable": guidance.acceptable,
                    "failures": guidance.failures,
                    "blockers": guidance.blockers,
                    "score": guidance.score,
                    "metrics": metrics,
                }
                return None

        if now_ms < self.cooldown_until_ms:
            self.scan_state = {
                "stage": "cooldown",
                "cooldown_remaining_ms": self.cooldown_until_ms - now_ms,
                "metrics": None,
            }
            return None

        frame = self._read_frame()
        metrics = self.palm_processor.get_registration_guidance_metrics(frame)
        if not metrics.get("hand_detected", False):
            self.hand_seen_since_ms = None
            self.scan_state = {"stage": "waiting_for_hand", "metrics": metrics}
            return None

        if self.hand_seen_since_ms is None:
            self.hand_seen_since_ms = now_ms
            self.scan_state = {
                "stage": "holding",
                "hold_elapsed_ms": 0,
                "hold_required_ms": self.hold_ms,
                "metrics": metrics,
            }
            return None

        hold_elapsed_ms = now_ms - self.hand_seen_since_ms
        if hold_elapsed_ms < self.hold_ms:
            self.scan_state = {
                "stage": "holding",
                "hold_elapsed_ms": hold_elapsed_ms,
                "hold_required_ms": self.hold_ms,
                "metrics": metrics,
            }
            return None

        self.scan_state = {"stage": "recognizing", "metrics": metrics}
        embedding = self.palm_processor.get_embedding_from_notebook_frame(
            frame,
            tta_enabled=RECOGNITION_TTA_ENABLED,
        )
        if embedding is None:
            self.hand_seen_since_ms = None
            self.scan_state = {"stage": "preprocessing_failed", "metrics": metrics}
            return None

        result = match_embedding_and_log(
            self.palm_processor,
            self.db,
            embedding,
            self.threshold,
            duration_ms=hold_elapsed_ms,
        )
        self.last_recognition_at = str(now_ms)
        self.cooldown_until_ms = now_ms + self.cooldown_ms
        self.hand_seen_since_ms = None
        self.scan_state = {"stage": "recognized", "result": result, "metrics": metrics}
        self.scan_broadcaster.broadcast({
            "stage": "recognized",
            "result": result,
            "timestamp": self.last_recognition_at,
        })
        self.db.upsert_device_status(
            worker_state=self.worker_state,
            camera_connected=True,
            last_error=None,
            fps=(1000 / self.frame_interval_ms) if self.frame_interval_ms > 0 else None,
            last_inference_ms=0.0,
            last_recognition_at=self.last_recognition_at,
        )
        if result.get("status") == "ALLOWED":
            try:
                self.lock_controller.unlock()
            except Exception:
                log.exception("Door unlock failed")
        return result

    def _run_preview_loop(self):
        while not self._stop_event.is_set():
            try:
                self.capture_preview_frame()
            except Exception as exc:
                self.db.upsert_device_status(
                    worker_state="error",
                    camera_connected=False,
                    last_error=str(exc),
                    fps=None,
                    last_inference_ms=None,
                    last_recognition_at=self.last_recognition_at,
                )
            time.sleep(self.preview_frame_interval_ms / 1000)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.db.upsert_device_status(
                    worker_state="error",
                    camera_connected=False,
                    last_error=str(exc),
                    fps=None,
                    last_inference_ms=None,
                    last_recognition_at=self.last_recognition_at,
                )
            time.sleep(self.frame_interval_ms / 1000)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._preview_thread = threading.Thread(target=self._run_preview_loop, daemon=True)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._preview_thread.start()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2)
        close = getattr(self.camera, "close", None)
        if callable(close):
            close()
        close_lock = getattr(self.lock_controller, "close", None)
        if callable(close_lock):
            close_lock()


def build_device_runtime(palm_processor, db):
    camera = OpenCVCameraSource(CAMERA_DEVICE_PATH)
    return DeviceRuntime(
        camera=camera,
        palm_processor=palm_processor,
        db=db,
        lock_controller=build_lock_controller(),
    )


if __name__ == "__main__":
    from app.config import DB_PATH
    from app.database import Database
    from app.palm_processor import PalmProcessor

    db = Database(DB_PATH)
    palm_processor = PalmProcessor()
    runtime = build_device_runtime(palm_processor, db)

    try:
        runtime.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        palm_processor.close()
        db.close()
