from dataclasses import dataclass, field
import threading
import time
import uuid

import cv2
import numpy as np

from app.camera import OpenCVCameraSource
from app.config import (
    CAMERA_DEVICE_INDEX,
    CAMERA_DEVICE_PATH,
    DEVICE_COOLDOWN_MS,
    DEVICE_FRAME_INTERVAL_MS,
    DEVICE_PREVIEW_FRAME_INTERVAL_MS,
    DEVICE_HOLD_MS,
    DEVICE_STATUS_HEARTBEAT_MS,
    DUPLICATE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    USB_REGISTRATION_CAPTURES,
    USB_REGISTRATION_MIN_VALID,
    USB_REGISTRATION_STORE_EMBEDDINGS,
)
from app.services.recognition_service import match_embedding_and_log
from app.services.registration_quality import SAMPLE_TARGETS, evaluate_guidance
from app.services.registration_ranking import RegistrationSample, rank_registration_samples


@dataclass
class DeviceRegistrationSession:
    id: str
    name: str
    current_sample_index: int = 0
    captured_samples: list = field(default_factory=list)
    last_guidance: dict | None = None


class SystemClock:
    def now(self):
        return int(time.time() * 1000)


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
        self.hand_seen_since_ms = None
        self.waiting_for_hand_release = False
        self.cooldown_until_ms = 0
        self.last_heartbeat_ms = None
        self.last_recognition_at = None
        self.worker_state = "running"
        self.registration_session = None
        self.scan_state = {"stage": "starting", "metrics": None}
        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._preview_thread = None
        self._stop_event = threading.Event()

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

    def get_latest_frame_jpeg(self):
        frame = self._latest_frame_copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        return encoded.tobytes()

    def start_registration(self, name: str):
        if self.registration_session is not None:
            raise RuntimeError("Registration already active")
        self.registration_session = DeviceRegistrationSession(id=str(uuid.uuid4()), name=name.strip())
        self.worker_state = "registration_active"
        self.hand_seen_since_ms = None
        self.waiting_for_hand_release = False
        self.cooldown_until_ms = 0
        return self.registration_session

    def cancel_registration(self):
        self.registration_session = None
        self.worker_state = "running"

    def capture_registration_sample(self):
        if self.registration_session is None:
            raise RuntimeError("No registration active")
        if self.registration_session.current_sample_index >= USB_REGISTRATION_CAPTURES:
            raise RuntimeError("All registration samples captured")
        guidance = self.registration_session.last_guidance
        if not guidance or not guidance.get("acceptable", False):
            raise RuntimeError("Frame does not satisfy guidance")
        frame = self._read_frame()
        embedding = self.palm_processor.get_embedding_from_notebook_frame(frame)
        if embedding is None:
            raise RuntimeError("Notebook preprocessing failed")
        sample = {
            "sample_index": self.registration_session.current_sample_index,
            "quality_score": float(guidance.get("score", 1.0)),
            "embedding": embedding,
        }
        self.registration_session.captured_samples.append(sample)
        self.registration_session.current_sample_index += 1
        return sample

    def finalize_registration(self):
        if self.registration_session is None:
            raise RuntimeError("No registration active")

        samples = [
            RegistrationSample(
                sample_index=item["sample_index"],
                quality_score=item["quality_score"],
                embedding=item["embedding"],
            )
            for item in self.registration_session.captured_samples
        ]
        ranked = rank_registration_samples(
            samples,
            keep=USB_REGISTRATION_STORE_EMBEDDINGS,
            min_similarity=self.threshold,
        )
        if len(ranked) < USB_REGISTRATION_MIN_VALID:
            raise RuntimeError("Not enough valid registration samples")

        embeddings = [sample.embedding.astype(np.float32) for sample in ranked]
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        stored = self.db.get_all_embeddings()
        duplicate = self.palm_processor.compute_similarity(avg_embedding, stored, DUPLICATE_THRESHOLD)
        if duplicate["status"] == "ALLOWED":
            raise RuntimeError(f"This palm is already registered as '{duplicate['name']}'")
        user_id = self.db.add_user(
            self.registration_session.name,
            avg_embedding,
            individual_embeddings=embeddings,
        )
        name = self.registration_session.name
        self.registration_session = None
        self.worker_state = "running"
        return {"user_id": user_id, "name": name, "stored_embeddings": len(embeddings)}

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

        if self.registration_session is not None:
            frame = self._read_frame()
            previous_metrics = None
            if self.registration_session.last_guidance:
                previous_metrics = self.registration_session.last_guidance.get("metrics")
            metrics = self.palm_processor.get_registration_guidance_metrics(frame, previous_metrics)
            target_index = min(self.registration_session.current_sample_index, USB_REGISTRATION_CAPTURES - 1)
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
            self.waiting_for_hand_release = False
            self.scan_state = {"stage": "waiting_for_hand", "metrics": metrics}
            return None

        if self.waiting_for_hand_release:
            self.hand_seen_since_ms = None
            self.scan_state = {"stage": "waiting_for_hand_release", "metrics": metrics}
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
        embedding = self.palm_processor.get_embedding_from_notebook_frame(frame)
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
        self.waiting_for_hand_release = True
        self.scan_state = {"stage": "recognized", "result": result, "metrics": metrics}
        self.db.upsert_device_status(
            worker_state=self.worker_state,
            camera_connected=True,
            last_error=None,
            fps=(1000 / self.frame_interval_ms) if self.frame_interval_ms > 0 else None,
            last_inference_ms=0.0,
            last_recognition_at=self.last_recognition_at,
        )
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


def build_device_runtime(palm_processor, db):
    camera = OpenCVCameraSource(CAMERA_DEVICE_PATH or CAMERA_DEVICE_INDEX)
    return DeviceRuntime(camera=camera, palm_processor=palm_processor, db=db)


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
