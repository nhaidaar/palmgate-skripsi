import cv2
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    EMBEDDING_DIM,
    HAND_LANDMARKER_PATH,
    IMG_SIZE,
    MIN_PALM_WIDTH,
    MODEL_PATH,
    NOTEBOOK_REMBG_ENABLED,
    PALM_ROI_SCALE,
    TTA_ROTATIONS,
)
from app.notebook_preprocessing import NotebookPreprocessor

log = logging.getLogger("palmgate")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
log.propagate = True

# MediaPipe hand landmark indices (same as legacy API)
WRIST = 0
INDEX_FINGER_MCP = 5
MIDDLE_FINGER_MCP = 9
PINKY_MCP = 17


class PalmProcessor:
    def __init__(self, model_path=MODEL_PATH, hand_model_path=HAND_LANDMARKER_PATH):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
        )
        self.interpreter = None
        self._input_index = None
        self._output_index = None
        self._embedding_dim = EMBEDDING_DIM
        self._hand_landmarker = None
        self.notebook_preprocessor = NotebookPreprocessor(rembg_enabled=NOTEBOOK_REMBG_ENABLED)

        if hand_model_path is not None:
            self._load_hand_model(hand_model_path)

        if model_path is not None:
            self._load_model(model_path)

    def _load_hand_model(self, hand_model_path):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            # Lower thresholds to handle float16 model variance and webcam conditions
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def warmup_notebook_preprocessor(self):
        if self.notebook_preprocessor.rembg_enabled:
            self.notebook_preprocessor._get_rembg_session()

    def _load_model(self, model_path):
        kwargs = {"model_path": str(model_path)}
        try:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(num_threads=4, **kwargs)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(num_threads=4, **kwargs)
        except TypeError:
            try:
                from tflite_runtime.interpreter import Interpreter
                self.interpreter = Interpreter(**kwargs)
            except ImportError:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(**kwargs)

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]

        shape = output_details[0].get("shape", [])
        if hasattr(shape, "tolist"):
            shape = shape.tolist()
        if len(shape) != 2 or int(shape[-1]) <= 0:
            raise RuntimeError(f"Embedding model must output [1, dim], got {shape}")
        self._embedding_dim = int(shape[-1])
        log.info("MODEL | embedding output index=%d dim=%d", self._output_index, self._embedding_dim)

    def extract_palm_roi(self, frame_rgb: np.ndarray):
        if self._hand_landmarker is None:
            log.warning("DETECT | hand_landmarker not loaded")
            return None

        h, w = frame_rgb.shape[:2]
        log.debug("DETECT | image received  shape=%s  dtype=%s  min=%d  max=%d",
                  frame_rgb.shape, frame_rgb.dtype,
                  int(frame_rgb.min()), int(frame_rgb.max()))

        # Reject clearly broken frames (all-black or all-white)
        mean_brightness = float(frame_rgb.mean())
        log.debug("DETECT | mean brightness=%.1f", mean_brightness)
        if mean_brightness < 5:
            log.warning("DETECT | frame appears to be all-black — camera may not be ready")
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            log.warning(
                "DETECT | no hand found  (image %dx%d, brightness=%.1f)"
                " — try: hold palm flat, fill ~50%% of frame, good lighting",
                w, h, mean_brightness,
            )
            return None

        log.info("DETECT | hand found  hands=%d  landmarks=%d",
                 len(result.hand_landmarks), len(result.hand_landmarks[0]))

        landmarks = result.hand_landmarks[0]

        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_FINGER_MCP]
        pinky_mcp = landmarks[PINKY_MCP]
        middle_mcp = landmarks[MIDDLE_FINGER_MCP]

        log.debug("DETECT | wrist=(%.3f,%.3f)  index_mcp=(%.3f,%.3f)"
                  "  pinky_mcp=(%.3f,%.3f)  middle_mcp=(%.3f,%.3f)",
                  wrist.x, wrist.y,
                  index_mcp.x, index_mcp.y,
                  pinky_mcp.x, pinky_mcp.y,
                  middle_mcp.x, middle_mcp.y)

        def _point(index):
            landmark = landmarks[index]
            return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

        wrist_pt = _point(WRIST)
        index_pt = _point(INDEX_FINGER_MCP)
        middle_pt = _point(MIDDLE_FINGER_MCP)
        pinky_pt = _point(PINKY_MCP)

        palm_width = float(np.linalg.norm(index_pt - pinky_pt))
        if palm_width < MIN_PALM_WIDTH:
            log.warning("DETECT | palm too small width=%.1fpx", palm_width)
            return None

        angle = float(np.degrees(np.arctan2(pinky_pt[1] - index_pt[1], pinky_pt[0] - index_pt[0])))
        center = (wrist_pt + middle_pt) / 2.0
        rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, rotation, (w, h), flags=cv2.INTER_LINEAR)

        half = (palm_width * PALM_ROI_SCALE) / 2.0
        cx, cy = float(center[0]), float(center[1])
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))

        log.debug(
            "DETECT | palm_width=%.1fpx angle=%.1f center=(%.1f,%.1f) box=[%d:%d, %d:%d]",
            palm_width,
            angle,
            cx,
            cy,
            y1,
            y2,
            x1,
            x2,
        )

        roi = rotated[y1:y2, x1:x2]
        if roi.size == 0:
            log.warning("DETECT | ROI is empty after crop — hand may be at image edge")
            return None

        log.info("DETECT | ROI extracted  shape=%s", roi.shape)
        return roi

    def apply_clahe(self, gray_img: np.ndarray) -> np.ndarray:
        return self.clahe.apply(gray_img)

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        enhanced = self.apply_clahe(gray)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)

    def get_embedding(self, frame_rgb: np.ndarray, tta_enabled: bool = False):
        roi = self.extract_palm_roi(frame_rgb)
        if roi is None:
            return None
        processed = self.preprocess_roi(roi)
        return self._run_inference_with_optional_tta(processed, tta_enabled=tta_enabled)

    def get_embedding_from_notebook_frame(self, frame_rgb: np.ndarray, tta_enabled: bool = False):
        # Compatibility wrapper: the new embedding model was trained on MediaPipe ROI,
        # not the old rembg/FFT notebook preprocessing path.
        return self.get_embedding(frame_rgb, tta_enabled=tta_enabled)

    def get_registration_guidance_metrics(self, frame_rgb: np.ndarray, previous_metrics: dict | None = None):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        base = {
            "hand_detected": False,
            "hand_clipped": True,
            "height_ratio": 0.0,
            "rotation_degrees": 999.0,
            "center_x_ratio": 0.0,
            "brightness": float(gray.mean()),
            "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "steady": False,
        }
        if self._hand_landmarker is None:
            return base

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._hand_landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return base

        landmarks = result.hand_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        h, w = gray.shape[:2]
        pad_x = int((max_x - min_x) * w * 0.12)
        pad_y = int((max_y - min_y) * h * 0.12)
        x1 = max(0, int(min_x * w) - pad_x)
        y1 = max(0, int(min_y * h) - pad_y)
        x2 = min(w, int(max_x * w) + pad_x)
        y2 = min(h, int(max_y * h) + pad_y)
        hand_gray = gray[y1:y2, x1:x2]
        if hand_gray.size:
            brightness = float(hand_gray.mean())
            blur_score = float(cv2.Laplacian(hand_gray, cv2.CV_64F).var())
        else:
            brightness = base["brightness"]
            blur_score = base["blur_score"]

        index_mcp = landmarks[INDEX_FINGER_MCP]
        pinky_mcp = landmarks[PINKY_MCP]
        rotation_degrees = float(
            np.degrees(np.arctan2(pinky_mcp.y - index_mcp.y, pinky_mcp.x - index_mcp.x))
        )

        metrics = {
            "hand_detected": True,
            "hand_clipped": min_x < 0.03 or max_x > 0.97 or min_y < 0.03 or max_y > 0.97,
            "height_ratio": float(max_y - min_y),
            "rotation_degrees": rotation_degrees,
            "center_x_ratio": float((min_x + max_x) / 2),
            "brightness": brightness,
            "blur_score": blur_score,
            "steady": False,
        }
        if previous_metrics and previous_metrics.get("hand_detected"):
            metrics["steady"] = (
                abs(metrics["center_x_ratio"] - previous_metrics.get("center_x_ratio", 0.0)) <= 0.03
                and abs(metrics["height_ratio"] - previous_metrics.get("height_ratio", 0.0)) <= 0.04
                and abs(metrics["rotation_degrees"] - previous_metrics.get("rotation_degrees", 999.0)) <= 4.0
            )
        return metrics

    def get_embedding_from_roi(
        self,
        roi_rgb: np.ndarray,
        rotation_angle: float = 0.0,
        tta_enabled: bool = False,
    ):
        """Process a pre-extracted, already-aligned palm ROI from the browser."""
        if roi_rgb is None or roi_rgb.size == 0:
            log.warning("DETECT | received empty ROI from client")
            return None

        log.info("DETECT | using client-side ROI  shape=%s", roi_rgb.shape)
        processed = self.preprocess_roi(roi_rgb)
        return self._run_inference_with_optional_tta(processed, tta_enabled=tta_enabled)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return (vector / norm).astype(np.float32)

    def _rotate_model_input(self, processed: np.ndarray, angle_degrees: float) -> np.ndarray:
        if abs(angle_degrees) < 1e-6:
            return processed
        h, w = processed.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_degrees, 1.0)
        return cv2.warpAffine(
            processed,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ).astype(np.float32)

    def _run_inference_with_optional_tta(self, processed: np.ndarray, tta_enabled: bool = False) -> np.ndarray:
        if not tta_enabled:
            return self._run_inference(processed)
        embeddings = [self._run_inference(self._rotate_model_input(processed, angle)) for angle in TTA_ROTATIONS]
        return self._normalize_embedding(np.mean(embeddings, axis=0))

    def _run_inference(self, processed: np.ndarray) -> np.ndarray:
        if self.interpreter is None:
            raise RuntimeError("TFLite model not loaded")
        if self._output_index is None:
            raise RuntimeError("TFLite model output tensor not configured")

        input_data = np.expand_dims(processed.astype(np.float32), axis=0)
        self.interpreter.set_tensor(self._input_index, input_data)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self._output_index)[0]
        return self._normalize_embedding(embedding)

    def compute_similarity(self, embedding: np.ndarray, stored_embeddings: list, threshold: float) -> dict:
        if not stored_embeddings:
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.0,
                "closest_match": None,
                "user_id": None,
            }

        query = self._normalize_embedding(embedding)
        best_score = -1.0
        best_match = None
        best_user_id = None

        for entry in stored_embeddings:
            stored = np.asarray(entry["embedding"], dtype=np.float32).reshape(-1)
            if stored.shape != query.shape:
                log.warning(
                    "MATCH | skipped incompatible embedding dim stored=%d query=%d user=%s",
                    stored.shape[0],
                    query.shape[0],
                    entry.get("name"),
                )
                continue
            stored = self._normalize_embedding(stored)
            score = float(np.dot(query, stored))
            if score > best_score:
                best_score = score
                best_match = entry["name"]
                best_user_id = entry["id"]

        if best_score < 0.0:
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.0,
                "closest_match": None,
                "user_id": None,
            }

        if best_score >= threshold:
            return {
                "status": "ALLOWED",
                "name": best_match,
                "similarity": round(best_score, 4),
                "closest_match": best_match,
                "user_id": best_user_id,
            }

        return {
            "status": "DENIED",
            "name": "Unknown",
            "similarity": round(best_score, 4),
            "closest_match": best_match,
            "user_id": None,
        }

    def close(self):
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
