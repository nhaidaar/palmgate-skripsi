import cv2
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    HAND_LANDMARKER_PATH,
    IMG_SIZE,
    MODEL_PATH,
    NOTEBOOK_REMBG_ENABLED,
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
        self._gap_output_index = None
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

    def _load_model(self, model_path):
        kwargs = {"model_path": str(model_path), "experimental_preserve_all_tensors": True}
        try:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(num_threads=4, **kwargs)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(num_threads=4, **kwargs)
        except TypeError:
            # Older tflite_runtime without num_threads
            try:
                from tflite_runtime.interpreter import Interpreter
                self.interpreter = Interpreter(**kwargs)
            except ImportError:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(**kwargs)

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        self._input_index = input_details[0]["index"]

        # Find the GlobalAveragePooling2D output (1280-dim) before the Dense head.
        # experimental_preserve_all_tensors=True is required to read this after invoke().
        tensor_details = self.interpreter.get_tensor_details()
        gap_candidates = []
        for t in tensor_details:
            shape = t.get("shape", [])
            if hasattr(shape, "tolist"):
                shape = shape.tolist()
            if shape == [1, 1280]:
                gap_candidates.append(t["index"])

        if gap_candidates:
            # Use the last 1280-dim tensor found (closest to the output head)
            self._gap_output_index = gap_candidates[-1]
            log.info("MODEL | GAP embedding tensor index=%d  (found %d candidate/s)",
                     self._gap_output_index, len(gap_candidates))
        else:
            # Fallback: use the final softmax output as a coarse embedding
            output_details = self.interpreter.get_output_details()
            self._gap_output_index = output_details[0]["index"]
            log.warning("MODEL | GAP tensor not found — using softmax output (dim=%s) as embedding",
                        output_details[0]["shape"].tolist())

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

        wrist      = landmarks[WRIST]
        index_mcp  = landmarks[INDEX_FINGER_MCP]
        pinky_mcp  = landmarks[PINKY_MCP]
        middle_mcp = landmarks[MIDDLE_FINGER_MCP]

        log.debug("DETECT | wrist=(%.3f,%.3f)  index_mcp=(%.3f,%.3f)"
                  "  pinky_mcp=(%.3f,%.3f)  middle_mcp=(%.3f,%.3f)",
                  wrist.x, wrist.y,
                  index_mcp.x, index_mcp.y,
                  pinky_mcp.x, pinky_mcp.y,
                  middle_mcp.x, middle_mcp.y)

        # ── Rotation normalisation ────────────────────────────────
        # The training pipeline (calculate_roi in the notebook) rotates each
        # frame so the index-MCP → pinky-MCP knuckle line is horizontal before
        # cropping.  We replicate that here using the same landmarks so the ROI
        # fed to the model matches the distribution it was trained on.
        dx = (pinky_mcp.x - index_mcp.x) * w
        dy = (pinky_mcp.y - index_mcp.y) * h
        theta_deg = float(np.degrees(np.arctan2(dy, dx)))

        knuckle_cx = int((index_mcp.x + pinky_mcp.x) / 2 * w)
        knuckle_cy = int((index_mcp.y + pinky_mcp.y) / 2 * h)
        R = cv2.getRotationMatrix2D((knuckle_cx, knuckle_cy), theta_deg, 1.0)
        frame_rot = cv2.warpAffine(frame_rgb, R, (w, h), flags=cv2.INTER_LINEAR)

        log.debug("DETECT | knuckle rotation theta=%.1f°  center=(%d,%d)",
                  theta_deg, knuckle_cx, knuckle_cy)

        # Project landmark coords into the rotated frame
        def _rot_pt(lm):
            x_px = lm.x * w
            y_px = lm.y * h
            rx = R[0, 0] * x_px + R[0, 1] * y_px + R[0, 2]
            ry = R[1, 0] * x_px + R[1, 1] * y_px + R[1, 2]
            return rx, ry

        mid_rx, mid_ry = _rot_pt(middle_mcp)
        wrist_rx, wrist_ry = _rot_pt(wrist)
        idx_rx,  idx_ry  = _rot_pt(index_mcp)
        pnk_rx,  pnk_ry  = _rot_pt(pinky_mcp)

        cx = int(mid_rx)
        cy = int((mid_ry + wrist_ry) / 2)

        palm_width = abs(int(idx_rx - pnk_rx))
        roi_size = max(int(palm_width * 1.5), 60)
        half = roi_size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        log.debug("DETECT | palm_width=%dpx  roi_size=%dpx  center=(%d,%d)"
                  "  box=[%d:%d, %d:%d]",
                  palm_width, roi_size, cx, cy, y1, y2, x1, x2)

        roi = frame_rot[y1:y2, x1:x2]
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

    def get_embedding(self, frame_rgb: np.ndarray):
        roi = self.extract_palm_roi(frame_rgb)
        if roi is None:
            return None
        processed = self.preprocess_roi(roi)
        return self._run_inference(processed)

    def get_embedding_from_notebook_frame(self, frame_rgb: np.ndarray):
        result = self.notebook_preprocessor.extract_full_hand_roi(frame_rgb)
        if result is None:
            return None
        return self._run_inference(result.model_input)

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

    def get_embedding_from_roi(self, roi_rgb: np.ndarray, rotation_angle: float = 0.0):
        """Process a pre-extracted palm ROI, skipping hand detection.

        The browser already runs MediaPipe in VIDEO mode and can crop the ROI
        client-side, eliminating the slow server-side detection round-trip.

        rotation_angle: knuckle-line angle (degrees) computed by the browser from
        the index-MCP → pinky-MCP vector.  The ROI is rotated to cancel this tilt
        so the preprocessing matches the training pipeline (calculate_roi in the
        notebook always aligns the knuckle line to horizontal before cropping).
        """
        if roi_rgb is None or roi_rgb.size == 0:
            log.warning("DETECT | received empty ROI from client")
            return None

        log.info("DETECT | using client-side ROI  shape=%s  rotation=%.1f°",
                 roi_rgb.shape, rotation_angle)

        roi = roi_rgb
        if abs(rotation_angle) > 0.5:
            h, w = roi.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            R = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
            roi = cv2.warpAffine(roi, R, (w, h), flags=cv2.INTER_LINEAR)

        processed = self.preprocess_roi(roi)
        return self._run_inference(processed)

    def _run_inference(self, processed: np.ndarray) -> np.ndarray:
        if self.interpreter is None:
            raise RuntimeError("TFLite model not loaded")

        input_data = np.expand_dims(processed, axis=0)
        self.interpreter.set_tensor(self._input_index, input_data)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self._gap_output_index)[0]
        return embedding.copy()

    def compute_similarity(self, embedding: np.ndarray, stored_embeddings: list, threshold: float) -> dict:
        if not stored_embeddings:
            return {
                "status": "DENIED",
                "name": "Unknown",
                "similarity": 0.0,
                "closest_match": None,
                "user_id": None,
            }

        best_score = -1.0
        best_match = None
        best_user_id = None

        norm_a = np.linalg.norm(embedding)

        for entry in stored_embeddings:
            stored = entry["embedding"]
            norm_b = np.linalg.norm(stored)
            if norm_a == 0 or norm_b == 0:
                continue
            score = float(np.dot(embedding, stored) / (norm_a * norm_b))
            if score > best_score:
                best_score = score
                best_match = entry["name"]
                best_user_id = entry["id"]

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
