"""Legacy rembg/FFT ROI preprocessing kept for notebook reference only.

Do not use this path for production registration or recognition. The active
runtime implementation is MediaPipe ROI extraction in app.palm_processor.
"""

from dataclasses import dataclass
import logging

import cv2
import numpy as np

from app.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMG_SIZE, NOTEBOOK_REMBG_MODEL

log = logging.getLogger("palmgate")


@dataclass(frozen=True)
class NotebookPreprocessResult:
    roi: np.ndarray
    model_input: np.ndarray
    bbox: tuple[int, int, int, int]
    rotation_degrees: float
    roi_size: int
    contour_area: float


class NotebookPreprocessor:
    def __init__(self, rembg_enabled: bool = True, rembg_model: str = NOTEBOOK_REMBG_MODEL):
        self.rembg_enabled = rembg_enabled
        self.rembg_model = rembg_model
        self._rembg_session = None
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID,
        )

    def _get_rembg_session(self):
        if self._rembg_session is None:
            from rembg import new_session

            self._rembg_session = new_session(
                self.rembg_model,
                providers=["CPUExecutionProvider"],
            )
        return self._rembg_session

    def preprocess_roi_to_model_input(self, roi: np.ndarray) -> np.ndarray:
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        enhanced = self.clahe.apply(gray.astype(np.uint8))
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)

    def _prepare_hand_mask_input(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, dsize=(480, 640), interpolation=cv2.INTER_CUBIC)
        if self.rembg_enabled:
            from PIL import Image
            from rembg import remove

            output = np.array(remove(Image.fromarray(resized), session=self._get_rembg_session()))
            if output.ndim == 3 and output.shape[2] == 4:
                alpha = output[:, :, 3]
                rgb = output[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return np.where(alpha > 0, gray, 0).astype(np.uint8)
            if output.ndim == 3:
                return cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
            return output.astype(np.uint8)
        return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    def _threshold_hand(self, gray: np.ndarray) -> np.ndarray:
        padded = cv2.copyMakeBorder(
            gray,
            top=80,
            bottom=80,
            left=80,
            right=80,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        blurred = cv2.GaussianBlur(padded, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def _find_largest_contour(self, mask: np.ndarray):
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        erosion = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _calculate_roi(
        self,
        hand_mask: np.ndarray,
        gray_image: np.ndarray,
        contour,
        low_freq: int = 50,
        padding: int = 80,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
        gray_padded = cv2.copyMakeBorder(
            gray_image,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        moments = cv2.moments(hand_mask)
        if moments["m00"] == 0:
            return None

        x_c = int(moments["m10"] / moments["m00"])
        y_c = int(moments["m01"] / moments["m00"])

        points = contour.reshape(-1, 2)
        if len(points) < low_freq + 2:
            return None

        left_id = np.argmin(points.sum(-1))
        points = np.concatenate([points[left_id:], points[:left_id]])
        dist_c = np.sqrt(np.square(points - [x_c, y_c]).sum(-1))
        freq = np.fft.rfft(dist_c)
        filtered = np.concatenate([freq[:low_freq], np.zeros(len(freq) - low_freq)])
        smooth = np.fft.irfft(filtered, n=len(dist_c))
        derivative = np.diff(smooth)
        sign_change = np.diff(np.sign(derivative))
        minima_idx = np.where(sign_change > 0)[0] + 1
        if len(minima_idx) < 2:
            return None

        minima_points = points[minima_idx]
        minima_dists = smooth[minima_idx]
        depth_order = np.argsort(minima_dists)
        v1 = minima_points[depth_order[0]]

        min_separation = max(30, len(points) * 0.05)
        v2 = None
        for index in depth_order[1:]:
            candidate = minima_points[index]
            if np.sqrt(np.sum((candidate - v1) ** 2)) >= min_separation:
                v2 = candidate
                break
        if v2 is None:
            return None
        if v1[0] > v2[0]:
            v1, v2 = v2, v1

        theta = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
        center = ((v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0)
        rotation_degrees = float(np.degrees(theta))
        rotation = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
        h, w = hand_mask.shape[:2]
        rotated_gray = cv2.warpAffine(gray_image, rotation, (w, h))

        v1_r = (np.dot(rotation[:, :2], v1) + rotation[:, 2]).astype(int)
        v2_r = (np.dot(rotation[:, :2], v2) + rotation[:, 2]).astype(int)
        center_mass = np.array([x_c, y_c], dtype=float)
        center_mass_r = (np.dot(rotation[:, :2], center_mass) + rotation[:, 2]).astype(int)

        roi_size = int(np.sqrt(np.sum((v2_r - v1_r) ** 2)))
        if roi_size <= 0:
            return None

        mid_x = (v1_r[0] + v2_r[0]) // 2
        mid_y = (v1_r[1] + v2_r[1]) // 2
        palm_below = center_mass_r[1] >= mid_y

        if palm_below:
            ux, uy = mid_x - roi_size // 2, mid_y
            lx, ly = mid_x + roi_size // 2, mid_y + roi_size
        else:
            ux, uy = mid_x - roi_size // 2, mid_y - roi_size
            lx, ly = mid_x + roi_size // 2, mid_y

        ux, uy = max(0, ux), max(0, uy)
        lx, ly = min(w, lx), min(h, ly)
        roi = rotated_gray[uy:ly, ux:lx]
        if roi.size == 0:
            return None
        return roi, (ux, uy, lx, ly), rotation_degrees

    def extract_full_hand_roi(self, frame_rgb: np.ndarray) -> NotebookPreprocessResult | None:
        if frame_rgb.size == 0 or float(frame_rgb.mean()) < 5:
            log.warning("NOTEBOOK | frame rejected: empty or too dark")
            return None

        gray = self._prepare_hand_mask_input(frame_rgb)
        mask = self._threshold_hand(gray)
        contour = self._find_largest_contour(mask)
        if contour is None:
            log.warning("NOTEBOOK | no contour found in mask")
            return None

        log.info("NOTEBOOK | contour found, area=%.0f", cv2.contourArea(contour))

        calculated = self._calculate_roi(mask, gray, contour)
        if calculated is None:
            log.warning("NOTEBOOK | ROI calculation failed (FFT valley detection)")
            return None

        roi, bbox, rotation_degrees = calculated
        h, w = roi.shape[:2]
        log.info("NOTEBOOK | ROI extracted: %dx%d, rotation=%.1f°", w, h, rotation_degrees)
        if h < 50 or w < 50:
            log.warning("NOTEBOOK | ROI too small: %dx%d", w, h)
            return None

        model_input = self.preprocess_roi_to_model_input(roi)
        return NotebookPreprocessResult(
            roi=roi,
            model_input=model_input,
            bbox=bbox,
            rotation_degrees=rotation_degrees,
            roi_size=min(h, w),
            contour_area=float(cv2.contourArea(contour)),
        )
