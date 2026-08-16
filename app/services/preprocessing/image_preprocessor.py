"""
image_preprocessor.py

OpenCV & PIL-based image preprocessing and quality analysis for document images.
Operations:
- Quality estimation: blur score (Laplacian variance), brightness, contrast.
- Orientation and skew detection (Hough transform angle detection).
- Non-destructive image normalization.
"""

from __future__ import annotations
import math
import cv2
import numpy as np
from PIL import Image
import io


class ImagePreprocessor:
    """Non-destructive image preprocessor and quality analyzer."""

    @staticmethod
    def estimate_quality(cv_img: np.ndarray) -> dict[str, float]:
        """
        Estimate image quality metrics:
        - blur_score: Laplacian variance (>100 is sharp, <50 is blurry)
        - brightness: Mean pixel luminance [0, 255]
        - contrast: Standard deviation of luminance
        - skew_angle: Estimated skew angle in degrees (-45 to 45)
        """
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img

        # 1. Blur score
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness & Contrast
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # 3. Skew Angle Detection via MinAreaRect of thresholded text contours
        skew_angle = ImagePreprocessor._detect_skew_angle(gray)

        return {
            "blur_score": round(laplacian_var, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "skew_angle": round(skew_angle, 2),
            "is_blurry": laplacian_var < 50.0,
            "is_low_contrast": contrast < 30.0,
        }

    @staticmethod
    def _detect_skew_angle(gray: np.ndarray) -> float:
        """Estimate skew angle using binary threshold and minAreaRect on text regions."""
        try:
            # Invert colors (text is white on black background)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if coords.shape[0] < 50:
                return 0.0

            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Clamp small noise
            if abs(angle) < 0.3:
                return 0.0
            return float(angle)
        except Exception:
            return 0.0

    @staticmethod
    def deskew_image(cv_img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees) around center to correct skew."""
        if abs(angle) < 0.5 or abs(angle) > 45.0:
            return cv_img

        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            cv_img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """Convert raw image bytes to OpenCV BGR ndarray."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback PIL
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def cv2_to_bytes(cv_img: np.ndarray, ext: str = ".jpg") -> bytes:
        """Encode OpenCV image back to bytes."""
        success, encoded_img = cv2.imencode(ext, cv_img)
        if not success:
            raise ValueError("Failed to encode image to bytes")
        return encoded_img.tobytes()
