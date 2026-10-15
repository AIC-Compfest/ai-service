"""Geometry-preserving, opt-in preparation variants for raster OCR."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_preprocessor import ImagePreprocessor


@dataclass(frozen=True)
class PreparedImage:
    """An OCR image and the transform that maps its coordinates to the source."""

    image: np.ndarray
    operations: tuple[str, ...]
    inverse_transform: np.ndarray


class OCRPreprocessor:
    """Build a conservative OCR variant without encoding document-specific rules."""

    def __init__(self, min_side_for_upscale: int, max_upscale: float):
        self.min_side_for_upscale = min_side_for_upscale
        self.max_upscale = max_upscale

    def prepare(self, image: np.ndarray) -> PreparedImage:
        """Deskew, normalize contrast, and optionally upscale while tracking geometry."""
        height, width = image.shape[:2]
        quality = ImagePreprocessor.estimate_quality(image)
        angle = float(quality["skew_angle"])
        prepared = image.copy()
        forward = np.eye(3, dtype=np.float32)
        operations: list[str] = []

        if abs(angle) >= 0.5:
            center = (width / 2.0, height / 2.0)
            affine = cv2.getRotationMatrix2D(center, angle, 1.0)
            prepared = cv2.warpAffine(
                prepared,
                affine,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            forward = self._to_homography(affine) @ forward
            operations.append("deskew")

        if quality["is_low_contrast"] or quality["is_blurry"]:
            prepared = self._normalize_document(prepared)
            operations.append("contrast_normalization")

        shortest_side = min(height, width)
        if shortest_side and shortest_side < self.min_side_for_upscale:
            scale = min(self.max_upscale, self.min_side_for_upscale / shortest_side)
            if scale > 1.0:
                prepared = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                forward = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32) @ forward
                operations.append("adaptive_upscale")

        return PreparedImage(prepared, tuple(operations), np.linalg.inv(forward))

    @staticmethod
    def _normalize_document(image: np.ndarray) -> np.ndarray:
        """Reduce uneven illumination while retaining colored document content."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _to_homography(affine: np.ndarray) -> np.ndarray:
        return np.vstack((affine, np.array([0, 0, 1], dtype=np.float32)))
