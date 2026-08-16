"""
detector.py

Signature and Stamp Presence Detector — PRD §33.
Uses OpenCV contour analysis, stroke density, and color mask detection to identify
presence of handwritten wet signatures and official company stamps.

Rules (PRD §33, §70):
- Only detects PRESENCE: signature_present = true / false.
- Never claims signature authenticity, identity, or ownership.
"""

from __future__ import annotations
import cv2
import numpy as np
from app.core.logging import get_logger
from app.schemas.evidence import BoundingBox
from app.schemas.extraction import SignatureDetectionResponse
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class SignatureDetector:
    """Computer Vision based Signature and Stamp presence detector."""

    def detect(self, document_id: str, image_bytes: bytes, target_region: tuple[float, float, float, float] | None = None) -> SignatureDetectionResponse:
        """
        Detect signature presence in image bytes.
        target_region: Optional normalized bbox (ymin, xmin, ymax, xmax) in [0, 1] to focus on bottom signature zone.
        """
        cv_img = ImagePreprocessor.bytes_to_cv2(image_bytes)
        h, w = cv_img.shape[:2]

        # Crop to target region if specified, else focus on bottom 45% where signatures normally appear
        if target_region:
            y1 = int(target_region[0] * h)
            x1 = int(target_region[1] * w)
            y2 = int(target_region[2] * h)
            x2 = int(target_region[3] * w)
            roi = cv_img[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            y1 = int(0.55 * h)
            x1 = 0
            y2 = h
            x2 = w
            roi = cv_img[y1:y2, x1:x2]
            offset_x, offset_y = 0, y1

        if roi.size == 0:
            return SignatureDetectionResponse(
                document_id=document_id,
                signature_present=False,
                confidence=0.0,
                bbox=None,
            )

        # 1. Color Mask for Blue/Purple Ink (Common in Indonesian Logistics Stamps & Pens)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Blue ink range: H 90-130, S 40-255, V 40-255
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = int(cv2.countNonZero(blue_mask))

        # 2. High-Frequency Stroke Density for Black/Dark Ink Signatures
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_pixels = int(cv2.countNonZero(edges))

        # 3. Find Contours with Signature-like Aspect Ratio & Density
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sig_candidate_bbox = None
        max_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if 300 < area < 40000:
                cx, cy, cw, ch = cv2.boundingRect(c)
                aspect = cw / float(ch)
                # Signatures are typically wider than tall (aspect 1.2 to 5.0)
                if 1.0 <= aspect <= 6.0 and area > max_area:
                    max_area = area
                    sig_candidate_bbox = (offset_x + cx, offset_y + cy, offset_x + cx + cw, offset_y + cy + ch)

        # Determine presence & confidence
        has_blue_ink = blue_pixels > 80
        has_stroke_pattern = edge_pixels > 250 and sig_candidate_bbox is not None

        if has_blue_ink or has_stroke_pattern:
            sig_present = True
            confidence = 0.92 if (has_blue_ink and has_stroke_pattern) else 0.85
            if sig_candidate_bbox:
                bx1, by1, bx2, by2 = sig_candidate_bbox
                bbox_model = BoundingBox(
                    x1=round(bx1 / w * 1000, 1),
                    y1=round(by1 / h * 1000, 1),
                    x2=round(bx2 / w * 1000, 1),
                    y2=round(by2 / h * 1000, 1),
                )
            else:
                bbox_model = BoundingBox(x1=200, y1=600, x2=800, y2=900)
        else:
            sig_present = False
            confidence = 0.88
            bbox_model = None

        logger.info(
            "signature_detected",
            document_id=document_id,
            signature_present=sig_present,
            confidence=confidence,
            blue_pixels=blue_pixels,
            edge_pixels=edge_pixels,
        )

        return SignatureDetectionResponse(
            document_id=document_id,
            signature_present=sig_present,
            confidence=confidence,
            bbox=bbox_model,
        )
