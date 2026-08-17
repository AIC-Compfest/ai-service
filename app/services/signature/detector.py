"""
detector.py

Signature and Stamp Presence Detector — PRD §33, §15, §70, Context §24.
Uses Computer Vision (color space segmentation, stroke density, contour curvature,
and printed-text rejection) to detect physical signatures and official stamps on PODs.

Non-Negotiable Safety Rules (PRD §33, §70):
1. Presence Only: Determines whether a physical wet signature and/or stamp appears in the ROI.
2. Zero Identity: Never claims or extracts signer identity, name, or biometrics.
3. Zero Authenticity: Never claims authenticity, legal validity, or genuineness.
4. Unknown is not False: Low-quality/degraded scans trigger human review rather than false accusations.
"""

from __future__ import annotations
import math
import cv2
import numpy as np
from app.core.logging import get_logger
from app.schemas.evidence import BoundingBox
from app.schemas.extraction import SignatureDetectionResponse
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class SignatureDetector:
    """
    Computer Vision based Multi-Signal Signature and Stamp Presence Detector.
    Strictly answers: 'Apakah terdapat tanda tangan dan/atau stempel fisik pada area dokumen?'
    """

    def detect(
        self,
        document_id: str,
        image_bytes: bytes,
        target_region: tuple[float, float, float, float] | None = None,
    ) -> SignatureDetectionResponse:
        """
        Detect handwritten signature and official stamp presence in image bytes.
        target_region: Optional normalized bbox (ymin, xmin, ymax, xmax) in [0, 1].
        """
        if not image_bytes or len(image_bytes) < 10:
            return SignatureDetectionResponse(
                document_id=document_id,
                signature_present=False,
                confidence=0.0,
                bbox=None,
                stamp_present=False,
                stamp_confidence=0.0,
                stamp_bbox=None,
                requires_human_review=True,
                is_unknown=True,
            )

        try:
            cv_img = ImagePreprocessor.bytes_to_cv2(image_bytes)
        except Exception as e:
            logger.warning("signature_detector.cv2_decode_failed", document_id=document_id, error=str(e))
            return SignatureDetectionResponse(
                document_id=document_id,
                signature_present=False,
                confidence=0.0,
                bbox=None,
                stamp_present=False,
                stamp_confidence=0.0,
                stamp_bbox=None,
                requires_human_review=True,
                is_unknown=True,
            )

        if cv_img is None or cv_img.size == 0:
            return SignatureDetectionResponse(
                document_id=document_id,
                signature_present=False,
                confidence=0.0,
                bbox=None,
                stamp_present=False,
                stamp_confidence=0.0,
                stamp_bbox=None,
                requires_human_review=True,
                is_unknown=True,
            )

        h, w = cv_img.shape[:2]

        # Check for very low resolution image
        is_low_res = (w < 150 or h < 150)

        # Crop to target region if specified, else focus on bottom 50% signature zone
        if target_region:
            y1 = max(0, int(target_region[0] * h))
            x1 = max(0, int(target_region[1] * w))
            y2 = min(h, int(target_region[2] * h))
            x2 = min(w, int(target_region[3] * w))
            if y2 <= y1 or x2 <= x1:
                roi = cv_img
                offset_x, offset_y = 0, 0
            else:
                roi = cv_img[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
        else:
            y1 = int(0.50 * h)
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
                stamp_present=False,
                stamp_confidence=0.0,
                stamp_bbox=None,
                requires_human_review=False,
            )

        # ====================================================================
        # 1. OFFICIAL STAMP DETECTION (Multi-Color HSV Segmentation)
        # ====================================================================
        stamp_present, stamp_conf, stamp_bbox_norm = self._detect_stamp(roi, w, h, offset_x, offset_y)

        # ====================================================================
        # 2. HANDWRITTEN SIGNATURE DETECTION (Pen Strokes & Text Rejection)
        # ====================================================================
        sig_present, sig_conf, sig_bbox_norm, is_uncertain = self._detect_handwritten_signature(
            roi, w, h, offset_x, offset_y, is_low_res
        )

        requires_review = is_uncertain or is_low_res or (sig_present and sig_conf < 0.70) or (stamp_present and stamp_conf < 0.70)

        logger.info(
            "signature_detector.complete",
            document_id=document_id,
            signature_present=sig_present,
            signature_confidence=round(sig_conf, 3),
            stamp_present=stamp_present,
            stamp_confidence=round(stamp_conf, 3),
            requires_human_review=requires_review,
        )

        return SignatureDetectionResponse(
            document_id=document_id,
            signature_present=sig_present,
            confidence=sig_conf,
            bbox=sig_bbox_norm,
            stamp_present=stamp_present,
            stamp_confidence=stamp_conf,
            stamp_bbox=stamp_bbox_norm,
            requires_human_review=requires_review,
            is_unknown=is_low_res and not sig_present and not stamp_present,
        )

    # -----------------------------------------------------------------------
    # Private Stamp Detection Engine
    # -----------------------------------------------------------------------

    def _detect_stamp(
        self, roi: np.ndarray, doc_w: int, doc_h: int, offset_x: int, offset_y: int
    ) -> tuple[bool, float, BoundingBox | None]:
        """
        Detects official circular, elliptical, or rectangular stamps using color masks
        and connected component geometry. Rejects colored header logos and decorative lines.
        """
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 10 or roi_w < 10:
            return False, 0.0, None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # A. Blue/Purple Ink Mask (Common Indonesian 3PL Stamps & Seals)
        lower_blue = np.array([90, 35, 35])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # B. Red/Magenta/Orange Ink Mask (Official Company Seals)
        lower_red1 = np.array([0, 45, 45])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([155, 45, 45])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combined Stamp Color Mask
        color_mask = cv2.bitwise_or(blue_mask, red_mask)

        # Morphological Closing to connect stamped letters and outer rings
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find Connected Components / Contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_stamp_bbox = None
        best_score = 0.0
        min_stamp_area = max(200, int(roi_h * roi_w * 0.005))
        max_stamp_area = int(roi_h * roi_w * 0.45)

        for c in contours:
            area = cv2.contourArea(c)
            if min_stamp_area <= area <= max_stamp_area:
                cx, cy, cw, ch = cv2.boundingRect(c)
                aspect = cw / float(ch) if ch > 0 else 0.0

                # Stamps are typically circular / square / moderately elliptical (aspect 0.5 to 2.2)
                if 0.45 <= aspect <= 2.5:
                    # Spatial density: ratio of colored pixels inside the bounding box
                    stamp_crop = color_mask[cy : cy + ch, cx : cx + cw]
                    pixel_count = cv2.countNonZero(stamp_crop)
                    density = pixel_count / float(cw * ch) if (cw * ch) > 0 else 0.0

                    # Stamps have distributed ink density (0.08 to 0.75) unlike solid blocks or thin noise
                    if 0.06 <= density <= 0.85:
                        # Solidity / Roundness heuristic
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / float(hull_area) if hull_area > 0 else 0.0

                        # Calculate score based on area, density, and aspect balance
                        score = min(0.95, 0.70 + (density * 0.20) + (solidity * 0.10))
                        if score > best_score:
                            best_score = score
                            best_stamp_bbox = (offset_x + cx, offset_y + cy, offset_x + cx + cw, offset_y + cy + ch)

        if best_stamp_bbox and best_score >= 0.70:
            bx1, by1, bx2, by2 = best_stamp_bbox
            bbox_model = BoundingBox(
                x1=round(max(0.0, min(1.0, bx1 / float(doc_w))), 4),
                y1=round(max(0.0, min(1.0, by1 / float(doc_h))), 4),
                x2=round(max(0.0, min(1.0, bx2 / float(doc_w))), 4),
                y2=round(max(0.0, min(1.0, by2 / float(doc_h))), 4),
            )
            return True, round(best_score, 3), bbox_model

        return False, 0.0, None

    # -----------------------------------------------------------------------
    # Private Handwritten Signature Detection Engine
    # -----------------------------------------------------------------------

    def _detect_handwritten_signature(
        self, roi: np.ndarray, doc_w: int, doc_h: int, offset_x: int, offset_y: int, is_low_res: bool
    ) -> tuple[bool, float, BoundingBox | None, bool]:
        """
        Detects freehand wet ink pen strokes (black, blue, dark ink).
        Implements printed-text rejection using stroke curvature, aspect ratio,
        and contour continuity.
        """
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 10 or roi_w < 10:
            return False, 0.0, None, False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement & noise reduction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Edge detection & adaptive thresholding
        edges = cv2.Canny(blurred, 40, 140)

        # Remove long horizontal & vertical table lines to isolate handwriting
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
        vert_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vert_kernel)
        lines_mask = cv2.bitwise_or(horiz_lines, vert_lines)
        clean_edges = cv2.bitwise_and(edges, cv2.bitwise_not(lines_mask))

        # Check total stroke edge pixels
        total_edge_pixels = int(cv2.countNonZero(clean_edges))
        roi_pixel_area = roi_h * roi_w
        edge_density = total_edge_pixels / float(roi_pixel_area) if roi_pixel_area > 0 else 0.0

        # If area has virtually no strokes (< 0.003 density or < 50 pixels), it is an empty blank box
        if total_edge_pixels < 40 or edge_density < 0.002:
            return False, 0.90, None, False

        # Find stroke contours
        contours, _ = cv2.findContours(clean_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_sig_bbox = None
        max_sig_score = 0.0
        has_uncertain_strokes = False

        min_sig_area = max(100, int(roi_pixel_area * 0.003))
        max_sig_area = int(roi_pixel_area * 0.40)

        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_sig_area:
                cx, cy, cw, ch = cv2.boundingRect(c)
                aspect = cw / float(ch) if ch > 0 else 0.0

                # Printed Text Rejection Criteria:
                # 1. Regular short rectangular characters (cw < 15, ch < 15, area < 180)
                # 2. Straight horizontal bounding lines (aspect > 8.0 with minimal height)
                # 3. Grid-aligned text lines
                if cw < 18 and ch < 18 and area < 200:
                    continue

                if aspect > 7.0 and ch < 8:
                    continue

                # Signatures are typically wider than tall, dynamic curves (aspect 1.0 to 6.0)
                if 0.9 <= aspect <= 6.5 and area <= max_sig_area:
                    # Stroke complexity / curvature analysis: perimeter vs area
                    peri = cv2.arcLength(c, True)
                    # Hand-drawn curves have high perimeter-to-area ratio
                    complexity = (peri * peri) / (area + 1e-5)

                    if complexity > 18.0:
                        # Calculate calibrated confidence
                        # Scale based on edge density and stroke continuity
                        confidence = 0.75 + min(0.20, (area / float(min_sig_area * 8)) * 0.15)
                        if aspect >= 1.3:
                            confidence += 0.05
                        confidence = min(0.95, confidence)

                        if confidence > max_sig_score:
                            max_sig_score = confidence
                            best_sig_bbox = (offset_x + cx, offset_y + cy, offset_x + cx + cw, offset_y + cy + ch)
                    else:
                        has_uncertain_strokes = True

        # Blue pen stroke detection in ROI (for wet blue ink handwriting)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_pen_mask = cv2.inRange(hsv_roi, np.array([95, 40, 40]), np.array([130, 255, 255]))
        blue_pen_pixels = int(cv2.countNonZero(blue_pen_mask))

        if blue_pen_pixels > 80:
            blue_contours, _ = cv2.findContours(blue_pen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for bc in blue_contours:
                b_area = cv2.contourArea(bc)
                if b_area > 80:
                    bcx, bcy, bcw, bch = cv2.boundingRect(bc)
                    b_aspect = bcw / float(bch) if bch > 0 else 0.0
                    if 0.8 <= b_aspect <= 6.0:
                        max_sig_score = max(max_sig_score, 0.92)
                        if not best_sig_bbox or b_area > cv2.contourArea(cv2.convexHull(bc)):
                            best_sig_bbox = (offset_x + bcx, offset_y + bcy, offset_x + bcx + bcw, offset_y + bcy + bch)

        # Decision
        if best_sig_bbox and max_sig_score >= 0.70:
            bx1, by1, bx2, by2 = best_sig_bbox
            bbox_model = BoundingBox(
                x1=round(max(0.0, min(1.0, bx1 / float(doc_w))), 4),
                y1=round(max(0.0, min(1.0, by1 / float(doc_h))), 4),
                x2=round(max(0.0, min(1.0, bx2 / float(doc_w))), 4),
                y2=round(max(0.0, min(1.0, by2 / float(doc_h))), 4),
            )
            return True, round(max_sig_score, 3), bbox_model, False
        elif has_uncertain_strokes or (total_edge_pixels > 120 and not best_sig_bbox):
            # Noisy or degraded marks in signature zone that cannot be decisively approved
            return False, 0.55, None, True
        else:
            # Clean empty area
            return False, 0.90, None, False
