"""
test_phase10_signature_detection.py

Comprehensive Unit & Safety Test Suite for Phase 10:
SIGNATURE & STAMP PRESENCE DETECTION ENGINE (PRD §33, §15, §70, Context §24).

Verifies:
1. Multi-signal handwritten signature detection (Black & Blue pen).
2. Multi-color stamp presence detection (Blue/Purple & Red/Orange).
3. Simultaneous signature + stamp detection.
4. Printed text rejection.
5. Blank signature box rejection.
6. Degraded / noisy scan low-confidence & human review trigger.
7. Zero identity & zero authenticity safety invariant.
8. Bounding box normalization [0.0, 1.0].
9. Header logo rejection.
10. Low-resolution / corrupted image safety.
"""

import io
import cv2
import numpy as np
import pytest
from app.services.signature.detector import SignatureDetector
from app.schemas.extraction import SignatureDetectionResponse


# ============================================================================
# Synthetic Image Fixture Generators (Deterministic CV Test Fixtures)
# ============================================================================

def make_blank_pod_image(width=800, height=1000) -> np.ndarray:
    """Creates a clean white document with standard printed form layout."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Add title and borders
    cv2.putText(img, "BUKTI SERAH TERIMA PENGIRIMAN (POD)", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.line(img, (50, 80), (750, 80), (100, 100, 100), 2)
    # Add printed table metadata
    cv2.putText(img, "No. Resi / AWB : AWB-JKT-2026-9901", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "Penerima       : PT Logistik Sukses Makmur", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "Tanggal Terima : 2026-08-10 14:20 WIB", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    # Signature Box in lower half (y: 650 to 880, x: 450 to 750)
    cv2.rectangle(img, (450, 650), (750, 880), (180, 180, 180), 1)
    cv2.putText(img, "Tanda Tangan & Cap Penerima:", (460, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return img


def draw_curved_handwritten_signature(img: np.ndarray, color=(20, 20, 20), center=(600, 760)):
    """Draws organic dynamic Bézier/sinusoidal handwritten curves resembling a real wet signature."""
    cx, cy = center
    # Main dynamic looping signature stroke
    pts1 = np.array([
        [cx - 90, cy + 20],
        [cx - 70, cy - 40],
        [cx - 40, cy + 35],
        [cx - 10, cy - 30],
        [cx + 20, cy + 15],
        [cx + 50, cy - 25],
        [cx + 80, cy + 10],
        [cx + 100, cy - 10],
    ], np.int32)
    cv2.polylines(img, [pts1], False, color, 2, cv2.LINE_AA)

    # Big initial loop
    cv2.ellipse(img, (cx - 65, cy - 10), (25, 40), 35, 0, 360, color, 2, cv2.LINE_AA)
    # Underline flourish
    cv2.line(img, (cx - 85, cy + 45), (cx + 105, cy + 30), color, 2, cv2.LINE_AA)


def draw_official_stamp(img: np.ndarray, stamp_type="BLUE_PURPLE", center=(620, 770)):
    """Draws an official double-ring company stamp with text in standard Indonesian logistics ink."""
    cx, cy = center
    if stamp_type == "BLUE_PURPLE":
        color = (180, 40, 60)  # BGR for blue/purple stamp
    elif stamp_type == "RED_ORANGE":
        color = (30, 40, 210)  # BGR for official red/vermilion seal
    else:
        color = (140, 50, 50)

    # Outer and Inner Rings
    cv2.circle(img, (cx, cy), 52, color, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 42, color, 1, cv2.LINE_AA)
    # Center star and stamp text
    cv2.putText(img, "* PT LOGISTIK *", (cx - 46, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(img, "RECEIVED", (cx - 36, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 2, cv2.LINE_AA)
    cv2.putText(img, "JAKARTA", (cx - 30, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def img_to_bytes(img: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


# ============================================================================
# Test Cases
# ============================================================================

class TestPhase10SignatureDetection:
    """Phase 10 — Signature & Stamp Presence Detection Test Suite."""

    @pytest.fixture(autouse=True)
    def setup_detector(self):
        self.detector = SignatureDetector()

    def test_clean_wet_signature_detected(self):
        """1. Clean POD with dark pen signature -> signature_present == True."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(15, 15, 15), center=(600, 760))
        res = self.detector.detect("DOC-TEST-001", img_to_bytes(img))

        assert res.signature_present is True
        assert res.confidence >= 0.75
        assert res.bbox is not None
        assert 0.0 <= res.bbox.x1 < res.bbox.x2 <= 1.0
        assert 0.0 <= res.bbox.y1 < res.bbox.y2 <= 1.0

    def test_blue_stamp_presence_detected(self):
        """2. Blue/purple company stamp -> stamp_present == True."""
        img = make_blank_pod_image()
        draw_official_stamp(img, stamp_type="BLUE_PURPLE", center=(600, 760))
        res = self.detector.detect("DOC-TEST-002", img_to_bytes(img))

        assert res.stamp_present is True
        assert res.stamp_confidence >= 0.70
        assert res.stamp_bbox is not None
        assert 0.0 <= res.stamp_bbox.x1 < res.stamp_bbox.x2 <= 1.0

    def test_red_stamp_presence_detected(self):
        """3. Red/orange company seal -> stamp_present == True."""
        img = make_blank_pod_image()
        draw_official_stamp(img, stamp_type="RED_ORANGE", center=(580, 750))
        res = self.detector.detect("DOC-TEST-003", img_to_bytes(img))

        assert res.stamp_present is True
        assert res.stamp_confidence >= 0.70
        assert res.stamp_bbox is not None

    def test_blank_signature_box_returns_false(self):
        """4. Clearly empty signature zone -> signature_present == False, stamp_present == False."""
        img = make_blank_pod_image()
        res = self.detector.detect("DOC-TEST-004", img_to_bytes(img))

        assert res.signature_present is False
        assert res.stamp_present is False
        assert res.bbox is None
        assert res.stamp_bbox is None

    def test_printed_text_not_misclassified_as_signature(self):
        """5. Printed standard text in box should NOT be classified as confident signature."""
        img = make_blank_pod_image()
        # Add straight printed computer font text inside the signature box
        cv2.putText(img, "TANGGAL PENERIMAAN DOKUMEN", (470, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        cv2.putText(img, "STATUS: DELIVERED TO CONVOY", (470, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        res = self.detector.detect("DOC-TEST-005", img_to_bytes(img))

        # Printed text must not achieve high-confidence signature presence
        assert res.signature_present is False or res.confidence < 0.70

    def test_noisy_scan_signature_low_confidence(self):
        """6. Blurred/noisy signature -> low confidence or requires_human_review == True."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(80, 80, 80), center=(600, 760))
        # Apply heavy Gaussian blur + salt/pepper noise
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy_img = cv2.add(blurred, noise)

        res = self.detector.detect("DOC-TEST-006", img_to_bytes(noisy_img))
        assert res.requires_human_review is True or res.confidence < 0.75

    def test_zero_identity_or_authenticity_claims(self):
        """7. Assert response model strictly DOES NOT contain prohibited identity or authenticity fields."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(10, 10, 10))
        res = self.detector.detect("DOC-TEST-007", img_to_bytes(img))

        res_dict = res.model_dump()
        prohibited_fields = [
            "signer_identity",
            "signer_name",
            "signature_owner",
            "signature_authentic",
            "signature_valid",
            "signature_genuine",
            "biometric_match",
            "identity_match",
        ]
        for field in prohibited_fields:
            assert field not in res_dict, f"Prohibited field '{field}' detected in detector output!"

    def test_bounding_box_normalized_coordinates(self):
        """8. Assert coordinates are strictly normalized 0.0 <= x1, y1, x2, y2 <= 1.0."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(20, 20, 20), center=(550, 720))
        res = self.detector.detect("DOC-TEST-008", img_to_bytes(img))

        assert res.bbox is not None
        assert 0.0 <= res.bbox.x1 <= 1.0
        assert 0.0 <= res.bbox.y1 <= 1.0
        assert 0.0 <= res.bbox.x2 <= 1.0
        assert 0.0 <= res.bbox.y2 <= 1.0
        assert res.bbox.x1 < res.bbox.x2
        assert res.bbox.y1 < res.bbox.y2

    def test_black_pen_signature(self):
        """9. Verify standard black ballpoint pen signature."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(10, 10, 10), center=(580, 750))
        res = self.detector.detect("DOC-TEST-009", img_to_bytes(img))

        assert res.signature_present is True
        assert res.confidence >= 0.70

    def test_blue_pen_signature(self):
        """10. Verify blue ink ballpoint pen signature."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(180, 40, 20), center=(580, 750))
        res = self.detector.detect("DOC-TEST-010", img_to_bytes(img))

        assert res.signature_present is True
        assert res.confidence >= 0.70

    def test_stamp_and_signature_simultaneously(self):
        """11. Document contains both wet signature and official stamp simultaneously."""
        img = make_blank_pod_image()
        draw_curved_handwritten_signature(img, color=(15, 15, 15), center=(530, 760))
        draw_official_stamp(img, stamp_type="BLUE_PURPLE", center=(670, 760))
        res = self.detector.detect("DOC-TEST-011", img_to_bytes(img))

        assert res.signature_present is True
        assert res.stamp_present is True
        assert res.bbox is not None
        assert res.stamp_bbox is not None

    def test_colored_logo_not_detected_as_stamp(self):
        """12. Colored header logo in top 15% should NOT be detected as stamp in signature area."""
        img = make_blank_pod_image()
        # Add red/blue logo in header (y: 20 to 60)
        cv2.circle(img, (700, 45), 25, (190, 40, 50), -1)
        res = self.detector.detect("DOC-TEST-012", img_to_bytes(img))

        # Bottom signature zone should remain clean
        assert res.stamp_present is False
        assert res.signature_present is False

    def test_signature_bbox_points_to_actual_region(self):
        """13. Signature bounding box should accurately localize around the signature coordinates."""
        img = make_blank_pod_image(width=1000, height=1000)
        # Signature drawn explicitly at (600, 750)
        draw_curved_handwritten_signature(img, color=(20, 20, 20), center=(600, 750))
        res = self.detector.detect("DOC-TEST-013", img_to_bytes(img))

        assert res.bbox is not None
        # x center is 0.60, y center is 0.75
        assert 0.45 <= res.bbox.x1 <= 0.65
        assert 0.65 <= res.bbox.y1 <= 0.85

    def test_empty_document_is_unknown_or_safe(self):
        """14. Empty bytes or corrupted input handled safely without false accusations."""
        res = self.detector.detect("DOC-TEST-014", b"")
        assert res.signature_present is False
        assert res.requires_human_review is True
        assert res.is_unknown is True

    def test_low_resolution_document_requires_review(self):
        """15. Very low resolution thumbnail treated conservatively with requires_human_review == True."""
        small_img = np.ones((80, 80, 3), dtype=np.uint8) * 255
        cv2.line(small_img, (10, 40), (70, 40), (20, 20, 20), 2)
        res = self.detector.detect("DOC-TEST-015", img_to_bytes(small_img))

        assert res.requires_human_review is True
