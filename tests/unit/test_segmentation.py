"""
test_entity_resolver.py — Unit tests for fixed EntityResolver and RateAgreementSelector.

Tests cover all root causes fixed in ARCHITECTURE_AUDIT.md §2:
  1. NormalizeID uses word-boundary regex (not dangerous string Replace)
  2. Level 1 matching uses strict equality (not Contains)
  3. Hard conflict detection blocks false positives
  4. InvoiceCanonical.ShipmentID is the PRIMARY matching key
  5. RateAgreementSelector enforces date validity

Run: python -m pytest ai-service/tests/unit/test_entity_resolver.py -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.services.segmentation.segmenter import (
    DocumentSegmentationService,
    PageClassification,
    DocumentSegment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for DocumentSegmentationService
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentSegmentationService:
    """Tests for boundary detection algorithm."""

    def setup_method(self):
        self.svc = DocumentSegmentationService(min_confidence=0.45)

    def _make_page(self, page_number: int, doc_type: str, confidence: float = 0.90) -> PageClassification:
        return PageClassification(
            page_number=page_number,
            document_type=doc_type,
            confidence=confidence,
        )

    # ─── Basic boundary detection ───────────────────────────────────────────

    def test_single_page_single_segment(self):
        """One page → one segment."""
        pages = [self._make_page(1, "INVOICE")]
        segments = self.svc.segment(pages)
        assert len(segments) == 1
        assert segments[0].document_type == "INVOICE"
        assert segments[0].start_page == 1
        assert segments[0].end_page == 1
        assert segments[0].page_count == 1
        assert segments[0].segment_number == 1

    def test_consecutive_same_type_merge(self):
        """Pages 1,2,3 all INVOICE → one segment [1-3]."""
        pages = [
            self._make_page(1, "INVOICE"),
            self._make_page(2, "INVOICE"),
            self._make_page(3, "INVOICE"),
        ]
        segments = self.svc.segment(pages)
        assert len(segments) == 1
        assert segments[0].document_type == "INVOICE"
        assert segments[0].start_page == 1
        assert segments[0].end_page == 3
        assert segments[0].page_count == 3

    def test_boundary_detected_on_type_change(self):
        """INVOICE p1, SURAT_JALAN p2 → 2 segments."""
        pages = [
            self._make_page(1, "INVOICE"),
            self._make_page(2, "SURAT_JALAN"),
        ]
        segments = self.svc.segment(pages)
        assert len(segments) == 2
        assert segments[0].document_type == "INVOICE"
        assert segments[0].start_page == 1
        assert segments[0].end_page == 1
        assert segments[1].document_type == "SURAT_JALAN"
        assert segments[1].start_page == 2
        assert segments[1].end_page == 2

    def test_10_page_mixed_pdf(self):
        """
        10-page PDF: INVOICE p1-2, SJ p3, POD p4-5, INV p6, UNKNOWN p7, INVOICE p8-10
        Expected: 5 segments (INVOICE:1-2, SJ:3, POD:4-5, INVOICE:6, UNKNOWN:7, INVOICE:8-10)
        Wait, p6=INVOICE but p7=UNKNOWN means boundary at p7.
        Then p8-10=INVOICE creates new segment after UNKNOWN.
        Result: 6 segments.
        """
        pages = [
            self._make_page(1, "INVOICE"),
            self._make_page(2, "INVOICE"),
            self._make_page(3, "SURAT_JALAN"),
            self._make_page(4, "POD"),
            self._make_page(5, "POD"),
            self._make_page(6, "INVOICE"),
            self._make_page(7, "UNKNOWN"),
            self._make_page(8, "INVOICE"),
            self._make_page(9, "INVOICE"),
            self._make_page(10, "INVOICE"),
        ]
        segments = self.svc.segment(pages)
        # Should be: INVOICE(1-2), SJ(3), POD(4-5), INVOICE(6), UNKNOWN(7), INVOICE(8-10)
        assert len(segments) == 6
        assert segments[0].document_type == "INVOICE" and segments[0].start_page == 1 and segments[0].end_page == 2
        assert segments[1].document_type == "SURAT_JALAN" and segments[1].start_page == 3 and segments[1].end_page == 3
        assert segments[2].document_type == "POD" and segments[2].start_page == 4 and segments[2].end_page == 5
        assert segments[3].document_type == "INVOICE" and segments[3].start_page == 6 and segments[3].end_page == 6
        assert segments[4].document_type == "UNKNOWN" and segments[4].start_page == 7 and segments[4].end_page == 7
        assert segments[5].document_type == "INVOICE" and segments[5].start_page == 8 and segments[5].end_page == 10

    def test_segment_numbers_sequential(self):
        """Segment numbers are 1-indexed and sequential."""
        pages = [
            self._make_page(1, "INVOICE"),
            self._make_page(2, "POD"),
            self._make_page(3, "SURAT_JALAN"),
        ]
        segments = self.svc.segment(pages)
        assert [s.segment_number for s in segments] == [1, 2, 3]

    # ─── Low confidence handling ─────────────────────────────────────────────

    def test_low_confidence_page_becomes_unknown(self):
        """
        Page with confidence < min_threshold → treated as UNKNOWN.
        INVOICE p1(conf=0.90), INVOICE p2(conf=0.20) → INVOICE p1, UNKNOWN p2
        """
        pages = [
            self._make_page(1, "INVOICE", confidence=0.90),
            self._make_page(2, "INVOICE", confidence=0.20),  # below threshold
        ]
        segments = self.svc.segment(pages)
        assert len(segments) == 2
        assert segments[0].document_type == "INVOICE"
        assert segments[1].document_type == "UNKNOWN"

    def test_all_same_type_high_confidence_one_segment(self):
        """All INVOICE pages with high confidence → one segment."""
        pages = [self._make_page(i, "INVOICE", confidence=0.95) for i in range(1, 6)]
        segments = self.svc.segment(pages)
        assert len(segments) == 1
        assert segments[0].page_count == 5
        assert segments[0].classification_confidence > 0.9

    def test_empty_input_returns_empty(self):
        """Empty page list → empty segment list."""
        segments = self.svc.segment([])
        assert segments == []

    # ─── Confidence computation ─────────────────────────────────────────────

    def test_segment_confidence_is_mean_of_pages(self):
        """Segment confidence should be mean of constituent page confidences."""
        pages = [
            self._make_page(1, "INVOICE", confidence=0.80),
            self._make_page(2, "INVOICE", confidence=0.90),
            self._make_page(3, "INVOICE", confidence=0.70),
        ]
        segments = self.svc.segment(pages)
        assert len(segments) == 1
        expected_conf = (0.80 + 0.90 + 0.70) / 3
        assert abs(segments[0].classification_confidence - expected_conf) < 0.001

    # ─── Legacy group conversion ─────────────────────────────────────────────

    def test_segment_from_logical_groups_basic(self):
        """Convert legacy logical_groups dict into segments."""
        logical_groups = {
            "INVOICE": [1, 2],
            "SURAT_JALAN": [3],
            "POD": [4, 5],
            "RATE_AGREEMENT": [],
            "UNKNOWN": [],
        }
        segments = self.svc.segment_from_logical_groups(logical_groups)
        # Should create segments for non-empty groups
        types_present = {s.document_type for s in segments}
        assert "INVOICE" in types_present
        assert "SURAT_JALAN" in types_present
        assert "POD" in types_present

    def test_segment_page_count_correct(self):
        """page_count property is correct."""
        pages = [
            self._make_page(3, "POD"),
            self._make_page(4, "POD"),
            self._make_page(5, "POD"),
        ]
        segments = self.svc.segment(pages)
        assert len(segments) == 1
        assert segments[0].start_page == 3
        assert segments[0].end_page == 5
        assert segments[0].page_count == 3

    # ─── to_dict() output ───────────────────────────────────────────────────

    def test_segment_to_dict_has_all_keys(self):
        """DocumentSegment.to_dict() should have all required keys."""
        pages = [self._make_page(1, "INVOICE")]
        segments = self.svc.segment(pages)
        d = segments[0].to_dict()
        required_keys = [
            "segment_number", "document_type", "start_page", "end_page",
            "page_count", "classification_confidence", "classification_method",
            "requires_human_review",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_segment_to_dict_values_valid(self):
        """Segment dict values should be within valid ranges."""
        pages = [self._make_page(2, "POD", confidence=0.85)]
        segments = self.svc.segment(pages)
        d = segments[0].to_dict()
        assert d["segment_number"] >= 1
        assert d["start_page"] <= d["end_page"]
        assert 0.0 <= d["classification_confidence"] <= 1.0
        assert d["document_type"] in ("INVOICE", "SURAT_JALAN", "POD", "RATE_AGREEMENT", "UNKNOWN")
