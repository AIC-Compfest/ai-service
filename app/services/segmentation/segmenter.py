"""
segmenter.py — Document Boundary Detection Service

Converts page-level classifications into logical document segments with proper
boundary detection. This is the key architectural fix for PRD Objective-B.

Architectural mandate: Physical File ≠ Logical Document
  - Input:  list of (page_number, doc_type, confidence) per page
  - Output: list of DocumentSegment with contiguous page ranges per type

Boundary detection algorithm:
  1. Walk pages in order
  2. If current page type != previous page type → start new segment
  3. Consecutive pages with the SAME type merge into one segment
  4. Unknown pages: try to merge with adjacent segment if confidence is low
     else keep as UNKNOWN segment

Design rules:
  - Correctness > compactness
  - A 10-page PDF with alternating Invoice/POD pages creates 10 segments (not 2)
    because each boundary is a real document change.
  - UNKNOWN pages between two segments of different types stay UNKNOWN.
  - Minimum confidence threshold: if page confidence < threshold, treat as UNKNOWN.

Reference: ARCHITECTURE_AUDIT.md §6.2, PRD §Objective-B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Minimum confidence to trust a page classification as part of a named segment.
# Pages below this threshold are classified as UNKNOWN.
MIN_PAGE_CONFIDENCE = 0.45

# Known document types (must match DocumentSegment.document_type values in Go model)
KNOWN_TYPES = {"INVOICE", "SURAT_JALAN", "POD", "RATE_AGREEMENT", "UNKNOWN"}


@dataclass
class PageClassification:
    """Result of per-page classification."""
    page_number: int
    document_type: str          # INVOICE | SURAT_JALAN | POD | RATE_AGREEMENT | UNKNOWN
    confidence: float           # [0.0, 1.0]
    requires_human_review: bool = False
    method: str = "ml_classifier"


@dataclass
class DocumentSegment:
    """
    A logical document occupying a contiguous range of pages within a physical file.
    
    Invariants:
      - start_page <= end_page
      - all pages [start_page, end_page] have been classified as document_type
      - classification_confidence is the mean confidence of member pages
    """
    segment_number: int         # 1-indexed within the physical document
    document_type: str          # INVOICE | SURAT_JALAN | POD | RATE_AGREEMENT | UNKNOWN
    start_page: int             # Inclusive
    end_page: int               # Inclusive
    classification_confidence: float
    classification_method: str = "page_boundary_detection"
    requires_human_review: bool = False

    # Metadata
    page_confidences: list[float] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_number": self.segment_number,
            "document_type": self.document_type,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_count": self.page_count,
            "classification_confidence": round(self.classification_confidence, 4),
            "classification_method": self.classification_method,
            "requires_human_review": self.requires_human_review,
        }


class DocumentSegmentationService:
    """
    Converts page-level classifications into logical document segments.
    
    This service implements the boundary detection algorithm that groups
    consecutive same-type pages into segments, and splits at type transitions.
    """

    def __init__(
        self,
        min_confidence: float = MIN_PAGE_CONFIDENCE,
    ):
        self.min_confidence = min_confidence

    def segment(self, page_classifications: list[PageClassification]) -> list[DocumentSegment]:
        """
        Convert page-level classifications into logical document segments.
        
        Args:
            page_classifications: List of PageClassification, one per page.
                                  Must be in page_number order.

        Returns:
            List of DocumentSegment with non-overlapping, contiguous page ranges.
            Ordered by start_page.
        
        Algorithm:
            1. For each page, apply confidence threshold (low-confidence → UNKNOWN)
            2. Walk pages: if current type != previous type → boundary → start new segment
            3. Collect consecutive same-type pages into one segment
            4. Compute segment-level confidence as mean of member pages
        """
        if not page_classifications:
            return []

        # Sort by page number to ensure correct ordering
        pages = sorted(page_classifications, key=lambda p: p.page_number)

        segments: list[DocumentSegment] = []
        seg_number = 1

        # Initialize with first page
        current_type = self._effective_type(pages[0])
        current_start = pages[0].page_number
        current_confidences = [pages[0].confidence]
        current_requires_review = pages[0].requires_human_review

        for i in range(1, len(pages)):
            page = pages[i]
            page_type = self._effective_type(page)

            if page_type == current_type:
                # Same type → extend current segment
                current_confidences.append(page.confidence)
                if page.requires_human_review:
                    current_requires_review = True
            else:
                # Type changed → boundary detected → save current segment
                seg = self._make_segment(
                    segment_number=seg_number,
                    document_type=current_type,
                    start_page=current_start,
                    end_page=pages[i - 1].page_number,
                    confidences=current_confidences,
                    requires_review=current_requires_review,
                )
                segments.append(seg)
                seg_number += 1

                # Start new segment
                current_type = page_type
                current_start = page.page_number
                current_confidences = [page.confidence]
                current_requires_review = page.requires_human_review

        # Don't forget the last segment
        seg = self._make_segment(
            segment_number=seg_number,
            document_type=current_type,
            start_page=current_start,
            end_page=pages[-1].page_number,
            confidences=current_confidences,
            requires_review=current_requires_review,
        )
        segments.append(seg)

        return segments

    def _effective_type(self, page: PageClassification) -> str:
        """
        Returns the effective document type for a page.
        If the page confidence is below the minimum threshold, treat as UNKNOWN.
        
        This prevents low-confidence misclassifications from creating false segment
        boundaries or incorrectly extending named segments.
        """
        if page.confidence < self.min_confidence:
            return "UNKNOWN"
        if page.document_type not in KNOWN_TYPES:
            return "UNKNOWN"
        return page.document_type

    @staticmethod
    def _make_segment(
        *,
        segment_number: int,
        document_type: str,
        start_page: int,
        end_page: int,
        confidences: list[float],
        requires_review: bool,
    ) -> DocumentSegment:
        """Create a DocumentSegment from collected page data."""
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return DocumentSegment(
            segment_number=segment_number,
            document_type=document_type,
            start_page=start_page,
            end_page=end_page,
            classification_confidence=mean_confidence,
            classification_method="page_boundary_detection",
            requires_human_review=requires_review or mean_confidence < 0.65,
            page_confidences=confidences,
        )

    def segment_from_logical_groups(
        self,
        logical_groups: dict[str, list[int]],
        page_confidences: dict[int, float] | None = None,
    ) -> list[DocumentSegment]:
        """
        Convert a logical_groups dict (legacy pipeline format) into DocumentSegments.
        
        This is the TRANSITION method for backward compatibility with the existing
        pipeline.py output format: {"INVOICE": [1,2], "POD": [5,6], ...}
        
        NOTE: This does NOT detect true boundaries — it just creates one segment
        per document type from the groups dict. For proper boundary detection, use
        segment() with per-page classifications instead.
        
        Args:
            logical_groups: Dict mapping doc_type → list of page numbers
            page_confidences: Optional dict of page_number → confidence
        
        Returns:
            List of DocumentSegment, one per non-empty group.
        """
        if page_confidences is None:
            page_confidences = {}

        # Flatten to (page_number, doc_type) pairs
        page_type_map: dict[int, str] = {}
        for doc_type, page_nums in logical_groups.items():
            if doc_type not in KNOWN_TYPES:
                continue
            for pn in page_nums:
                page_type_map[pn] = doc_type

        if not page_type_map:
            return []

        # Convert to PageClassification list
        page_classifications = [
            PageClassification(
                page_number=pn,
                document_type=doc_type,
                confidence=page_confidences.get(pn, 0.80),
            )
            for pn, doc_type in page_type_map.items()
        ]

        return self.segment(page_classifications)
