"""
page_kind.py

Per-page digital-vs-scan classifier (Step 1 of feat/table-reliability).

Replaces the coarse ``len(words) < 5`` heuristic with a multi-signal detector
that examines:

  1. Word coverage  – union area of word bboxes / page area.
  2. Image coverage – total embedded-image area / page area.
  3. Full-page image flag – any single image >= 90 % of page area.
  4. Invisible-text layer – presence of render-mode-3 glyphs (OCR overlay).

Outputs one of four page kinds:

  DIGITAL           – selectable vector text dominates, few/no full-page images.
  SCAN              – full-page raster image with little or no selectable text.
  HYBRID_OCR_LAYER  – full-page image PLUS invisible OCR text layer (already-OCR'd PDF).
  MIXED             – significant text AND large embedded images on the same page.

All thresholds are sourced from ``app.core.config.Settings`` and can be
overridden via environment variables or the ``.env`` file.

Design rules
------------
- This module is **pure-function**: ``detect_page_kind`` takes only a
  ``fitz.Page`` and returns a ``PageKindResult``; no side-effects.
- The legacy ``len(words) < 5`` safety-net is NOT replaced — it remains in
  ``OCREngine.process_pdf_page`` and acts as a last-resort catch.
- When ``settings.page_kind_routing`` is ``False`` (default), the OCR engine
  ignores this module entirely, preserving old behaviour.
- Coordinates and areas use **PDF points** (not pixels) so DPI does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF – version verified 1.28.0

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PageKind:
    DIGITAL = "DIGITAL"
    SCAN = "SCAN"
    HYBRID_OCR_LAYER = "HYBRID_OCR_LAYER"
    MIXED = "MIXED"


@dataclass(frozen=True)
class PageKindResult:
    """
    Result of per-page digital-vs-scan classification.

    Attributes
    ----------
    kind:
        One of ``PageKind.*`` constants.
    word_coverage:
        Fraction of page area covered by the union of word bboxes (0–1).
    image_coverage:
        Fraction of page area covered by embedded images (0–1, capped at 1).
    has_full_page_image:
        True if any single image occupies >= 90 % of the page area.
    has_invisible_text:
        True if any glyph uses render mode 3 (invisible / OCR text layer).
    word_count:
        Number of selectable words returned by ``get_text("words")``.
    reason:
        Human-readable explanation string for debugging.
    signals:
        Dict with raw signal values for logging / evidence JSON.
    """

    kind: str
    word_coverage: float
    image_coverage: float
    has_full_page_image: bool
    has_invisible_text: bool
    word_count: int
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_kind": self.kind,
            "word_coverage": round(self.word_coverage, 4),
            "image_coverage": round(self.image_coverage, 4),
            "has_full_page_image": self.has_full_page_image,
            "has_invisible_text": self.has_invisible_text,
            "word_count": self.word_count,
            "page_kind_reason": self.reason,
            "page_kind_signals": self.signals,
        }


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


def detect_page_kind(
    page: fitz.Page,
    *,
    word_coverage_min: float = 0.05,
    image_coverage_scan_threshold: float = 0.70,
    full_page_image_threshold: float = 0.90,
    legacy_word_count_floor: int = 5,
) -> PageKindResult:
    """
    Classify a PDF page as DIGITAL, SCAN, HYBRID_OCR_LAYER, or MIXED.

    Parameters
    ----------
    page:
        A ``fitz.Page`` object from an open ``fitz.Document``.
    word_coverage_min:
        Minimum word-bbox union / page area to consider the page DIGITAL.
        Defaults to ``settings.page_kind_word_coverage_min`` (0.05).
    image_coverage_scan_threshold:
        Image area / page area above which the page is SCAN-dominant.
        Defaults to ``settings.page_kind_image_coverage_scan_threshold`` (0.70).
    full_page_image_threshold:
        Single-image area / page area above which ``has_full_page_image=True``.

    Returns
    -------
    PageKindResult
    """
    rect = page.rect
    page_area = float(rect.width * rect.height)
    if page_area <= 0:
        return PageKindResult(
            kind=PageKind.SCAN,
            word_coverage=0.0,
            image_coverage=0.0,
            has_full_page_image=False,
            has_invisible_text=False,
            word_count=0,
            reason="degenerate page (zero area) – defaulting to SCAN",
        )

    # ------------------------------------------------------------------
    # Signal 1: word coverage
    # ------------------------------------------------------------------
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, blk, ln, wrd)
    word_count = len(words)
    word_union_area = _union_area([fitz.Rect(w[0], w[1], w[2], w[3]) for w in words])
    word_coverage = min(word_union_area / page_area, 1.0)

    # ------------------------------------------------------------------
    # Signal 2: image coverage
    # ------------------------------------------------------------------
    img_infos = page.get_image_info()  # list of dicts with 'bbox'
    img_total_area = 0.0
    has_full_page_image = False
    for info in img_infos:
        bbox = info.get("bbox")
        if bbox is None:
            continue
        r = fitz.Rect(bbox)
        area = float(r.width * r.height)
        img_total_area += area
        if area / page_area >= full_page_image_threshold:
            has_full_page_image = True
    image_coverage = min(img_total_area / page_area, 1.0)

    # ------------------------------------------------------------------
    # Signal 3: invisible text layer (render mode 3 = invisible)
    # get_texttrace() returns list of dicts; each span has 'color_space'
    # and rendering mode baked in the 'char' entries' 'flags' field.
    #
    # In PyMuPDF 1.28, get_texttrace() returns a list of "spans" where
    # each span dict has a 'type' key that encodes the render mode.
    # render mode 3 == TR3 == invisible (used by PDF2Text OCR workflows).
    # ------------------------------------------------------------------
    has_invisible_text = False
    try:
        traces = page.get_texttrace()
        for span in traces:
            # Each span has 'type' field; type & 0x7 gives the render mode
            render_mode = span.get("type", 0) & 0x7
            if render_mode == 3:
                has_invisible_text = True
                break
    except Exception as exc:
        logger.debug("page_kind.texttrace_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------
    kind, reason = _classify(
        word_coverage=word_coverage,
        image_coverage=image_coverage,
        has_full_page_image=has_full_page_image,
        has_invisible_text=has_invisible_text,
        word_count=word_count,
        word_coverage_min=word_coverage_min,
        image_coverage_scan_threshold=image_coverage_scan_threshold,
        legacy_word_count_floor=legacy_word_count_floor,
    )

    signals = {
        "word_count": word_count,
        "word_coverage": round(word_coverage, 4),
        "image_coverage": round(image_coverage, 4),
        "has_full_page_image": has_full_page_image,
        "has_invisible_text": has_invisible_text,
        "n_images": len(img_infos),
    }

    logger.debug(
        "page_kind_detected",
        kind=kind,
        reason=reason,
        **signals,
    )

    return PageKindResult(
        kind=kind,
        word_coverage=word_coverage,
        image_coverage=image_coverage,
        has_full_page_image=has_full_page_image,
        has_invisible_text=has_invisible_text,
        word_count=word_count,
        reason=reason,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------


def _classify(
    *,
    word_coverage: float,
    image_coverage: float,
    has_full_page_image: bool,
    has_invisible_text: bool,
    word_count: int,
    word_coverage_min: float,
    image_coverage_scan_threshold: float,
    legacy_word_count_floor: int,
) -> tuple[str, str]:
    """Apply classification rules. Returns (kind, reason)."""

    # Rule 1: full-page image + invisible text → already-OCR'd PDF (hybrid)
    if has_full_page_image and has_invisible_text:
        return (
            PageKind.HYBRID_OCR_LAYER,
            "full-page image + invisible render-mode-3 text layer detected",
        )

    # Rule 2: full-page image only → SCAN
    if has_full_page_image and word_coverage < word_coverage_min:
        return (
            PageKind.SCAN,
            f"full-page image, word_coverage={word_coverage:.3f} < {word_coverage_min}",
        )

    # Rule 3: large image coverage but some text → could be MIXED or SCAN
    if image_coverage >= image_coverage_scan_threshold:
        if word_coverage >= word_coverage_min:
            return (
                PageKind.MIXED,
                f"image_coverage={image_coverage:.3f} >= threshold={image_coverage_scan_threshold}"
                f" but word_coverage={word_coverage:.3f} >= {word_coverage_min}",
            )
        return (
            PageKind.SCAN,
            f"image_coverage={image_coverage:.3f} >= threshold={image_coverage_scan_threshold}"
            f", word_coverage={word_coverage:.3f} < {word_coverage_min}",
        )

    # Rule 4: minimal selectable text → SCAN. This legacy safety net is only
    # reached after stronger image/OCR-layer evidence above, but before the
    # word-coverage rule: a near-empty footer can occupy a high fraction of a
    # small crop despite not being a digital document page.
    if word_count < legacy_word_count_floor:
        return (
            PageKind.SCAN,
            f"word_count={word_count} < {legacy_word_count_floor} legacy safety net",
        )

    # Rule 5: good word coverage → DIGITAL
    if word_coverage >= word_coverage_min:
        return (
            PageKind.DIGITAL,
            f"word_coverage={word_coverage:.3f} >= {word_coverage_min}",
        )

    # Fallback → SCAN (conservative)
    return (
        PageKind.SCAN,
        f"no dominant signal (word_coverage={word_coverage:.3f}, image_coverage={image_coverage:.3f})"
        " – defaulting to SCAN",
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _union_area(rects: list[fitz.Rect]) -> float:
    """
    Compute the area of the union of a list of rectangles.

    Uses a sweep-line approach: not pixel-perfect but fast and accurate
    enough for coverage estimates on standard document pages.

    Falls back to sum of individual areas (upper bound) for large inputs
    to avoid O(n²) overhead when there are thousands of words.
    """
    if not rects:
        return 0.0

    # Compute the exact union of axis-aligned rectangles. Word bounding boxes
    # frequently overlap (for example OCR overlays), so a summed-area shortcut
    # would violate the coverage contract and could classify an OCR layer as
    # DIGITAL incorrectly.
    valid = [r for r in rects if not r.is_empty]
    if not valid:
        return 0.0
    x_edges = sorted({coordinate for rect in valid for coordinate in (rect.x0, rect.x1)})
    total = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rect.y0, rect.y1) for rect in valid if rect.x0 < right and rect.x1 > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= end:
                end = max(end, y1)
            else:
                covered += end - start
                start, end = y0, y1
        covered += end - start
        total += (right - left) * covered
    return total
