"""
Resolve the real page number of extracted fields.

Extractors flatten the lines of every page before matching and historically stamped
``page=1`` on every field. That is wrong for multi-page documents: a bbox on page 3
would be drawn on page 1 by any viewer. This module re-attributes each located field
to the page whose OCR lines/tables actually contain its bbox (and, as tie-breaker, its
text value). It is generic: no document type or vendor specific rules.
"""
from __future__ import annotations

from typing import Iterable

_MIN_OVERLAP = 0.3


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_ratio(field_box: tuple[float, float, float, float], other: Iterable[float]) -> float:
    o = tuple(float(v) for v in other)
    if len(o) != 4:
        return 0.0
    if max(o) > 1.0:  # legacy 0..1000 scale, same convention as _bbox_norm_to_model
        o = tuple(v / 1000.0 for v in o)
    ix = max(0.0, min(field_box[2], o[2]) - max(field_box[0], o[0]))
    iy = max(0.0, min(field_box[3], o[3]) - max(field_box[1], o[1]))
    fa = _area(field_box)
    return (ix * iy) / fa if fa > 0 else 0.0


def resolve_field_pages(fields: list, pages: list[dict], default_page: int = 1) -> list:
    """Set ``field.source.page`` in place; returns ``fields`` for chaining.

    Fields without a bbox get ``default_page`` (the first page of their segment/group).
    """
    for f in fields:
        src = getattr(f, "source", None)
        if src is None:
            continue
        bbox = getattr(src, "bbox", None)
        if bbox is None:
            src.page = default_page
            continue
        fb = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        value = str(getattr(f, "value", "") or "").strip().lower()
        best_page, best_score = default_page, 0.0
        for p in pages:
            page_no = p.get("page_number")
            if not isinstance(page_no, int):
                continue
            score = 0.0
            for line in p.get("lines", []):
                r = _overlap_ratio(fb, line.get("bbox_norm") or [])
                if r < _MIN_OVERLAP:
                    continue
                s = r
                if value and value in str(line.get("text", "")).lower():
                    s += 0.5
                score = max(score, s)
            for table in p.get("tables", []):
                r = _overlap_ratio(fb, table.get("bbox_norm") or [])
                if r >= _MIN_OVERLAP:
                    score = max(score, r * 0.9)
            if score > best_score:
                best_page, best_score = page_no, score
        src.page = best_page
    return fields
