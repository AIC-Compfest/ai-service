"""Docling-based document understanding engine (the only OCR / text-extraction engine).

* PDF page   : Docling reads the vector text layer directly (``ocr=False``)  -> ``engine_used = docling_vector``.
* Image page : Docling runs RapidOCR internally (``ocr=True``)               -> ``engine_used = docling_rapidocr``.
* PDF page without a text layer (scan inside a PDF) falls back to OCR on the canonical PNG, ``fallback_used = True``.

The output keeps the public page-evidence contract: ``lines[]`` (text, confidence, ``bbox_norm`` in [0, 1], tokens),
``regions[]`` and ``tables[]`` (filled later by the type-aware table discovery in the pipeline).

Coordinates are normalized against the page size reported by Docling itself (PDF points / image pixels); the
canonical PNG is a uniform rescale of the same page, so the normalized values are valid in the PNG space.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import fitz
import yaml

from app.core.config import settings
from app.core.logging import get_logger
from app.services.docling.runner import DoclingResult, get_runner

logger = get_logger(__name__)

_DEFAULT_CONFIDENCE = 0.95


@lru_cache(maxsize=1)
def _line_construction_config() -> dict:
    try:
        policy = yaml.safe_load((settings.config_dir / "extraction" / "review_policy.yaml").read_text(encoding="utf-8")) or {}
        return policy.get("line_construction", {})
    except Exception:
        return {}


class DoclingOCREngine:
    """Processes one page (PDF or image) through Docling and returns the internal page-evidence dict."""

    def process(
        self,
        raw_bytes: bytes,
        *,
        is_pdf: bool,
        page_number: int,
        quality: dict,
        page_width: int,
        page_height: int,
        fallback_image_bytes: bytes | None = None,
    ) -> dict:
        """``raw_bytes`` is a single-page PDF (``is_pdf``) or a PNG. ``fallback_image_bytes`` lets a text-less PDF page
        be OCR-ed from its canonical PNG."""
        runner = get_runner()
        if is_pdf:
            result = runner.convert_page(pdf_bytes=raw_bytes, image_loader=lambda: fallback_image_bytes)
        else:
            result = runner.convert(raw_bytes, is_pdf=False, ocr=True)
        if result is None:
            # Do not make document ingestion unavailable because an optional
            # layout engine cannot be imported.  Vector PDFs retain their
            # text layer, which is enough for the field extractors and the
            # reviewer mapping screen.  Scanned/image files still report a
            # precise error because there is no text layer to recover.
            if is_pdf:
                fallback = _extract_pdf_text(raw_bytes, page_width, page_height)
                if fallback:
                    lines = fallback
                    return {
                        "page_number": page_number,
                        "width": page_width,
                        "height": page_height,
                        "engine_used": "pymupdf_vector_fallback",
                        "fallback_used": True,
                        "average_confidence": _avg_conf(lines),
                        "quality": quality,
                        "coord_space": "page_image_normalized",
                        "lines": lines,
                        "regions": _detect_regions(lines),
                        "tables": [],
                    }
            raise RuntimeError(f"No OCR/text extraction result for page {page_number}; Docling is unavailable and no PDF text layer was found")

        lines, split_count = _split_merged_lines(_extract_lines(result))
        regions = _detect_regions(lines)
        return {
            "page_number": page_number,
            "width": page_width,
            "height": page_height,
            "engine_used": "docling_rapidocr" if result.used_ocr else "docling_vector",
            "fallback_used": bool(is_pdf and result.used_ocr),
            "average_confidence": _avg_conf(lines),
            "quality": quality,
            "coord_space": "page_image_normalized",
            "line_split_count": split_count,
            "lines": lines,
            "regions": regions,
            "tables": [],  # filled by discover_page_tables in the pipeline
        }


def _extract_lines(result: DoclingResult) -> list[dict]:
    """Visual text rows from positioned words, with line/text cells as fallbacks.

    Docling can merge multiple visual rows into one ``TextCell``. Word cells
    retain their positions, so use them to construct the public line contract
    whenever they are available instead of trusting a merged line cell.
    """
    lines = _lines_from_words(result) if result.word_cells else (_lines_from_cells(result) if result.line_cells else _lines_from_texts(result))
    lines.sort(key=lambda item: (item["bbox_norm"][1], item["bbox_norm"][0]))
    _assign_row_indices(lines)
    return lines


def _split_merged_lines(lines: list[dict]) -> tuple[list[dict], int]:
    """Split oversized merged line boxes at spatially separated word tokens."""
    config = _line_construction_config()
    width_floor = float(config.get("split_line_width_fraction", 0.60))
    gap_floor = float(config.get("split_gap_fraction", 0.045))
    median_multiple = float(config.get("split_height_median_multiple", 3.0))
    heights = sorted(max(0.0, line["bbox_norm"][3] - line["bbox_norm"][1]) for line in lines if line.get("bbox_norm"))
    median_height = heights[len(heights) // 2] if heights else 0.0
    output, split_count = [], 0
    for line in lines:
        box = line.get("bbox_norm") or [0, 0, 0, 0]
        tokens = list(line.get("tokens", []))
        width, height = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
        if width < width_floor or len(tokens) < 2:
            output.append(line)
            continue
        tokens.sort(key=lambda token: (token.get("bbox_norm", [0, 0, 0, 0])[1], token.get("bbox_norm", [0, 0, 0, 0])[0]))
        rows: list[list[dict]] = []
        for token in tokens:
            tbox = token.get("bbox_norm") or [0, 0, 0, 0]
            center = (float(tbox[1]) + float(tbox[3])) / 2
            tolerance = float(_line_construction_config().get("row_baseline_tolerance", 0.55))
            row = next((row for row in rows if abs(center - sum((float(t["bbox_norm"][1]) + float(t["bbox_norm"][3])) / 2 for t in row) / len(row)) <= max(0.002, tolerance * (float(tbox[3]) - float(tbox[1])))), None)
            if row is None: rows.append([token])
            else: row.append(token)
        split_groups = []
        for row in rows:
            row.sort(key=lambda token: token["bbox_norm"][0])
            current = []
            for token in row:
                tbox = token["bbox_norm"]
                prev_box = current[-1]["bbox_norm"] if current else None
                gap = float(tbox[0]) - float(prev_box[2]) if prev_box else 0.0
                if prev_box and gap >= gap_floor and (width >= width_floor or (median_height and height > median_multiple * median_height)):
                    split_groups.append(current)
                    current = []
                current.append(token)
            if current: split_groups.append(current)
        if len(split_groups) <= 1:
            output.append(line)
            continue
        split_count += len(split_groups) - 1
        for group in split_groups:
            boxes = [token["bbox_norm"] for token in group]
            group_box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            output.append({**line, "text": " ".join(token["text"] for token in group), "bbox_norm": group_box, "tokens": group})
    output.sort(key=lambda item: (item["bbox_norm"][1], item["bbox_norm"][0]))
    _assign_row_indices(output)
    return output, split_count


def _lines_from_words(result: DoclingResult) -> list[dict]:
    positioned = []
    for cell in result.word_cells:
        text = " ".join(str(cell.text).split())
        box = _norm_cell(cell, result)
        if text and box is not None:
            positioned.append((text, box, _confidence(cell)))
    positioned.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))

    rows: list[list[tuple[str, tuple[float, float, float, float], float]]] = []
    for word in positioned:
        _, box, _ = word
        center = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        row_index = next((i for i, row in enumerate(rows) if _same_visual_row(center, height, row)), None)
        if row_index is None:
            rows.append([word])
        else:
            rows[row_index].append(word)

    lines = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item[1][0])
        box = (min(item[1][0] for item in row), min(item[1][1] for item in row), max(item[1][2] for item in row), max(item[1][3] for item in row))
        confidence = sum(item[2] for item in row) / len(row)
        tokens = [_token(text, conf, word_box, estimated=False) for text, word_box, conf in row]
        for token in tokens:
            token["row_idx"] = row_index
        lines.append({"text": " ".join(item[0] for item in row), "confidence": round(confidence, 3), "bbox_norm": _rounded(box), "tokens": tokens})
    return lines


def _same_visual_row(center: float, height: float, row: list[tuple[str, tuple[float, float, float, float], float]]) -> bool:
    row_center = sum((item[1][1] + item[1][3]) / 2 for item in row) / len(row)
    row_height = max(item[1][3] - item[1][1] for item in row)
    tolerance = max(0.002, float(_line_construction_config().get("row_baseline_tolerance", 0.55)) * max(height, row_height))
    return abs(center - row_center) <= tolerance


def _extract_pdf_text(pdf_bytes: bytes, target_width: int, target_height: int) -> list[dict]:
    """Recover positioned vector text when Docling cannot start.

    The fallback deliberately uses the same normalized line/token contract as
    Docling so downstream classification, mapping, confidence and HITL review
    need no special handling.  It is only used for PDFs with an actual text
    layer; it must not pretend that a scanned PDF has been OCRed.
    """
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = document[0]
        source_width, source_height = float(page.rect.width), float(page.rect.height)
        grouped: dict[tuple[int, int], list[tuple[float, float, float, float, str]]] = {}
        for x0, y0, x1, y1, text, block, line, _word in page.get_text("words"):
            value = " ".join(str(text).split())
            if value:
                grouped.setdefault((block, line), []).append((x0, y0, x1, y1, value))
        lines: list[dict] = []
        for words in grouped.values():
            words.sort(key=lambda item: item[0])
            x0 = min(item[0] for item in words)
            y0 = min(item[1] for item in words)
            x1 = max(item[2] for item in words)
            y1 = max(item[3] for item in words)
            box = _norm_bbox_from_values(x0, y0, x1, y1, source_width, source_height)
            if box is None:
                continue
            tokens = []
            for wx0, wy0, wx1, wy1, value in words:
                word_box = _norm_bbox_from_values(wx0, wy0, wx1, wy1, source_width, source_height)
                if word_box is not None:
                    tokens.append(_token(value, _DEFAULT_CONFIDENCE, word_box, estimated=False))
            lines.append({"text": " ".join(item[4] for item in words), "confidence": _DEFAULT_CONFIDENCE, "bbox_norm": _rounded(box), "tokens": tokens})
        lines.sort(key=lambda item: (item["bbox_norm"][1], item["bbox_norm"][0]))
        _assign_row_indices(lines)
        return lines
    except Exception as exc:
        logger.warning("pdf_vector_fallback_failed", error=str(exc), target_width=target_width, target_height=target_height)
        return []


def _norm_bbox_from_values(x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
        return None
    return max(0.0, x0) / width, max(0.0, y0) / height, min(width, x1) / width, min(height, y1) / height


def _lines_from_cells(result: DoclingResult) -> list[dict]:
    words = [(cell, _norm_cell(cell, result)) for cell in result.word_cells]
    words = [(cell, box) for cell, box in words if box is not None and str(cell.text).strip()]
    lines: list[dict] = []
    for cell in result.line_cells:
        text = " ".join(str(cell.text).split())
        box = _norm_cell(cell, result)
        if not text or box is None:
            continue
        confidence = _confidence(cell)
        inside = sorted(
            (item for item in words if box[0] - 1e-3 <= (item[1][0] + item[1][2]) / 2 <= box[2] + 1e-3 and box[1] - 1e-3 <= (item[1][1] + item[1][3]) / 2 <= box[3] + 1e-3),
            key=lambda item: item[1][0],
        )
        if inside and " ".join(" ".join(str(c.text).split()) for c, _ in inside) == text:
            tokens = [_token(" ".join(str(c.text).split()), _confidence(c), b, estimated=False) for c, b in inside]
        else:
            tokens = _estimated_tokens(text, confidence, box)
        lines.append({"text": text, "confidence": round(confidence, 3), "bbox_norm": _rounded(box), "tokens": tokens})
    return lines


def _lines_from_texts(result: DoclingResult) -> list[dict]:
    lines: list[dict] = []
    for item in result.texts:
        text = " ".join(str(getattr(item, "text", "")).split())
        prov = (getattr(item, "prov", None) or [None])[0]
        box = _norm_bbox(getattr(prov, "bbox", None), result.page_width, result.page_height)
        if not text or box is None:
            continue
        confidence = _confidence(item)
        lines.append({"text": text, "confidence": round(confidence, 3), "bbox_norm": _rounded(box), "tokens": _estimated_tokens(text, confidence, box)})
    return lines


def _confidence(item: Any) -> float:
    value = getattr(item, "confidence", None)
    try:
        return float(value) if value is not None else _DEFAULT_CONFIDENCE
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE


def _norm_cell(cell: Any, result: DoclingResult) -> tuple[float, float, float, float] | None:
    try:
        bbox = cell.rect.to_bounding_box()
    except AttributeError:
        return None
    return _norm_bbox(bbox, result.page_width, result.page_height)


def _norm_bbox(bbox: Any, width: float, height: float) -> tuple[float, float, float, float] | None:
    """Normalize a docling bbox to [0, 1] with a top-left origin (BOTTOMLEFT boxes are flipped)."""
    if bbox is None or width <= 0 or height <= 0:
        return None
    try:
        left, right, top, bottom = float(bbox.l), float(bbox.r), float(bbox.t), float(bbox.b)
    except (AttributeError, TypeError, ValueError):
        return None
    origin = str(getattr(getattr(bbox, "coord_origin", None), "name", "")).upper()
    if "BOTTOMLEFT" in origin:
        top, bottom = height - top, height - bottom
    x1, x2 = sorted((max(0.0, min(width, left)), max(0.0, min(width, right))))
    y1, y2 = sorted((max(0.0, min(height, top)), max(0.0, min(height, bottom))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 / width, y1 / height, x2 / width, y2 / height


def _rounded(box: tuple[float, float, float, float]) -> list[float]:
    return [round(value, 6) for value in box]


def _token(text: str, confidence: float, box: tuple[float, float, float, float], *, estimated: bool) -> dict:
    return {"text": text, "confidence": round(confidence, 3), "bbox_norm": _rounded(box), "_estimated": estimated, "row_idx": 0}


def _estimated_tokens(text: str, confidence: float, box: tuple[float, float, float, float]) -> list[dict]:
    """Split a line box between its words in proportion to their length (OCR engines give line boxes only)."""
    words = text.split()
    total = sum(len(word) for word in words) + max(len(words) - 1, 0)
    if not words or total <= 0:
        return []
    width, cursor, tokens = box[2] - box[0], box[0], []
    for word in words:
        span = width * len(word) / total
        tokens.append(_token(word, confidence, (cursor, box[1], cursor + span, box[3]), estimated=True))
        cursor += width * (len(word) + 1) / total
    return tokens


def _assign_row_indices(lines: list[dict]) -> None:
    """Stable visual rows: lines whose vertical centres match share one ``row_idx`` on all their tokens."""
    rows: list[tuple[float, float]] = []
    for line in lines:
        box = line["bbox_norm"]
        center, height = (box[1] + box[3]) / 2, max(box[3] - box[1], 1e-6)
        row = next((i for i, (cy, h) in enumerate(rows) if abs(center - cy) <= 0.6 * max(height, h)), None)
        if row is None:
            row = len(rows)
            rows.append((center, height))
        for token in line["tokens"]:
            token["row_idx"] = row


def _detect_regions(lines: list[dict]) -> list[dict]:
    """Layout regions (header, metadata_block, table, totals, signature_block) from line position and keywords."""
    regions: list[dict] = []
    if not lines:
        return regions

    def has_keyword(line: dict, keywords: tuple[str, ...]) -> bool:
        text = line["text"].casefold()
        return any(keyword in text for keyword in keywords)

    def add_region(region_type: str, region_lines: list[dict], confidence: float) -> None:
        if not region_lines:
            return
        regions.append({
            "region_type": region_type,
            "bbox_norm": _rounded((
                min(line["bbox_norm"][0] for line in region_lines),
                min(line["bbox_norm"][1] for line in region_lines),
                max(line["bbox_norm"][2] for line in region_lines),
                max(line["bbox_norm"][3] for line in region_lines),
            )),
            "confidence": round(confidence, 3),
            "text_snippet": " ".join(line["text"] for line in region_lines)[:100],
        })

    add_region("header", [l for l in lines if l["bbox_norm"][1] < 0.18 or (l["bbox_norm"][1] < 0.30 and has_keyword(l, ("invoice", "faktur", "surat jalan", "rate card")))], 0.95)
    add_region("metadata_block", [l for l in lines if 0.15 <= l["bbox_norm"][1] < 0.38 or has_keyword(l, ("kepada", "billed to", "tanggal", "date", "nomor", "no."))], 0.92)
    add_region("table", [l for l in lines if 0.35 <= l["bbox_norm"][1] < 0.75 or has_keyword(l, ("description", "quantity", "qty", "harga", "rate", "jumlah"))], 0.94)
    add_region("totals", [l for l in lines if l["bbox_norm"][1] >= 0.60 and has_keyword(l, ("total", "subtotal", "tax", "ppn", "amount", "grand total"))], 0.95)
    add_region("signature_block", [l for l in lines if l["bbox_norm"][1] >= 0.75 or (l["bbox_norm"][1] >= 0.55 and has_keyword(l, ("hormat kami", "signature", "tanda tangan", "penerima")))], 0.90)
    return regions


def _avg_conf(lines: list[dict]) -> float:
    if not lines:
        return 0.0
    return round(sum(line["confidence"] for line in lines) / len(lines), 3)
