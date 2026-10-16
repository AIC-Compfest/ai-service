"""
engine.py

Cascaded OCR and Document Layout Understanding Engine (PRD §22).
Architecture:
- High-precision PyMuPDF vector extraction for digital PDFs.
- Computer Vision / OCR for raster scans, JPGs, and photographed PODs.
- Document Layout Understanding: detects header, metadata, address, table, totals, signature, and footer regions.
- Table Structural Understanding: preserves table bounding box, rows, and cell coordinates.
"""

from __future__ import annotations
import os
import threading
import re
from pathlib import Path
from typing import Any
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.core.config import candidates_config_dir, settings
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..preprocessing.ocr_prep import OCRPreprocessor, PreparedImage
from ..preprocessing.page_kind import PageKind, PageKindResult, detect_page_kind

logger = get_logger(__name__)


def _paddle_payload(result: Any) -> dict[str, Any]:
    """Read PaddleOCR v3 Result JSON without depending on private classes."""
    value = result.json if hasattr(result, "json") else result
    if callable(value):
        value = value()
    return value.get("res", value) if isinstance(value, dict) else {}


class OCRToken:
    def __init__(
        self,
        text: str,
        confidence: float,
        bbox_pixel: tuple[float, float, float, float],
        bbox_norm: tuple[float, float, float, float],
        estimated: bool = False,
        row_idx: int = 0,
    ):
        self.text = text
        self.confidence = confidence
        self.bbox_pixel = bbox_pixel  # (x1, y1, x2, y2)
        self.bbox_norm = bbox_norm    # (x1, y1, x2, y2) in [0.0, 1.0]
        # True only where the word box was derived from a larger OCR box.
        self.estimated = estimated
        self.row_idx = row_idx

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox_norm": [round(c, 6) for c in self.bbox_norm],
            "_estimated": self.estimated,
            "row_idx": self.row_idx,
        }

    def to_full_dict(self) -> dict:
        """Return diagnostic data, including pixel coordinates, for OCR inspection."""
        return {**self.to_dict(), "bbox_pixel": [round(c, 1) for c in self.bbox_pixel]}


class OCRLine:
    def __init__(self, text: str, confidence: float, bbox_pixel: tuple[float, float, float, float], bbox_norm: tuple[float, float, float, float], tokens: list[OCRToken] | None = None):
        self.text = text
        self.confidence = confidence
        self.bbox_pixel = bbox_pixel
        self.bbox_norm = bbox_norm
        self.tokens = tokens or []

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox_norm": [round(c, 6) for c in self.bbox_norm],
            "tokens": [t.to_dict() for t in self.tokens],
        }

    def to_full_dict(self) -> dict:
        return {
            **self.to_dict(),
            "bbox_pixel": [round(c, 1) for c in self.bbox_pixel],
            "tokens": [token.to_full_dict() for token in self.tokens],
        }


class LayoutRegion:
    def __init__(self, region_type: str, bbox_norm: tuple[float, float, float, float], confidence: float, text: str = ""):
        self.region_type = region_type  # header, metadata, address, table, totals, signature, footer, qr_barcode
        self.bbox_norm = bbox_norm
        self.confidence = confidence
        self.text = text

    def to_dict(self) -> dict:
        return {
            "region_type": self.region_type,
            "bbox_norm": [round(c, 6) for c in self.bbox_norm],
            "confidence": round(self.confidence, 3),
            "text_snippet": self.text[:100],
        }


class TableStructure:
    def __init__(self, bbox_norm: tuple[float, float, float, float], headers: list[str], rows: list[list[str]], confidence: float = 0.95, provider: str = "", reason: str = "", validation: dict | None = None, columns: list[dict] | None = None, cells: list[dict] | None = None, role: str = "", role_reason: str = "", summary_rows: list[list[str]] | None = None, table_id: str = "", continued_from: str = ""):
        self.bbox_norm = bbox_norm
        self.headers = headers
        self.rows = rows
        self.confidence = confidence
        self.provider, self.reason, self.validation = provider, reason, validation or {}
        self.columns, self.cells = columns or [], cells or []
        self.role, self.role_reason, self.summary_rows = role, role_reason, summary_rows or []
        self.table_id, self.continued_from = table_id, continued_from

    def to_dict(self) -> dict:
        payload = {
            "bbox_norm": [round(c, 6) for c in self.bbox_norm],
            "headers": self.headers,
            "rows": self.rows,
            "row_count": len(self.rows),
            "col_count": len(self.headers),
            "confidence": round(self.confidence, 3),
        }
        if self.provider:
            payload.update({"provider": self.provider, "reason": self.reason, "validation": self.validation, "columns": self.columns, "cells": self.cells})
            # Additive fields (K6): role/summary/identity of the table.
            payload.update({"role": self.role, "role_reason": self.role_reason, "summary_rows": self.summary_rows, "table_id": self.table_id})
            if self.continued_from:
                payload["continued_from"] = self.continued_from
        return payload


class PageOCREvidence:
    def __init__(self, page_number: int, width: int, height: int, engine_used: str, fallback_used: bool, lines: list[OCRLine], regions: list[LayoutRegion], tables: list[TableStructure], quality: dict, page_image: dict | None = None, page_kind: PageKindResult | None = None):
        self.page_number = page_number
        self.width = width
        self.height = height
        self.engine_used = engine_used
        self.fallback_used = fallback_used
        self.lines = lines
        self.regions = regions
        self.tables = tables
        self.quality = quality
        self.page_image = page_image or {}
        self.page_kind = page_kind

    def full_text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def average_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([l.confidence for l in self.lines]))

    def to_dict(self) -> dict:
        """CandidatePipeline-compatible, JSON-serializable page evidence."""
        payload = {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "engine_used": self.engine_used,
            "fallback_used": self.fallback_used,
            "average_confidence": round(self.average_confidence(), 3),
            "quality": self.quality,
            "page_image": self.page_image,
            "coord_space": "page_image_normalized",
            "lines": [l.to_dict() for l in self.lines],
            "regions": [r.to_dict() for r in self.regions],
            "tables": [t.to_dict() for t in self.tables],
        }
        if self.page_kind is not None:
            payload.update(self.page_kind.to_dict())
        return payload


def to_evidence_json(pages: list[PageOCREvidence]) -> dict[str, list[dict[str, Any]]]:
    """Build the public OCR-to-candidate payload from OCR page evidence objects."""
    return {"pages": [page.to_dict() for page in pages]}


_GLOBAL_PADDLE_OCR = None
# Paddle predictors are not thread-safe and initialisation is slow: serialise both (the API may call from a thread pool).
_PADDLE_LOCK = threading.RLock()

class OCREngine:
    """Cascaded OCR and Layout Understanding Engine."""

    def __init__(self, confidence_threshold: float = 0.85, max_image_side: int = 2400):
        self.confidence_threshold = confidence_threshold
        self.max_image_side = max_image_side

    @property
    def paddle_ocr(self):
        """Lazily initialize the sole raster OCR backend."""
        global _GLOBAL_PADDLE_OCR
        if _GLOBAL_PADDLE_OCR is None:
            with _PADDLE_LOCK:
                if _GLOBAL_PADDLE_OCR is not None:
                    return _GLOBAL_PADDLE_OCR
                try:
                    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                    from paddleocr import PaddleOCR
                    logger.info("initializing_paddleocr", language=settings.paddle_language, mkldnn=settings.paddle_enable_mkldnn)
                    options: dict[str, Any] = {"lang": settings.paddle_language, "use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False, "enable_mkldnn": settings.paddle_enable_mkldnn, "cpu_threads": settings.paddle_cpu_threads}
                    if settings.paddle_det_model:
                        options["text_detection_model_name"] = settings.paddle_det_model
                    if settings.paddle_rec_model:
                        options["text_recognition_model_name"] = settings.paddle_rec_model
                    if settings.paddle_det_limit_side_len > 0:
                        options.update(text_det_limit_type="max", text_det_limit_side_len=settings.paddle_det_limit_side_len)
                    _GLOBAL_PADDLE_OCR = PaddleOCR(**options)
                except Exception as e:
                    logger.error("paddleocr_init_failed", error=str(e))
                    raise RuntimeError("PaddleOCR initialization failed") from e
        return _GLOBAL_PADDLE_OCR

    def _estimate_vector_token_confidence(self, token_text: str) -> float:
        """Estimate confidence for digital PDF vector token based on character validity."""
        if not token_text:
            return 0.50
        printable_chars = sum(1 for c in token_text if c.isprintable())
        ratio = printable_chars / len(token_text)
        if ratio < 0.8:
            return round(0.50 + ratio * 0.3, 3)
        has_alphanumeric = any(c.isalnum() for c in token_text)
        if not has_alphanumeric:
            return 0.88
        return round(min(0.98, 0.90 + (ratio * 0.08)), 3)

    @staticmethod
    def _glyph_width(char: str) -> float:
        """Relative width approximation used only when OCR exposes phrase boxes."""
        if char in "ilI1|!.,:;'`":
            return 0.45
        if char in "MW@#%&":
            return 1.45
        if char.isupper():
            return 1.12
        if char.isdigit():
            return 0.95
        return 1.0

    def _estimate_token_boxes(
        self, text: str, bbox_pixel: tuple[float, float, float, float]
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Split an OCR phrase box using glyph widths and its literal whitespace.

        PaddleOCR's OCR result supplies one box for a recognised phrase, not
        per-word polygons. These boxes are therefore explicitly marked estimated.
        The allocation preserves relative word widths and multiple source spaces,
        rather than assigning every token the same width.
        """
        parts = list(re.finditer(r"\S+", text))
        if not parts:
            return []
        x1, y1, x2, y2 = bbox_pixel
        units: list[float] = []
        for index, match in enumerate(parts):
            units.append(sum(self._glyph_width(char) for char in match.group()))
            if index < len(parts) - 1:
                whitespace = text[match.end() : parts[index + 1].start()]
                units.append(max(0.35, 0.35 * len(whitespace)))
        total_units = sum(units) or 1.0
        scale = (x2 - x1) / total_units
        cursor = x1
        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        for index, match in enumerate(parts):
            word_width = units[index * 2] * scale
            token_x1 = cursor
            token_x2 = min(x2, cursor + word_width)
            boxes.append((match.group(), (token_x1, y1, token_x2, y2)))
            cursor = token_x2
            if index < len(parts) - 1:
                cursor += units[index * 2 + 1] * scale
        if boxes:
            word, (token_x1, token_y1, _, token_y2) = boxes[-1]
            boxes[-1] = (word, (token_x1, token_y1, x2, token_y2))
        return boxes

    def _read_paddle_lines(self, cv_img: np.ndarray, width: int, height: int) -> list[OCRLine]:
        """Run PaddleOCR and convert its v3 result fields to OCR evidence."""
        scale = 1.0
        ocr_input = cv_img
        if max(height, width) > self.max_image_side:
            scale = self.max_image_side / max(height, width)
            ocr_input = cv2.resize(cv_img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        lines: list[OCRLine] = []
        with _PADDLE_LOCK:
            results = list(self.paddle_ocr.predict(ocr_input))
        for result in results:
            payload = _paddle_payload(result)
            for text, probability, box in zip(payload.get("rec_texts", []), payload.get("rec_scores", []), payload.get("rec_boxes", [])):
                if not text.strip() or len(box) != 4:
                    continue
                lx0, ly0, lx1, ly1 = (float(value) / scale for value in box)
                line_box = (lx0, ly0, lx1, ly1)
                norm_bbox = self._safe_norm_bbox(lx0, ly0, lx1, ly1, width, height)
                estimated_boxes = self._estimate_token_boxes(text, line_box)
                tokens = [OCRToken(word, float(probability), token_box, self._safe_norm_bbox(*token_box, width, height), estimated=len(estimated_boxes) > 1) for word, token_box in estimated_boxes]
                lines.append(OCRLine(text, float(probability), line_box, norm_bbox, tokens))
        return self._assign_row_indices(lines)

    @staticmethod
    def _remap_lines(
        lines: list[OCRLine], inverse_transform: np.ndarray, width: int, height: int
    ) -> list[OCRLine]:
        """Map OCR boxes from a prepared image back into source-image coordinates."""
        def remap_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            x1, y1, x2, y2 = box
            corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(corners, inverse_transform)[0]
            xs, ys = mapped[:, 0], mapped[:, 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

        for line in lines:
            line.bbox_pixel = remap_box(line.bbox_pixel)
            line.bbox_norm = OCREngine._safe_norm_bbox(*line.bbox_pixel, width, height)
            for token in line.tokens:
                token.bbox_pixel = remap_box(token.bbox_pixel)
                token.bbox_norm = OCREngine._safe_norm_bbox(*token.bbox_pixel, width, height)
        return OCREngine._assign_row_indices(lines)

    def _read_prepared_lines(
        self, prepared: PreparedImage, source_width: int, source_height: int
    ) -> list[OCRLine]:
        """Read a prepared variant and preserve the source-image evidence geometry."""
        prepared_height, prepared_width = prepared.image.shape[:2]
        lines = self._read_paddle_lines(prepared.image, prepared_width, prepared_height)
        return self._remap_lines(lines, prepared.inverse_transform, source_width, source_height)

    @staticmethod
    def _safe_norm_bbox(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> tuple[float, float, float, float]:
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 <= x1: x2 = min(width, x1 + 1)
        if y2 <= y1: y2 = min(height, y1 + 1)
        return x1 / width, y1 / height, x2 / width, y2 / height

    @staticmethod
    def _assign_row_indices(lines: list[OCRLine]) -> list[OCRLine]:
        """Assign stable visual rows independently of OCR provider internals."""
        rows: list[tuple[float, float]] = []
        for line in sorted(lines, key=lambda item: (item.bbox_pixel[1], item.bbox_pixel[0], item.text)):
            center = (line.bbox_pixel[1] + line.bbox_pixel[3]) / 2
            height = max(1.0, line.bbox_pixel[3] - line.bbox_pixel[1])
            row = next((i for i, (cy, h) in enumerate(rows) if abs(center - cy) <= .6 * max(height, h)), None)
            if row is None:
                row = len(rows); rows.append((center, height))
            for token in line.tokens:
                token.row_idx = row
        return lines

    def process_pdf_page(self, page: fitz.Page, page_number: int, quality: dict, canonical_image_bytes: bytes | None = None, discover_tables: bool = True) -> PageOCREvidence:
        """Extract high-precision vector text, coordinates and layout from PDF page.
        Falls back to PaddleOCR if page is scanned/image-based with insufficient digital text."""
        rect = page.rect
        w = float(rect.width)
        h = float(rect.height)

        words_data = page.get_text("words")
        page_kind = self._page_kind(page)

        # Fallback to PaddleOCR for scanned / image-only PDF pages.
        use_raster_ocr = len(words_data) < settings.page_kind_legacy_word_count_floor
        if page_kind is not None:
            use_raster_ocr = page_kind.kind == PageKind.SCAN
        if use_raster_ocr:
            logger.info("ocr.fallback_to_paddleocr", page_number=page_number, words_found=len(words_data), page_kind=page_kind.kind if page_kind else "legacy")
            pix = page.get_pixmap(dpi=200, alpha=False)
            img_bytes = canonical_image_bytes or pix.tobytes("png")
            ocr_res = self.process_image(img_bytes, page_number, quality, discover_tables=discover_tables)
            ocr_res.fallback_used = True
            ocr_res.page_kind = page_kind
            return ocr_res

        lines_by_key: dict[tuple[int, int], list] = {}
        for item in words_data:
            x0, y0, x1, y1, word, b_no, l_no, w_no = item
            key = (b_no, l_no)
            if key not in lines_by_key:
                lines_by_key[key] = []
            lines_by_key[key].append((x0, y0, x1, y1, word))

        ocr_lines: list[OCRLine] = []
        for (b_no, l_no), w_items in sorted(lines_by_key.items()):
            w_items = sorted(w_items, key=lambda x: x[0])
            line_text = " ".join(item[4] for item in w_items)
            lx0 = min(item[0] for item in w_items)
            ly0 = min(item[1] for item in w_items)
            lx1 = max(item[2] for item in w_items)
            ly1 = max(item[3] for item in w_items)

            norm_bbox = self._safe_norm_bbox(lx0, ly0, lx1, ly1, w, h)

            tokens: list[OCRToken] = []
            token_confs: list[float] = []
            for item in w_items:
                tx0, ty0, tx1, ty1, tw = item
                t_norm = self._safe_norm_bbox(tx0, ty0, tx1, ty1, w, h)
                t_conf = self._estimate_vector_token_confidence(tw)
                token_confs.append(t_conf)
                tokens.append(OCRToken(text=tw, confidence=t_conf, bbox_pixel=(tx0, ty0, tx1, ty1), bbox_norm=t_norm))

            line_conf = float(np.mean(token_confs)) if token_confs else 0.95

            ocr_lines.append(
                OCRLine(
                    text=line_text,
                    confidence=round(line_conf, 3),
                    bbox_pixel=(lx0, ly0, lx1, ly1),
                    bbox_norm=norm_bbox,
                    tokens=tokens,
                )
            )

        ocr_lines = self._assign_row_indices(ocr_lines)
        regions = self._detect_layout_regions(ocr_lines, w, h)
        tables = self._detect_tables(ocr_lines, page_number, page) if discover_tables else []

        return PageOCREvidence(
            page_number=page_number,
            width=int(w),
            height=int(h),
            engine_used="pymupdf_vector",
            fallback_used=False,
            lines=ocr_lines,
            regions=regions,
            tables=tables,
            quality=quality,
            page_kind=page_kind,
        )

    @staticmethod
    def _page_kind(page: fitz.Page) -> PageKindResult | None:
        """Return feature-flagged routing evidence without changing legacy default."""
        if not settings.page_kind_routing:
            return None
        return detect_page_kind(
            page,
            word_coverage_min=settings.page_kind_word_coverage_min,
            image_coverage_scan_threshold=settings.page_kind_image_coverage_scan_threshold,
            full_page_image_threshold=settings.page_kind_full_page_image_threshold,
            legacy_word_count_floor=settings.page_kind_legacy_word_count_floor,
        )

    def process_image(
        self,
        image_bytes: bytes | str | Path,
        page_number: int,
        quality: dict,
        discover_tables: bool = True,
    ) -> PageOCREvidence:
        """Process raster image / scanned document using PaddleOCR."""
        if isinstance(image_bytes, (str, Path)):
            image_path = Path(image_bytes)
            image_bytes = image_path.read_bytes()
        cv_img = ImagePreprocessor.bytes_to_cv2(image_bytes)
        h, w = cv_img.shape[:2]

        ocr_lines: list[OCRLine] = []
        engine_used = "paddleocr"
        try:
            ocr_lines = self._read_paddle_lines(cv_img, w, h)
            if settings.enable_ocr_preprocess:
                preprocessor = OCRPreprocessor(
                    settings.ocr_preprocess_min_side_for_upscale,
                    settings.ocr_preprocess_max_upscale,
                )
                prepared = preprocessor.prepare(cv_img)
                prepared_lines = self._read_prepared_lines(prepared, w, h)
                raw_score = (self._average_line_confidence(ocr_lines), len(ocr_lines))
                prepared_score = (self._average_line_confidence(prepared_lines), len(prepared_lines))
                use_prepared = prepared_score > raw_score
                if use_prepared:
                    ocr_lines = prepared_lines
                quality = {
                    **quality,
                    "ocr_preprocessing": {
                        "enabled": True,
                        "selected": "prepared" if use_prepared else "raw",
                        "operations": list(prepared.operations),
                        "raw_score": list(raw_score),
                        "prepared_score": list(prepared_score),
                    },
                }
        except Exception as e:
            logger.error("paddleocr_extraction_error", error=str(e))
            raise RuntimeError("PaddleOCR raster extraction failed") from e

        regions = self._detect_layout_regions(ocr_lines, w, h)
        tables = self._detect_tables(ocr_lines, page_number) if discover_tables else []

        return PageOCREvidence(
            page_number=page_number,
            width=w,
            height=h,
            engine_used=engine_used,
            fallback_used=True,
            lines=ocr_lines,
            regions=regions,
            tables=tables,
            quality=quality,
        )

    @staticmethod
    def _average_line_confidence(lines: list[OCRLine]) -> float:
        return float(np.mean([line.confidence for line in lines])) if lines else 0.0

    def _detect_layout_regions(self, lines: list[OCRLine], width: float, height: float) -> list[LayoutRegion]:
        """Detect structural layout regions (header, address, table, totals, signature, footer)."""
        regions: list[LayoutRegion] = []
        if not lines:
            return regions

        def has_keyword(line: OCRLine, keywords: tuple[str, ...]) -> bool:
            text = line.text.casefold()
            return any(keyword in text for keyword in keywords)

        def add_region(region_type: str, region_lines: list[OCRLine], confidence: float) -> None:
            if not region_lines:
                return
            regions.append(LayoutRegion(
                region_type,
                (
                    min(line.bbox_norm[0] for line in region_lines),
                    min(line.bbox_norm[1] for line in region_lines),
                    max(line.bbox_norm[2] for line in region_lines),
                    max(line.bbox_norm[3] for line in region_lines),
                ),
                confidence,
                " ".join(line.text for line in region_lines),
            ))

        header_lines = [
            line for line in lines
            if line.bbox_norm[1] < 0.18
            or (line.bbox_norm[1] < 0.30 and has_keyword(line, ("invoice", "faktur", "surat jalan", "rate card")))
        ]
        if header_lines:
            add_region("header", header_lines, 0.95)

        meta_lines = [
            line for line in lines
            if 0.15 <= line.bbox_norm[1] < 0.38
            or has_keyword(line, ("kepada", "billed to", "tanggal", "date", "nomor", "no."))
        ]
        if meta_lines:
            add_region("metadata_block", meta_lines, 0.92)

        table_lines = [
            line for line in lines
            if 0.35 <= line.bbox_norm[1] < 0.75
            or has_keyword(line, ("description", "quantity", "qty", "harga", "rate", "jumlah"))
        ]
        if table_lines:
            add_region("table", table_lines, 0.94)

        totals_lines = [
            line for line in lines
            if line.bbox_norm[1] >= 0.60
            and has_keyword(line, ("total", "subtotal", "tax", "ppn", "amount", "grand total"))
        ]
        if totals_lines:
            add_region("totals", totals_lines, 0.95)

        sig_lines = [
            line for line in lines
            if line.bbox_norm[1] >= 0.75
            or (line.bbox_norm[1] >= 0.55 and has_keyword(line, ("hormat kami", "signature", "tanda tangan", "penerima")))
        ]
        if sig_lines:
            add_region("signature_block", sig_lines, 0.90)

        return regions

    def _detect_tables(self, lines: list[OCRLine], page_number: int, source_page=None) -> list[TableStructure]:
        """Route discovery through providers; doc type is intentionally unknown here."""
        if not settings.enable_table_extractor:
            return []
        from app.services.candidates.context import build_page
        from app.services.candidates.pipeline import table_evidence
        from app.services.candidates.tables import TableExtractor
        page = build_page({"page_number": page_number, "lines": [line.to_dict() for line in lines], "_source_page": source_page})
        extractor = TableExtractor.cached(candidates_config_dir(), settings.table_provider_order)
        return [TableStructure(**_table_structure_args(table_evidence(table))) for table in extractor.extract(page, None, f"ocr-p{page_number}")]


def _table_structure_args(evidence: dict) -> dict:
    return {"bbox_norm": tuple(evidence["bbox_norm"]), "headers": evidence["headers"], "rows": evidence["rows"], "confidence": evidence["confidence"], "provider": evidence["provider"], "reason": evidence["reason"], "validation": evidence["validation"], "columns": evidence["columns"], "cells": evidence["cells"], "role": evidence.get("role", ""), "role_reason": evidence.get("role_reason", ""), "summary_rows": evidence.get("summary_rows", []), "table_id": evidence.get("table_id", "")}
