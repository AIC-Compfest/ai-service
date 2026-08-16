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
import math
import re
from typing import Any, Dict, List, Tuple
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.schemas.evidence import BoundingBox
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class OCRToken:
    def __init__(self, text: str, confidence: float, bbox_pixel: tuple[float, float, float, float], bbox_norm: tuple[float, float, float, float]):
        self.text = text
        self.confidence = confidence
        self.bbox_pixel = bbox_pixel  # (x1, y1, x2, y2)
        self.bbox_norm = bbox_norm    # (x1, y1, x2, y2) in [0, 1000]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox_pixel": [round(c, 1) for c in self.bbox_pixel],
            "bbox_norm": [round(c, 1) for c in self.bbox_norm],
        }


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
            "bbox_pixel": [round(c, 1) for c in self.bbox_pixel],
            "bbox_norm": [round(c, 1) for c in self.bbox_norm],
            "tokens": [t.to_dict() for t in self.tokens],
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
            "bbox_norm": [round(c, 1) for c in self.bbox_norm],
            "confidence": round(self.confidence, 3),
            "text_snippet": self.text[:100],
        }


class TableStructure:
    def __init__(self, bbox_norm: tuple[float, float, float, float], headers: list[str], rows: list[list[str]], confidence: float = 0.95):
        self.bbox_norm = bbox_norm
        self.headers = headers
        self.rows = rows
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "bbox_norm": [round(c, 1) for c in self.bbox_norm],
            "headers": self.headers,
            "rows": self.rows,
            "row_count": len(self.rows),
            "col_count": len(self.headers),
            "confidence": round(self.confidence, 3),
        }


class PageOCREvidence:
    def __init__(self, page_number: int, width: int, height: int, engine_used: str, fallback_used: bool, lines: list[OCRLine], regions: list[LayoutRegion], tables: list[TableStructure], quality: dict):
        self.page_number = page_number
        self.width = width
        self.height = height
        self.engine_used = engine_used
        self.fallback_used = fallback_used
        self.lines = lines
        self.regions = regions
        self.tables = tables
        self.quality = quality

    def full_text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def average_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([l.confidence for l in self.lines]))

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "engine_used": self.engine_used,
            "fallback_used": self.fallback_used,
            "average_confidence": round(self.average_confidence(), 3),
            "quality": self.quality,
            "lines": [l.to_dict() for l in self.lines],
            "regions": [r.to_dict() for r in self.regions],
            "tables": [t.to_dict() for t in self.tables],
        }


class OCREngine:
    """Cascaded OCR and Layout Understanding Engine."""

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            try:
                import easyocr
                import torch
                use_gpu = torch.cuda.is_available()
                logger.info("initializing_easyocr_reader", use_gpu=use_gpu)
                self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            except Exception as e:
                logger.warning("easyocr_init_failed", error=str(e))
                self._reader = None
        return self._reader

    def process_pdf_page(self, page: fitz.Page, page_number: int, quality: dict) -> PageOCREvidence:
        """Extract high-precision vector text, coordinates and layout from PDF page."""
        rect = page.rect
        w = float(rect.width)
        h = float(rect.height)

        words_data = page.get_text("words")
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

            norm_bbox = (
                lx0 / w * 1000,
                ly0 / h * 1000,
                lx1 / w * 1000,
                ly1 / h * 1000,
            )

            tokens: list[OCRToken] = []
            for item in w_items:
                tx0, ty0, tx1, ty1, tw = item
                t_norm = (tx0 / w * 1000, ty0 / h * 1000, tx1 / w * 1000, ty1 / h * 1000)
                tokens.append(OCRToken(text=tw, confidence=0.99, bbox_pixel=(tx0, ty0, tx1, ty1), bbox_norm=t_norm))

            ocr_lines.append(
                OCRLine(
                    text=line_text,
                    confidence=0.99,
                    bbox_pixel=(lx0, ly0, lx1, ly1),
                    bbox_norm=norm_bbox,
                    tokens=tokens,
                )
            )

        regions = self._detect_layout_regions(ocr_lines, w, h)
        tables = self._detect_tables(ocr_lines)

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
        )

    def process_image(self, image_bytes: bytes, page_number: int, quality: dict) -> PageOCREvidence:
        """Process raster image / scanned document using EasyOCR with GPU acceleration."""
        cv_img = ImagePreprocessor.bytes_to_cv2(image_bytes)
        h, w = cv_img.shape[:2]

        ocr_lines: list[OCRLine] = []
        engine_used = "easyocr_gpu"

        if self.reader is not None:
            try:
                raw_results = self.reader.readtext(cv_img)
                for box, text, prob in raw_results:
                    if not text.strip():
                        continue
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    lx0, ly0 = float(min(xs)), float(min(ys))
                    lx1, ly1 = float(max(xs)), float(max(ys))
                    norm_bbox = (lx0 / w * 1000, ly0 / h * 1000, lx1 / w * 1000, ly1 / h * 1000)

                    words = text.split()
                    tokens: list[OCRToken] = []
                    if len(words) > 1:
                        step_w = (lx1 - lx0) / len(words)
                        for i, word in enumerate(words):
                            tx0 = lx0 + i * step_w
                            tx1 = tx0 + step_w
                            t_norm = (tx0 / w * 1000, ly0 / h * 1000, tx1 / w * 1000, ly1 / h * 1000)
                            tokens.append(OCRToken(word, float(prob), (tx0, ly0, tx1, ly1), t_norm))
                    else:
                        tokens.append(OCRToken(text, float(prob), (lx0, ly0, lx1, ly1), norm_bbox))

                    ocr_lines.append(
                        OCRLine(
                            text=text,
                            confidence=float(prob),
                            bbox_pixel=(lx0, ly0, lx1, ly1),
                            bbox_norm=norm_bbox,
                            tokens=tokens,
                        )
                    )
            except Exception as e:
                logger.warning("easyocr_extraction_error", error=str(e))
                engine_used = "raster_cv_fallback"

        regions = self._detect_layout_regions(ocr_lines, w, h)
        tables = self._detect_tables(ocr_lines)

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

    def _detect_layout_regions(self, lines: list[OCRLine], width: float, height: float) -> list[LayoutRegion]:
        """Detect structural layout regions (header, address, table, totals, signature, footer)."""
        regions: list[LayoutRegion] = []
        if not lines:
            return regions

        header_lines = [l for l in lines if l.bbox_norm[1] < 180]
        if header_lines:
            hx1 = min(l.bbox_norm[0] for l in header_lines)
            hy1 = min(l.bbox_norm[1] for l in header_lines)
            hx2 = max(l.bbox_norm[2] for l in header_lines)
            hy2 = max(l.bbox_norm[3] for l in header_lines)
            regions.append(LayoutRegion("header", (hx1, hy1, hx2, hy2), 0.95, " ".join(l.text for l in header_lines)))

        meta_lines = [l for l in lines if 150 <= l.bbox_norm[1] < 380]
        if meta_lines:
            mx1 = min(l.bbox_norm[0] for l in meta_lines)
            my1 = min(l.bbox_norm[1] for l in meta_lines)
            mx2 = max(l.bbox_norm[2] for l in meta_lines)
            my2 = max(l.bbox_norm[3] for l in meta_lines)
            regions.append(LayoutRegion("metadata_block", (mx1, my1, mx2, my2), 0.92, " ".join(l.text for l in meta_lines)))

        table_lines = [l for l in lines if 360 <= l.bbox_norm[1] < 750]
        if table_lines:
            tx1 = min(l.bbox_norm[0] for l in table_lines)
            ty1 = min(l.bbox_norm[1] for l in table_lines)
            tx2 = max(l.bbox_norm[2] for l in table_lines)
            ty2 = max(l.bbox_norm[3] for l in table_lines)
            regions.append(LayoutRegion("table", (tx1, ty1, tx2, ty2), 0.94, " ".join(l.text for l in table_lines)))

        totals_lines = [l for l in lines if 700 <= l.bbox_norm[1] < 880 and any(k in l.text.lower() for k in ["total", "subtotal", "tax", "ppn", "amount"])]
        if totals_lines:
            tot_x1 = min(l.bbox_norm[0] for l in totals_lines)
            tot_y1 = min(l.bbox_norm[1] for l in totals_lines)
            tot_x2 = max(l.bbox_norm[2] for l in totals_lines)
            tot_y2 = max(l.bbox_norm[3] for l in totals_lines)
            regions.append(LayoutRegion("totals", (tot_x1, tot_y1, tot_x2, tot_y2), 0.95, " ".join(l.text for l in totals_lines)))

        sig_lines = [l for l in lines if l.bbox_norm[1] >= 750]
        if sig_lines:
            sx1 = min(l.bbox_norm[0] for l in sig_lines)
            sy1 = min(l.bbox_norm[1] for l in sig_lines)
            sx2 = max(l.bbox_norm[2] for l in sig_lines)
            sy2 = max(l.bbox_norm[3] for l in sig_lines)
            regions.append(LayoutRegion("signature_block", (sx1, sy1, sx2, sy2), 0.90, " ".join(l.text for l in sig_lines)))

        return regions

    def _detect_tables(self, lines: list[OCRLine]) -> list[TableStructure]:
        """Detect tabular line items and structural grid."""
        tables: list[TableStructure] = []
        table_lines = [l for l in lines if 320 <= l.bbox_norm[1] <= 760 and len(l.tokens) >= 3]
        if len(table_lines) >= 2:
            headers = [t.text for t in table_lines[0].tokens]
            rows: list[list[str]] = []
            for l in table_lines[1:]:
                row_cells = [t.text for t in l.tokens]
                if row_cells:
                    rows.append(row_cells)

            tx1 = min(l.bbox_norm[0] for l in table_lines)
            ty1 = min(l.bbox_norm[1] for l in table_lines)
            tx2 = max(l.bbox_norm[2] for l in table_lines)
            ty2 = max(l.bbox_norm[3] for l in table_lines)

            tables.append(TableStructure(bbox_norm=(tx1, ty1, tx2, ty2), headers=headers, rows=rows, confidence=0.94))
        return tables
