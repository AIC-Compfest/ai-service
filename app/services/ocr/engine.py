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
import re
from pathlib import Path
from typing import Any
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.core.logging import get_logger
from ..preprocessing.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


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
    def __init__(self, bbox_norm: tuple[float, float, float, float], headers: list[str], rows: list[list[str]], confidence: float = 0.95):
        self.bbox_norm = bbox_norm
        self.headers = headers
        self.rows = rows
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "bbox_norm": [round(c, 6) for c in self.bbox_norm],
            "headers": self.headers,
            "rows": self.rows,
            "row_count": len(self.rows),
            "col_count": len(self.headers),
            "confidence": round(self.confidence, 3),
        }


class PageOCREvidence:
    def __init__(self, page_number: int, width: int, height: int, engine_used: str, fallback_used: bool, lines: list[OCRLine], regions: list[LayoutRegion], tables: list[TableStructure], quality: dict, page_image: dict | None = None):
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

    def full_text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def average_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([l.confidence for l in self.lines]))

    def to_dict(self) -> dict:
        """CandidatePipeline-compatible, JSON-serializable page evidence."""
        return {
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


def to_evidence_json(pages: list[PageOCREvidence]) -> dict[str, list[dict[str, Any]]]:
    """Build the public OCR-to-candidate payload from OCR page evidence objects."""
    return {"pages": [page.to_dict() for page in pages]}


_GLOBAL_EASYOCR_READER = None

class OCREngine:
    """Cascaded OCR and Layout Understanding Engine."""

    def __init__(self, confidence_threshold: float = 0.85, max_image_side: int = 2400):
        self.confidence_threshold = confidence_threshold
        self.max_image_side = max_image_side

    @property
    def reader(self):
        global _GLOBAL_EASYOCR_READER
        if _GLOBAL_EASYOCR_READER is None:
            try:
                import easyocr
                import torch
                use_gpu = torch.cuda.is_available()
                logger.info("initializing_easyocr_reader", use_gpu=use_gpu)
                try:
                    _GLOBAL_EASYOCR_READER = easyocr.Reader(["en", "id"], gpu=use_gpu, verbose=False)
                except Exception:
                    _GLOBAL_EASYOCR_READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            except Exception as e:
                logger.warning("easyocr_init_failed", error=str(e))
                _GLOBAL_EASYOCR_READER = None
        return _GLOBAL_EASYOCR_READER

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
        """Split an EasyOCR phrase box using glyph widths and its literal whitespace.

        EasyOCR's standard API supplies one polygon for a recognised phrase, not
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

    def _read_image_lines(self, cv_img: np.ndarray, width: int, height: int) -> list[OCRLine]:
        """Run EasyOCR once and convert phrase detections into candidate-ready lines."""
        scale = 1.0
        ocr_input = cv_img
        if max(height, width) > self.max_image_side:
            scale = self.max_image_side / max(height, width)
            ocr_input = cv2.resize(cv_img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        lines: list[OCRLine] = []
        for box, text, probability in self.reader.readtext(
            ocr_input,
            decoder="beamsearch",
            beamWidth=5,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.65,
            low_text=0.30,
            link_threshold=0.35,
        ):
            if not text.strip():
                continue
            xs = [point[0] / scale for point in box]
            ys = [point[1] / scale for point in box]
            lx0, ly0, lx1, ly1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            line_box = (lx0, ly0, lx1, ly1)
            norm_bbox = self._safe_norm_bbox(lx0, ly0, lx1, ly1, width, height)
            estimated_boxes = self._estimate_token_boxes(text, line_box)
            tokens = [
                OCRToken(
                    word,
                    float(probability),
                    token_box,
                    self._safe_norm_bbox(*token_box, width, height),
                    estimated=len(estimated_boxes) > 1,
                )
                for word, token_box in estimated_boxes
            ]
            lines.append(OCRLine(text, float(probability), line_box, norm_bbox, tokens))
        return self._assign_row_indices(lines)

    @staticmethod
    def _safe_norm_bbox(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> tuple[float, float, float, float]:
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 <= x1: x2 = min(width, x1 + 1)
        if y2 <= y1: y2 = min(height, y1 + 1)
        return x1 / width, y1 / height, x2 / width, y2 / height

    @staticmethod
    def _assign_row_indices(lines: list[OCRLine]) -> list[OCRLine]:
        """Assign stable visual rows, independent of EasyOCR phrase line indices."""
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

    def process_pdf_page(self, page: fitz.Page, page_number: int, quality: dict, canonical_image_bytes: bytes | None = None) -> PageOCREvidence:
        """Extract high-precision vector text, coordinates and layout from PDF page.
        Falls back to EasyOCR if page is scanned/image-based with insufficient digital text."""
        rect = page.rect
        w = float(rect.width)
        h = float(rect.height)

        words_data = page.get_text("words")

        # Fallback to EasyOCR for scanned / image-only PDF pages (PRD Hybrid OCR requirement)
        if len(words_data) < 5:
            logger.info("ocr.fallback_to_easyocr", page_number=page_number, words_found=len(words_data))
            pix = page.get_pixmap(dpi=200, alpha=False)
            img_bytes = canonical_image_bytes or pix.tobytes("png")
            ocr_res = self.process_image(img_bytes, page_number, quality)
            ocr_res.fallback_used = True
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

            norm_bbox = (
                lx0 / w,
                ly0 / h,
                lx1 / w,
                ly1 / h,
            )

            tokens: list[OCRToken] = []
            token_confs: list[float] = []
            for item in w_items:
                tx0, ty0, tx1, ty1, tw = item
                t_norm = (tx0 / w, ty0 / h, tx1 / w, ty1 / h)
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

    def process_image(
        self,
        image_bytes: bytes | str | Path,
        page_number: int,
        quality: dict,
    ) -> PageOCREvidence:
        """Process raster image / scanned document using EasyOCR with GPU acceleration."""
        if isinstance(image_bytes, (str, Path)):
            image_path = Path(image_bytes)
            image_bytes = image_path.read_bytes()
        cv_img = ImagePreprocessor.bytes_to_cv2(image_bytes)
        h, w = cv_img.shape[:2]

        ocr_lines: list[OCRLine] = []
        engine_used = "easyocr_gpu"

        if self.reader is not None:
            try:
                ocr_lines = self._read_image_lines(cv_img, w, h)
                # A second OCR pass is deliberately limited to low-confidence pages.
                # Preserve the better result so preprocessing cannot degrade clean scans.
                if self._average_line_confidence(ocr_lines) < 0.55:
                    enhanced = ImagePreprocessor.enhance_document_for_ocr(cv_img)
                    enhanced_lines = self._read_image_lines(enhanced, w, h)
                    if self._average_line_confidence(enhanced_lines) > self._average_line_confidence(ocr_lines):
                        ocr_lines = enhanced_lines
                        quality = {**quality, "ocr_preprocessing": "clahe_otsu_retry"}
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

    def _detect_tables(self, lines: list[OCRLine]) -> list[TableStructure]:
        """Detect tabular line items and structural grid."""
        tables: list[TableStructure] = []
        table_lines = [line for line in lines if 0.32 <= line.bbox_norm[1] <= 0.76 and len(line.tokens) >= 3]
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
