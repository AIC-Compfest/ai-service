"""Docling table provider with per-page routing (steps 1, 2a, 2b and 4 of the Docling flow).

Routing uses ``detect_page_kind`` on the live PDF page: pages that carry a text layer (DIGITAL, MIXED,
HYBRID_OCR_LAYER) are converted without OCR from a one-page PDF; SCAN pages (and image uploads, which have no PDF
page) are converted with OCR from the canonical page PNG. Failures return no tables so the router falls back.
"""
from __future__ import annotations

import re

from app.core.config import settings
from app.core.logging import get_logger
from app.services.candidates.schema import Page, Table
from app.services.docling.adapter import docling_table_to_internal
from app.services.docling.runner import get_runner
from app.services.preprocessing.page_kind import PageKind, detect_page_kind

from .refine import refine_pymupdf_table
from .rows import summary_pattern

logger = get_logger(__name__)

DEFAULT_OPTIONS = {"min_cols": 2, "min_data_rows": 1}


class DoclingTableProvider:
    name = "docling"

    def __init__(self, options: dict | None = None):
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        labels = self.options.get("summary_labels") or []
        self.summary: re.Pattern | None = summary_pattern(list(labels)) if labels else None

    def route(self, page: Page) -> tuple[str, bytes | None, bool, bool]:
        """Return ``(kind, payload, is_pdf, ocr)``; payload None = nothing to convert."""
        source = page.source_page
        if source is not None and hasattr(source, "parent") and hasattr(source, "number"):
            kind = detect_page_kind(source).kind
            if kind != PageKind.SCAN:
                import fitz  # one-page PDF so Docling never re-parses the whole document per page

                single = fitz.open()
                single.insert_pdf(source.parent, from_page=source.number, to_page=source.number)
                return kind, single.tobytes(), True, False
            image = self._page_png(page)
            return kind, image, False, True
        return PageKind.SCAN, self._page_png(page), False, True

    @staticmethod
    def _page_png(page: Page) -> bytes | None:
        if page.image_path:
            try:
                with open(page.image_path, "rb") as handle:
                    return handle.read()
            except OSError:
                pass
        source = page.source_page
        if source is not None and hasattr(source, "get_pixmap"):
            return source.get_pixmap(dpi=200, alpha=False).tobytes("png")
        return None

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        kind, payload, is_pdf, ocr = self.route(page)
        if payload is None or (ocr and not settings.docling_ocr_scans):
            return []
        result = get_runner().convert(payload, is_pdf=is_pdf, ocr=ocr)
        if result is None:
            return []
        tables: list[Table] = []
        for item in result.tables:
            table = docling_table_to_internal(
                item,
                page_width=result.page_width,
                page_height=result.page_height,
                page_number=page.page_number,
                document_id=document_id,
                ordinal=len(tables) + 1,
                reason=f"docling:{'ocr' if ocr else 'text'}",
                min_cols=int(self.options["min_cols"]),
                min_data_rows=int(self.options["min_data_rows"]),
            )
            if table is not None:
                refine_pymupdf_table(table, self.summary, self.options.get("refine"))  # same TOTAL-row / header clean-up
                table.validation["page_kind"] = kind
                tables.append(table)
        return tables
