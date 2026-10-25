"""Docling TableFormer table provider, the only table provider.

Routing is decided by what the page carries: a page with ``pdf_bytes`` (PDF upload) is converted from its one-page PDF
without OCR (DIGITAL); image uploads, and PDF pages without a text layer, are converted with RapidOCR OCR from the
canonical page PNG (SCAN). The conversion is shared with the OCR engine through the runner cache, so a page is
converted once. Failures return no tables.
"""
from __future__ import annotations

import re

from app.core.logging import get_logger
from app.services.candidates.schema import Page, Table
from app.services.docling.adapter import docling_table_to_internal
from app.services.docling.runner import get_runner

from .refine import refine_pymupdf_table
from .rows import summary_pattern

logger = get_logger(__name__)

DEFAULT_OPTIONS = {"min_cols": 2, "min_data_rows": 1}
DIGITAL, SCAN = "DIGITAL", "SCAN"


class DoclingTableProvider:
    name = "docling"

    def __init__(self, options: dict | None = None):
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        labels = self.options.get("summary_labels") or []
        self.summary: re.Pattern | None = summary_pattern(list(labels)) if labels else None

    def route(self, page: Page) -> tuple[str, bytes | None, bool, bool]:
        """Return ``(kind, payload, is_pdf, ocr)``; payload None = nothing to convert."""
        if getattr(page, "pdf_bytes", None) is not None:
            return DIGITAL, page.pdf_bytes, True, False
        return SCAN, self._page_png(page), False, True

    @staticmethod
    def _page_png(page: Page) -> bytes | None:
        if page.image_path:
            try:
                with open(page.image_path, "rb") as handle:
                    return handle.read()
            except OSError:
                pass
        return None

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        kind, payload, is_pdf, _ = self.route(page)
        if payload is None:
            return []
        runner = get_runner()
        if is_pdf:
            result = runner.convert_page(pdf_bytes=payload, image_loader=lambda: self._page_png(page))
        else:
            result = runner.convert(payload, is_pdf=False, ocr=True)
        if result is None:
            return []
        ocr = result.used_ocr
        kind = SCAN if ocr else DIGITAL
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
