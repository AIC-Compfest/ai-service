"""Lazy, process-wide Docling runner (steps 2a/2b of the Docling flow).

* 2a: pages with a text layer are converted WITHOUT OCR (Docling reads the PDF text directly).
* 2b: pages routed as scans are converted WITH OCR from the canonical page PNG.

Docling is an optional dependency: importing this module never imports it, and a missing install or any conversion
error returns ``None`` so the router falls back to the other providers.
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DoclingResult:
    tables: list[Any]  # docling TableItem
    page_width: float
    page_height: float
    used_ocr: bool


class DoclingRunner:
    def __init__(self) -> None:
        self._converters: dict[bool, Any] = {}
        self._lock = threading.Lock()
        self._unavailable = False

    def _build(self, ocr: bool) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = ocr
        options.do_table_structure = True
        options.table_structure_options.mode = TableFormerMode.ACCURATE if settings.docling_table_mode.lower() == "accurate" else TableFormerMode.FAST
        options.accelerator_options = AcceleratorOptions(num_threads=settings.docling_num_threads, device="cpu")
        if settings.docling_artifacts_path:
            options.artifacts_path = settings.docling_artifacts_path
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options), InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)})

    def _converter(self, ocr: bool) -> Any:
        if ocr not in self._converters:
            self._converters[ocr] = self._build(ocr)
        return self._converters[ocr]

    def convert(self, payload: bytes, *, is_pdf: bool, ocr: bool) -> DoclingResult | None:
        if self._unavailable:
            return None
        try:
            from docling.datamodel.base_models import DocumentStream
        except ImportError:
            self._unavailable = True
            logger.warning("docling.not_installed", hint="pip install -r requirements-docling.txt")
            return None
        try:
            with self._lock:  # one conversion at a time: the models are not thread-safe and already use all threads
                converter = self._converter(ocr)
                name = "page.pdf" if is_pdf else "page.png"
                result = converter.convert(DocumentStream(name=name, stream=io.BytesIO(payload)))
            document = result.document
            page = next(iter(document.pages.values()), None)
            if page is None:
                return None
            return DoclingResult(list(document.tables), float(page.size.width), float(page.size.height), ocr)
        except Exception as exc:  # any Docling failure must degrade to the fallback providers, never fail the request
            logger.warning("docling.convert_failed", error=str(exc), ocr=ocr)
            return None


_RUNNER: DoclingRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> DoclingRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = DoclingRunner()
        return _RUNNER
