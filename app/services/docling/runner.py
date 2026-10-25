"""Lazy, process-wide Docling runner: the single document-understanding backbone of the pipeline.

* Pages with a text layer are converted WITHOUT OCR (Docling reads the PDF text directly).
* Image pages (and PDF pages without any text) are converted WITH OCR (RapidOCR, Docling's internal OCR engine) from
  the canonical page PNG.

One conversion yields everything the pipeline needs (text lines, words, tables), so the OCR engine and the table
provider share a small result cache keyed by the payload: a page is converted once, not once per consumer.

Docling is imported lazily; a missing install or any conversion error returns ``None`` (the page then has no
Docling evidence) instead of failing the request.
"""
from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_CACHE_SIZE = 4


@dataclass
class DoclingResult:
    tables: list[Any]  # docling TableItem
    texts: list[Any]  # docling TextItem (paragraph-level blocks)
    page_width: float
    page_height: float
    used_ocr: bool
    # Visual text lines / words of the page (docling TextCell, TOPLEFT or BOTTOMLEFT origin, page units).
    # Lines are the evidence granularity; ``texts`` merges lines into paragraphs and omits table cell text.
    line_cells: list[Any] = field(default_factory=list)
    word_cells: list[Any] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.line_cells or self.texts)


class DoclingRunner:
    def __init__(self) -> None:
        self._converters: dict[bool, Any] = {}
        self._lock = threading.Lock()
        self._unavailable = False
        self._cache: OrderedDict[tuple[str, bool, bool], DoclingResult] = OrderedDict()

    def _build(self, ocr: bool) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions, RapidOcrOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = ocr
        options.do_table_structure = True
        options.generate_parsed_pages = True  # exposes per-line / per-word cells with confidence
        options.table_structure_options.mode = TableFormerMode.ACCURATE if settings.docling_table_mode.lower() == "accurate" else TableFormerMode.FAST
        options.accelerator_options = AcceleratorOptions(num_threads=settings.docling_num_threads, device="cpu")
        if ocr:
            options.ocr_options = RapidOcrOptions()  # RapidOCR (onnxruntime) instead of Docling's default OCR engine
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
        except Exception as exc:
            # A dependency mismatch can also fail while Docling is imported
            # (not only when the package is missing).  In particular, older
            # Pydantic releases raise AttributeError while Docling builds its
            # OCR option models.  Treat it exactly like an unavailable engine
            # so the request can use the deterministic PDF fallback instead of
            # returning a 500 to the upload flow.
            self._unavailable = True
            logger.warning("docling.unavailable", error=str(exc), hint="install the pinned AI-service dependencies")
            return None
        key = (hashlib.sha1(payload).hexdigest(), is_pdf, ocr)
        try:
            with self._lock:  # one conversion at a time: the models are not thread-safe and already use all threads
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
                converter = self._converter(ocr)
                name = "page.pdf" if is_pdf else "page.png"
                result = converter.convert(DocumentStream(name=name, stream=io.BytesIO(payload)))
                converted = _to_result(result, ocr)
                if converted is not None:
                    self._cache[key] = converted
                    while len(self._cache) > _CACHE_SIZE:
                        self._cache.popitem(last=False)
                return converted
        except Exception as exc:  # any Docling failure must degrade to "no evidence", never fail the request
            logger.warning("docling.convert_failed", error=str(exc), ocr=ocr)
            return None

    def convert_page(self, *, pdf_bytes: bytes | None, image_loader: Callable[[], bytes | None]) -> DoclingResult | None:
        """Convert one page: vector text when a one-page PDF is given, RapidOCR from the page PNG otherwise.

        A PDF page without any text layer (a scan embedded in a PDF) falls back to OCR on the page PNG.
        """
        if pdf_bytes is not None:
            result = self.convert(pdf_bytes, is_pdf=True, ocr=False)
            if result is not None and result.has_text:
                return result
            # A vector PDF can be recovered cheaply by the OCR engine's PyMuPDF
            # fallback. Do not rasterize it and initialize all OCR/layout models
            # just because the optional native Docling PDF backend is unavailable.
            if _pdf_has_text(pdf_bytes):
                return result
        image = image_loader()
        if image is None:
            return None
        return self.convert(image, is_pdf=False, ocr=True)


def _to_result(result: Any, ocr: bool) -> DoclingResult | None:
    document = result.document
    page = next(iter(document.pages.values()), None)
    if page is None:
        return None
    parsed = result.pages[0].parsed_page if result.pages else None
    lines = list(parsed.textline_cells) if parsed is not None else []
    words = list(parsed.word_cells) if parsed is not None else []
    return DoclingResult(list(document.tables), list(document.texts), float(page.size.width), float(page.size.height), ocr, lines, words)


def _pdf_has_text(payload: bytes) -> bool:
    """Check for a vector text layer before taking the expensive raster OCR path."""
    try:
        import fitz

        document = fitz.open(stream=payload, filetype="pdf")
        return any(page.get_text().strip() for page in document)
    except Exception:
        return False


_RUNNER: DoclingRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> DoclingRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = DoclingRunner()
        return _RUNNER
