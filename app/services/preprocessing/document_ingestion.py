"""
document_ingestion.py

Document Ingestion and Multi-Page Handling for 3PL documents (PDF, JPG, PNG).
Features:
- SHA-256 source hash computation for immutable evidence tracking.
- Multi-page PDF page splitting and rendering to high-res pixmaps.
- Image validation and dimension extraction.
"""

from __future__ import annotations
import hashlib
import io
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageSequence
import numpy as np

from app.core.config import settings


CANONICAL_DPI = 200


class DocumentPageAsset:
    def __init__(self, page_number: int, width: int, height: int, image_bytes: bytes, is_pdf_page: bool = False, pdf_page: fitz.Page | None = None, page_image: dict | None = None):
        self.page_number = page_number
        self.width = width
        self.height = height
        self.image_bytes = image_bytes
        self.is_pdf_page = is_pdf_page
        self.pdf_page = pdf_page
        self.page_image = page_image or {}


class IngestedDocument:
    def __init__(self, document_id: str, source_filename: str, source_sha256: str, mime_type: str, raw_bytes: bytes, pages: list[DocumentPageAsset]):
        self.document_id = document_id
        self.source_filename = source_filename
        self.source_sha256 = source_sha256
        self.mime_type = mime_type
        self.raw_bytes = raw_bytes
        self.pages = pages


def render_canonical_pages(raw_bytes: bytes, source_filename: str, document_id: str, output_root: Path | None = None, dpi: int = CANONICAL_DPI) -> list[DocumentPageAsset]:
    """Create the immutable PNG coordinate space used by viewers and OCR evidence."""
    target = Path(output_root if output_root is not None else settings.page_output_dir) / document_id
    target.mkdir(parents=True, exist_ok=True)
    is_pdf = raw_bytes.startswith(b"%PDF") or source_filename.lower().endswith(".pdf")
    pages: list[DocumentPageAsset] = []

    def add(page_number: int, image: Image.Image, pdf_page: fitz.Page | None = None, rotation: int = 0) -> None:
        canonical = image.convert("RGB")
        path = target / f"page_{page_number:04d}.png"
        canonical.save(path, format="PNG")
        image_bytes = path.read_bytes()
        pages.append(DocumentPageAsset(page_number, canonical.width, canonical.height, image_bytes, pdf_page is not None, pdf_page, {
            "path": path.as_posix(), "width": canonical.width, "height": canonical.height,
            "dpi": dpi, "rotation_applied": rotation, "sha256": hashlib.sha256(image_bytes).hexdigest(),
        }))

    if is_pdf:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        for index, page in enumerate(doc):
            # get_pixmap applies the PDF display rotation and CropBox, matching viewers.
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            add(index + 1, Image.open(io.BytesIO(pix.tobytes("png"))), page, page.rotation)
    else:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            for index, frame in enumerate(ImageSequence.Iterator(source)):
                # EXIF transpose is deliberately performed before PNG persistence.
                add(index + 1, ImageOps.exif_transpose(frame).convert("RGB"), rotation=0)
    return pages


class DocumentIngestionService:
    @staticmethod
    def ingest_file(file_path: Path | str, document_id: str | None = None) -> IngestedDocument:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Source document not found: {p}")
        raw_bytes = p.read_bytes()
        return DocumentIngestionService.ingest_bytes(raw_bytes, source_filename=p.name, document_id=document_id)

    @staticmethod
    def ingest_bytes(raw_bytes: bytes, source_filename: str = "document.pdf", document_id: str | None = None, output_root: Path | None = None, dpi: int = CANONICAL_DPI) -> IngestedDocument:
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        doc_id = document_id or f"DOC-{sha256[:16]}"
        is_pdf = raw_bytes.startswith(b"%PDF") or source_filename.lower().endswith(".pdf")

        pages = render_canonical_pages(raw_bytes, source_filename, doc_id, output_root, dpi)
        mime_type = "application/pdf" if is_pdf else "image/png"

        return IngestedDocument(
            document_id=doc_id,
            source_filename=source_filename,
            source_sha256=sha256,
            mime_type=mime_type,
            raw_bytes=raw_bytes,
            pages=pages,
        )
