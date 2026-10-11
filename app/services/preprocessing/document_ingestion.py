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
from PIL import Image
import numpy as np


class DocumentPageAsset:
    def __init__(self, page_number: int, width: int, height: int, image_bytes: bytes, is_pdf_page: bool = False, pdf_page: fitz.Page | None = None):
        self.page_number = page_number
        self.width = width
        self.height = height
        self.image_bytes = image_bytes
        self.is_pdf_page = is_pdf_page
        self.pdf_page = pdf_page


class IngestedDocument:
    def __init__(self, document_id: str, source_filename: str, source_sha256: str, mime_type: str, raw_bytes: bytes, pages: list[DocumentPageAsset]):
        self.document_id = document_id
        self.source_filename = source_filename
        self.source_sha256 = source_sha256
        self.mime_type = mime_type
        self.raw_bytes = raw_bytes
        self.pages = pages


class DocumentIngestionService:
    @staticmethod
    def ingest_file(file_path: Path | str, document_id: str | None = None) -> IngestedDocument:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Source document not found: {p}")
        raw_bytes = p.read_bytes()
        return DocumentIngestionService.ingest_bytes(raw_bytes, source_filename=p.name, document_id=document_id)

    @staticmethod
    def ingest_bytes(raw_bytes: bytes, source_filename: str = "document.pdf", document_id: str | None = None) -> IngestedDocument:
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        doc_id = document_id or f"DOC-{sha256[:16]}"
        is_pdf = raw_bytes.startswith(b"%PDF") or source_filename.lower().endswith(".pdf")

        pages: list[DocumentPageAsset] = []
        if is_pdf:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            for pno in range(len(doc)):
                page = doc[pno]
                # Canonical evidence is always based on the exact raster sent to OCR.
                # 300 DPI preserves small invoice text and handwritten pen strokes.
                pix = page.get_pixmap(dpi=300, alpha=False)
                img_bytes = pix.tobytes("png")
                pages.append(
                    DocumentPageAsset(
                        page_number=pno + 1,
                        width=pix.width,
                        height=pix.height,
                        image_bytes=img_bytes,
                        is_pdf_page=True,
                        pdf_page=page,
                    )
                )
            mime_type = "application/pdf"
        else:
            # Raster Image (JPG/PNG)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            w, h = pil_img.size
            pages.append(
                DocumentPageAsset(
                    page_number=1,
                    width=w,
                    height=h,
                    image_bytes=raw_bytes,
                    is_pdf_page=False,
                    pdf_page=None,
                )
            )
            mime_type = "image/jpeg" if source_filename.lower().endswith((".jpg", ".jpeg")) else "image/png"

        return IngestedDocument(
            document_id=doc_id,
            source_filename=source_filename,
            source_sha256=sha256,
            mime_type=mime_type,
            raw_bytes=raw_bytes,
            pages=pages,
        )
