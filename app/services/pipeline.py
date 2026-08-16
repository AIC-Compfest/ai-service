"""
pipeline.py

Phase 5: Complete Document Understanding & OCR Pipeline Orchestrator (PRD §18, §19, §20, §21, §22, §33).
Pipeline:
RAW DOCUMENT (PDF/JPG/PNG)
     ↓
INGESTION (SHA-256 + Multi-page splitting)
     ↓
PREPROCESSING & QUALITY ESTIMATION (Blur, Brightness, Contrast, Skew)
     ↓
DOCUMENT CLASSIFICATION (Invoice, Surat Jalan, POD, Rate Agreement, Unknown)
     ↓
CASCADED OCR & LAYOUT UNDERSTANDING (Lines, BBoxes, Tables, Structural Regions)
     ↓
SIGNATURE & STAMP DETECTION
     ↓
STRUCTURED DOCUMENT EVIDENCE JSON (Ready for Phase 6 Extraction)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List

from app.core.logging import get_logger
from .preprocessing.document_ingestion import DocumentIngestionService, IngestedDocument
from .preprocessing.image_preprocessor import ImagePreprocessor
from .classification.classifier import DocumentClassifier
from .ocr.engine import OCREngine, PageOCREvidence
from .signature.detector import SignatureDetector

logger = get_logger(__name__)


class Phase5DocumentPipeline:
    def __init__(self):
        self.ingestion = DocumentIngestionService()
        self.preprocessor = ImagePreprocessor()
        self.classifier = DocumentClassifier()
        self.ocr_engine = OCREngine()
        self.signature_detector = SignatureDetector()

    def process_document(self, file_path_or_bytes: Path | str | bytes, filename: str = "document.pdf", document_id: str | None = None) -> dict:
        start_time = time.perf_counter()

        # 1. Ingestion
        if isinstance(file_path_or_bytes, (str, Path)):
            ingested = self.ingestion.ingest_file(file_path_or_bytes, document_id=document_id)
        else:
            ingested = self.ingestion.ingest_bytes(file_path_or_bytes, source_filename=filename, document_id=document_id)

        # 2. Process each page (Preprocessing + OCR + Layout)
        pages_evidence: list[dict] = []
        all_text = ""

        for p_asset in ingested.pages:
            # Quality estimation
            cv_img = ImagePreprocessor.bytes_to_cv2(p_asset.image_bytes)
            quality = self.preprocessor.estimate_quality(cv_img)

            # OCR & Layout
            if p_asset.is_pdf_page and p_asset.pdf_page is not None:
                p_ocr = self.ocr_engine.process_pdf_page(p_asset.pdf_page, p_asset.page_number, quality)
            else:
                p_ocr = self.ocr_engine.process_image(p_asset.image_bytes, p_asset.page_number, quality)

            all_text += p_ocr.full_text() + "\n"
            pages_evidence.append(p_ocr.to_dict())

        # 3. Document Classification
        classify_res = self.classifier.classify_text(all_text, document_id=ingested.document_id)

        # 4. Signature & Stamp Detection on last page / POD / Surat Jalan
        sig_res = None
        if len(ingested.pages) > 0:
            last_page = ingested.pages[-1]
            sig_res = self.signature_detector.detect(ingested.document_id, last_page.image_bytes)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 5. Formulate Canonical Structured Document Evidence (PRD §18)
        evidence = {
            "document_id": ingested.document_id,
            "source_file": ingested.source_filename,
            "source_sha256": ingested.source_sha256,
            "mime_type": ingested.mime_type,
            "page_count": len(ingested.pages),
            "classification": {
                "document_type": classify_res.document_type.value,
                "confidence": classify_res.confidence,
                "requires_human_review": classify_res.requires_human_review,
            },
            "signature_detection": {
                "signature_present": sig_res.signature_present if sig_res else False,
                "confidence": sig_res.confidence if sig_res else 0.0,
                "bbox": sig_res.bbox.model_dump() if sig_res and sig_res.bbox else None,
            },
            "pages": pages_evidence,
            "processing_metadata": {
                "pipeline_version": "phase_5_v1",
                "processing_time_ms": elapsed_ms,
            },
        }

        logger.info(
            "document_understanding_completed",
            document_id=ingested.document_id,
            doc_type=classify_res.document_type.value,
            pages=len(ingested.pages),
            time_ms=elapsed_ms,
        )

        return evidence
