"""
pipeline.py

Phase 5: Complete Document Understanding & OCR Pipeline Orchestrator.
Pipeline:
RAW DOCUMENT (PDF/JPG/PNG)
     ↓
INGESTION (SHA-256 + Multi-page splitting)
     ↓
PREPROCESSING & QUALITY ESTIMATION (Blur, Brightness, Contrast, Skew)
     ↓
PAGE-LEVEL CLASSIFICATION (Invoice, Surat Jalan, POD, Rate Agreement, Unknown)
     ↓
SEGMENT BOUNDARY DETECTION (DocumentSegmentationService) ← NEW
     ↓
CASCADED OCR & LAYOUT UNDERSTANDING (Lines, BBoxes, Tables, Structural Regions)
     ↓
SIGNATURE & STAMP DETECTION
     ↓
STRUCTURED DOCUMENT EVIDENCE JSON with SEGMENTS (Ready for Phase 6 Extraction)

Key fix (ARCHITECTURE_AUDIT §1):
  - Previously: pipeline grouped all pages of same type into one flat logical_group.
    No boundary detection → Invoice on page 1 and Invoice on page 9 merged into one.
  - Now: DocumentSegmentationService detects TYPE TRANSITIONS (boundaries).
    Invoice p1-2, SuratJalan p3-4, Invoice p5-6 → 3 separate segments.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.core.logging import get_logger
from .preprocessing.document_ingestion import DocumentIngestionService, IngestedDocument
from .preprocessing.image_preprocessor import ImagePreprocessor
from .classification.classifier import DocumentClassifier
from .candidates.table_runtime import candidate_page, discover_page_tables, link_segment_tables, tables_to_evidence
from .ocr.docling_engine import DoclingOCREngine
from .signature.detector import SignatureDetector
from .segmentation.segmenter import (
    DocumentSegmentationService,
    PageClassification,
)

logger = get_logger(__name__)


class Phase5DocumentPipeline:
    def __init__(self):
        self.ingestion = DocumentIngestionService()
        self.preprocessor = ImagePreprocessor()
        self.classifier = DocumentClassifier()
        self.ocr_engine = DoclingOCREngine()
        self.signature_detector = SignatureDetector()
        self.segmenter = DocumentSegmentationService()

    def process_document(
        self,
        file_path_or_bytes: Path | str | bytes,
        filename: str = "document.pdf",
        document_id: str | None = None,
    ) -> dict:
        start_time = time.perf_counter()

        # 1. Ingestion
        if isinstance(file_path_or_bytes, (str, Path)):
            ingested = self.ingestion.ingest_file(file_path_or_bytes, document_id=document_id)
        else:
            ingested = self.ingestion.ingest_bytes(
                file_path_or_bytes, source_filename=filename, document_id=document_id
            )

        # 2. Process each page (Preprocessing + Classification + OCR + Layout)
        pages_evidence: list[dict] = []
        all_text = ""

        # Legacy group buckets (kept for backward compat with existing callers)
        logical_groups: dict[str, list[int]] = {
            "INVOICE": [],
            "SURAT_JALAN": [],
            "POD": [],
            "RATE_AGREEMENT": [],
            "UNKNOWN": [],
        }

        # Per-page classification results for segmentation
        page_classifications: list[PageClassification] = []
        # Table discovery is type-aware (K1): it runs AFTER page classification with the live PDF page, so OCR is told
        # not to run its type-less discovery (that path stays for stand-alone OCR callers).
        candidate_pages: dict[int, Any] = {}
        page_tables: dict[int, list] = {}

        for p_asset in ingested.pages:
            # Quality estimation
            cv_img = ImagePreprocessor.bytes_to_cv2(p_asset.image_bytes)
            quality = self.preprocessor.estimate_quality(cv_img)

            # Text & layout via Docling: PDF pages are read from their text layer (one-page PDF), image pages are
            # OCR-ed by RapidOCR from the canonical PNG. Coordinates are normalized [0, 1], so they are valid in the
            # canonical PNG space that viewers use.
            if p_asset.is_pdf_page and p_asset.pdf_bytes is not None:
                page_payload, is_pdf = p_asset.pdf_bytes, True
            else:
                page_payload, is_pdf = p_asset.image_bytes, False
            page_dict = self.ocr_engine.process(
                page_payload,
                is_pdf=is_pdf,
                page_number=p_asset.page_number,
                quality=quality,
                page_width=p_asset.width,
                page_height=p_asset.height,
                fallback_image_bytes=p_asset.image_bytes,
            )
            # Evidence dimensions always describe the canonical PNG, including vector PDFs.
            page_dict["width"], page_dict["height"] = p_asset.width, p_asset.height

            page_text = "\n".join(line["text"] for line in page_dict["lines"])
            all_text += page_text + "\n"

            # Page-Level Classification (per PRD §18, §20)
            page_classify_res = self.classifier.classify_text(
                page_text, document_id=f"{ingested.document_id}_p{p_asset.page_number}"
            )
            page_type_str = page_classify_res.document_type.value

            # Collect for segmenter
            page_classifications.append(
                PageClassification(
                    page_number=p_asset.page_number,
                    document_type=page_type_str,
                    confidence=page_classify_res.confidence,
                    requires_human_review=page_classify_res.requires_human_review,
                    method=page_classify_res.classification_method,
                )
            )

            # Legacy group (for backward compat)
            if page_type_str in logical_groups:
                logical_groups[page_type_str].append(p_asset.page_number)
            else:
                logical_groups["UNKNOWN"].append(p_asset.page_number)

            # Page-Level Signature & Stamp Detection
            page_sig_res = self.signature_detector.detect(
                f"{ingested.document_id}_p{p_asset.page_number}", p_asset.image_bytes
            )

            page_dict["page_image"] = p_asset.page_image
            page_dict["coord_space"] = "page_image_normalized"
            if settings.enable_table_extractor:
                candidate = candidate_page(page_dict, pdf_bytes=p_asset.pdf_bytes if p_asset.is_pdf_page else None)
                found = discover_page_tables(candidate, page_type_str, ingested.document_id)
                candidate_pages[p_asset.page_number] = candidate
                page_tables[p_asset.page_number] = found
                page_dict["tables"] = tables_to_evidence(found)
            page_dict["classification"] = {
                "document_type": page_type_str,
                "confidence": page_classify_res.confidence,
                "probabilities": page_classify_res.probabilities,
                "requires_human_review": page_classify_res.requires_human_review,
                "model_version": page_classify_res.model_version,
                "method": page_classify_res.classification_method,
            }
            page_dict["signature_detection"] = {
                "signature_present": page_sig_res.signature_present if page_sig_res else False,
                "confidence": page_sig_res.confidence if page_sig_res else 0.0,
                "bbox": page_sig_res.bbox.model_dump() if page_sig_res and page_sig_res.bbox else None,
            }
            pages_evidence.append(page_dict)

        # -----------------------------------------------------------------------
        # 3. SEGMENT BOUNDARY DETECTION (ARCHITECTURE_AUDIT ROOT CAUSE A FIX)
        # This replaces simple logical_groups with proper segment objects
        # that have contiguous page ranges and boundary-detected type transitions.
        # -----------------------------------------------------------------------
        segments = self.segmenter.segment(page_classifications)
        segments_dicts = [s.to_dict() for s in segments]

        # 3b. Header-less continuations and multi-page tables per segment (additive evidence).
        table_groups: list[dict] = []
        if settings.enable_table_extractor and page_tables:
            page_tables, table_groups = link_segment_tables(candidate_pages, page_tables, segments_dicts, ingested.document_id)
            for page_dict in pages_evidence:
                number = page_dict["page_number"]
                if number in page_tables:
                    page_dict["tables"] = tables_to_evidence(page_tables[number])

        # 4. Overall Document-Level Classification (for backward compat only)
        # NOTE: For multi-type PDFs this will be mixed signal — don't rely on it.
        # Use segments[].document_type as the authoritative classification.
        overall_classify_res = self.classifier.classify_text(all_text, document_id=ingested.document_id)

        # 5. Overall Signature Detection (on last page for POD detection)
        sig_res = None
        if len(ingested.pages) > 0:
            last_page = ingested.pages[-1]
            sig_res = self.signature_detector.detect(ingested.document_id, last_page.image_bytes)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 6. Canonical Structured Document Evidence (PRD §18, §20)
        evidence = {
            "document_id": ingested.document_id,
            "source_file": ingested.source_filename,
            "source_sha256": ingested.source_sha256,
            "mime_type": ingested.mime_type,
            "page_count": len(ingested.pages),

            # AUTHORITATIVE: segments is the new canonical output.
            # Each segment represents one logical document with its page range.
            "segments": segments_dicts,

            # DEPRECATED: logical_document_groups retained for backward compat.
            # Do not use for new code — use segments[] instead.
            # Will be removed in a future pipeline version.
            "logical_document_groups": logical_groups,

            # Document-level classification (aggregate signal — may be mixed for multi-type PDFs)
            # Use segments[].document_type for per-document classification.
            "classification": {
                "document_type": overall_classify_res.document_type.value,
                "confidence": overall_classify_res.confidence,
                "requires_human_review": overall_classify_res.requires_human_review,
                "probabilities": overall_classify_res.probabilities,
                "model_version": overall_classify_res.model_version,
                "method": overall_classify_res.classification_method,
                "note": (
                    "AGGREGATE_SIGNAL_MULTI_DOC"
                    if len(segments_dicts) > 1
                    else "SINGLE_DOC"
                ),
            },
            "signature_detection": {
                "signature_present": sig_res.signature_present if sig_res else False,
                "confidence": sig_res.confidence if sig_res else 0.0,
                "bbox": sig_res.bbox.model_dump() if sig_res and sig_res.bbox else None,
            },
            "pages": pages_evidence,
            "table_groups": table_groups,
            "processing_metadata": {
                "pipeline_version": "phase_5_v3_segment_aware",
                "processing_time_ms": elapsed_ms,
                "model_version": "doc-classifier-v1",
                "segment_count": len(segments_dicts),
                "segmentation_method": "page_boundary_detection",
            },
        }

        logger.info(
            "document_understanding_completed",
            document_id=ingested.document_id,
            pages=len(ingested.pages),
            segment_count=len(segments_dicts),
            segments=[{
                "type": s["document_type"],
                "pages": f"{s['start_page']}-{s['end_page']}",
                "conf": s["classification_confidence"],
            } for s in segments_dicts],
            time_ms=elapsed_ms,
        )

        return evidence
