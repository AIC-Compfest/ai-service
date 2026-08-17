"""
extract.py — Extraction API endpoints.

Phase 6 Integration: All endpoints now route through Phase 5 DocumentPipeline
for robust OCR+layout before invoking the appropriate Phase 6 Extractor.

Endpoints:
  POST /v1/extract/{document_type}   — typed extraction (known doc type)
  POST /v1/extract/understand/evidence — Phase 5 evidence-only (no extraction)
  POST /v1/extract/pipeline/full     — auto-classify + Phase 6 extraction
"""

import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.logging import get_logger
from app.schemas.document_type import DocumentType
from app.schemas.extraction import ExtractionResponse
from app.services.extraction.registry import get_extractor
from app.services.pipeline import Phase5DocumentPipeline

router = APIRouter(prefix="/v1/extract", tags=["extraction"])
logger = get_logger(__name__)


@router.post("/understand/evidence")
async def understand_document_evidence(file: UploadFile) -> dict:
    """
    Phase 5 Evidence Endpoint.
    Returns full structured OCR evidence: lines, bboxes, layout regions, tables,
    classification result, and signature detection. No field extraction is performed.
    """
    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    logger.info("extract.evidence.request", document_id=document_id, filename=file.filename)
    pipeline = Phase5DocumentPipeline()
    return pipeline.process_document(file_bytes, filename=file.filename or "doc.pdf", document_id=document_id)


@router.post("/pipeline/full")
async def extract_full_pipeline(file: UploadFile) -> dict:
    """
    Phase 6 Unified Auto-Classify + Extract Endpoint (PRD §23).

    1. Runs Phase 5: document ingestion, OCR, layout understanding, classification,
       signature detection.
    2. Routes to correct Phase 6 extractor based on classification result.
    3. Returns structured extracted fields with full EvidenceSource (PRD §18).

    If classification is UNKNOWN or confidence < 0.70 → returns
    REQUIRES_HUMAN_REVIEW status without attempting extraction (PRD §20, §60).
    """
    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    filename = file.filename or "doc.pdf"

    logger.info("extract.pipeline.request", document_id=document_id, filename=filename)

    pipeline = Phase5DocumentPipeline()
    evidence = pipeline.process_document(file_bytes, filename=filename, document_id=document_id)

    classification = evidence.get("classification", {})
    doc_type_str = classification.get("document_type", "UNKNOWN")
    confidence = float(classification.get("confidence", 0.0))
    requires_review = classification.get("requires_human_review", True)

    try:
        doc_type = DocumentType(doc_type_str)
    except ValueError:
        doc_type = DocumentType.UNKNOWN

    # Guard: cannot extract UNKNOWN or low-confidence docs (PRD §20, §7.1)
    if doc_type == DocumentType.UNKNOWN or requires_review:
        error_code = (
            "LOW_CLASSIFICATION_CONFIDENCE" if confidence < 0.70
            else "UNKNOWN_DOCUMENT_TYPE"
        )
        logger.warning(
            "extract.pipeline.skipped",
            document_id=document_id,
            doc_type=doc_type_str,
            confidence=confidence,
            error_code=error_code,
        )
        return {
            "document_id": document_id,
            "status": "REQUIRES_HUMAN_REVIEW",
            "classification": classification,
            "extracted_fields": [],
            "extraction_attempted": False,
            "error_code": error_code,
        }

    # Phase 6: Structured extraction
    extractor = get_extractor(doc_type)
    fields = extractor.extract(document_id, evidence)

    unknown_fields = [f.field for f in fields if f.is_unknown]
    final_requires_review = len(unknown_fields) > 0 or confidence < 0.95

    logger.info(
        "extract.pipeline.complete",
        document_id=document_id,
        doc_type=doc_type_str,
        total_fields=len(fields),
        unknown_count=len(unknown_fields),
        requires_human_review=final_requires_review,
    )

    return {
        "document_id": document_id,
        "status": "SUCCESS",
        "classification": classification,
        "extracted_fields": [f.model_dump() for f in fields],
        "extraction_attempted": True,
        "unknown_fields": unknown_fields,
        "requires_human_review": final_requires_review,
        "pipeline_version": "phase6_v1",
    }


@router.post("/{document_type}", response_model=ExtractionResponse)
async def extract_document(document_type: DocumentType, file: UploadFile) -> ExtractionResponse:
    """
    Typed extraction endpoint. Caller specifies document type explicitly.
    Uses Phase 5 pipeline for OCR then Phase 6 extractor for field extraction.
    UNKNOWN document type is rejected — route to human review first (PRD §20).
    """
    if document_type == DocumentType.UNKNOWN:
        raise HTTPException(
            status_code=422,
            detail="Cannot extract UNKNOWN document type; route to human review first (PRD §20).",
        )

    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    logger.info(
        "extract.typed.request",
        document_id=document_id,
        document_type=document_type.value,
        filename=file.filename,
    )

    # Phase 5 pipeline
    pipeline = Phase5DocumentPipeline()
    evidence = pipeline.process_document(
        file_bytes, filename=file.filename or "doc.pdf", document_id=document_id
    )

    # Phase 6 extractor
    extractor = get_extractor(document_type)
    fields = extractor.extract(document_id, evidence)

    unknown_fields = [f.field for f in fields if f.is_unknown]
    requires_review = len(unknown_fields) > 0

    logger.info(
        "extract.typed.complete",
        document_id=document_id,
        doc_type=document_type.value,
        total_fields=len(fields),
        unknown_count=len(unknown_fields),
    )

    pages_meta = evidence.get("pages", [{}])
    engine_used = pages_meta[0].get("engine_used", "phase5_pipeline") if pages_meta else "phase5_pipeline"
    fallback = pages_meta[0].get("fallback_used", False) if pages_meta else False

    return ExtractionResponse(
        document_id=document_id,
        document_type=document_type,
        fields=fields,
        ocr_engine_used=engine_used,
        fallback_triggered=fallback,
        status="SUCCESS",
        requires_human_review=requires_review,
    )
