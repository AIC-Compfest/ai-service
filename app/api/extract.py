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

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from app.core.logging import get_logger
from app.core.runtime import get_pipeline, purge_old_pages, read_upload
from app.schemas.document_type import DocumentType
from app.schemas.extraction import ExtractionResponse
from app.services.extraction.ml.service import append_ml_only_fields, promote_ml_fields, run_ml_extraction
from app.services.extraction.label_value import merge_review_fields
from app.services.extraction.page_resolver import resolve_field_pages
from app.services.extraction.registry import get_extractor
from app.services.validation.document import validate_extraction

router = APIRouter(prefix="/v1/extract", tags=["extraction"])
logger = get_logger(__name__)


@router.post("/understand/evidence")
def understand_document_evidence(file: UploadFile) -> dict:
    """
    Phase 5 Evidence Endpoint.
    Returns full structured OCR evidence: lines, bboxes, layout regions, tables,
    classification result, and signature detection. No field extraction is performed.
    """
    purge_old_pages()
    document_id = str(uuid.uuid4())
    file_bytes = read_upload(file)
    logger.info("extract.evidence.request", document_id=document_id, filename=file.filename)
    pipeline = get_pipeline()
    return pipeline.process_document(file_bytes, filename=file.filename or "doc.pdf", document_id=document_id)


@router.post("/pipeline/full")
def extract_full_pipeline(
    file: UploadFile,
    request: Request,
    document_type: DocumentType | None = Form(default=None),
) -> dict:
    """
    Phase 6 Unified Auto-Classify + Extract Endpoint (PRD §23).

    1. Runs Phase 5: document ingestion, OCR, layout understanding, classification,
       signature detection.
    2. Routes to correct Phase 6 extractor based on classification result.
    3. Returns structured extracted fields with full EvidenceSource (PRD §18).

    If classification is UNKNOWN or confidence < 0.70 → returns
    REQUIRES_HUMAN_REVIEW status without attempting extraction (PRD §20, §60).
    """
    purge_old_pages()
    document_id = str(uuid.uuid4())
    file_bytes = read_upload(file)
    filename = file.filename or "doc.pdf"

    logger.info("extract.pipeline.request", document_id=document_id, filename=filename, correlation_id=request.headers.get("x-correlation-id", ""))

    pipeline = get_pipeline()
    evidence = pipeline.process_document(file_bytes, filename=filename, document_id=document_id)

    classification = dict(evidence.get("classification", {}))
    # A user-selected type is a reviewable routing instruction, not a model
    # prediction. It lets the typed extractor run when the classifier is
    # uncertain, while the original model confidence remains visible to HITL.
    declared_type = document_type if document_type and document_type != DocumentType.UNKNOWN else None
    if declared_type:
        classification["document_type"] = declared_type.value
        classification["classification_method"] = "USER_DECLARED"
    doc_type_str = classification.get("document_type", "UNKNOWN")
    confidence = float(classification.get("confidence", 0.0))
    requires_review = classification.get("requires_human_review", True) and declared_type is None

    try:
        doc_type = DocumentType(doc_type_str)
    except ValueError:
        doc_type = DocumentType.UNKNOWN

    # Check if this is a multi-page container with classified logical groups
    logical_groups = evidence.get("logical_document_groups", {})
    has_logical_groups = evidence.get("page_count", 1) > 1 and any(
        len(pages) > 0 for l_type, pages in logical_groups.items() if l_type != "UNKNOWN"
    )

    # Guard: cannot extract UNKNOWN or low-confidence docs if single-doc and no logical groups (PRD §20, §7.1)
    if not has_logical_groups and (doc_type == DocumentType.UNKNOWN or requires_review):
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
    if doc_type != DocumentType.UNKNOWN:
        extractor = get_extractor(doc_type)
        fields = extractor.extract(document_id, evidence)
    else:
        # Default to first non-empty logical group extractor if doc_type is unknown
        first_type = next((DocumentType(k) for k, v in logical_groups.items() if len(v) > 0 and k != "UNKNOWN"), DocumentType.INVOICE)
        extractor = get_extractor(first_type)
        fields = extractor.extract(document_id, evidence)

    # Candidate generation + LightGBM on the whole document (additive; see extraction/ml/service.py)
    main_type = doc_type if doc_type != DocumentType.UNKNOWN else first_type
    ml_extraction = run_ml_extraction(document_id, main_type.value, evidence)
    fields = promote_ml_fields(fields, ml_extraction, document_id)
    fields = append_ml_only_fields(fields, ml_extraction, document_id)
    fields = merge_review_fields(document_id, evidence, main_type.value, fields, ml_extraction)
    resolve_field_pages(fields, evidence.get("pages", []))
    arithmetic_validation = validate_extraction(fields, evidence.get("pages", []))

    # Multi-document extraction across logical segments (ARCHITECTURE ROOT CAUSE A FIX)
    segments = evidence.get("segments", [])
    segment_extractions: list[dict] = []

    for seg in segments:
        seg_type_str = seg.get("document_type", "UNKNOWN")
        start_p = seg.get("start_page", 1)
        end_p = seg.get("end_page", 1)
        if seg_type_str != "UNKNOWN":
            try:
                s_type = DocumentType(seg_type_str)
                s_extractor = get_extractor(s_type)
                s_evidence = dict(evidence)
                # Filter evidence pages strictly to this segment's page range
                s_evidence["pages"] = [
                    p for p in evidence.get("pages", [])
                    if start_p <= p.get("page_number", 0) <= end_p
                ]
                s_fields = s_extractor.extract(document_id, s_evidence)
                s_ml = run_ml_extraction(document_id, seg_type_str, s_evidence)
                s_fields = promote_ml_fields(s_fields, s_ml, document_id)
                s_fields = append_ml_only_fields(s_fields, s_ml, document_id)
                s_fields = merge_review_fields(document_id, s_evidence, seg_type_str, s_fields, s_ml)
                resolve_field_pages(s_fields, s_evidence["pages"], default_page=start_p)
                s_arithmetic = validate_extraction(s_fields, s_evidence["pages"])
                segment_extractions.append({
                    "segment_number": seg.get("segment_number"),
                    "document_type": seg_type_str,
                    "start_page": start_p,
                    "end_page": end_p,
                    "page_count": seg.get("page_count", end_p - start_p + 1),
                    "classification_confidence": seg.get("classification_confidence", 1.0),
                    "extracted_fields": [f.model_dump() for f in s_fields],
                    "ml_extraction": s_ml,
                    "arithmetic_validation": s_arithmetic,
                })
            except Exception as e:
                logger.warning(
                    "segment_extraction_failed",
                    segment_number=seg.get("segment_number"),
                    doc_type=seg_type_str,
                    error=str(e),
                )

    # Legacy: Multi-document extraction across logical groups for backward compat
    logical_groups = evidence.get("logical_document_groups", {})
    logical_extractions: dict[str, list[dict]] = {}

    for l_type_str, p_list in logical_groups.items():
        if len(p_list) > 0 and l_type_str != "UNKNOWN":
            try:
                l_type = DocumentType(l_type_str)
                l_extractor = get_extractor(l_type)
                # Filter evidence pages for this document type
                l_evidence = dict(evidence)
                l_evidence["pages"] = [
                    p for p in evidence.get("pages", []) if p.get("page_number") in p_list
                ]
                l_fields = l_extractor.extract(document_id, l_evidence)
                resolve_field_pages(l_fields, l_evidence["pages"], default_page=min(p_list))
                logical_extractions[l_type_str] = [f.model_dump() for f in l_fields]
            except Exception as e:
                logger.warning("logical_extraction_failed", doc_type=l_type_str, error=str(e))

    unknown_fields = [f.field for f in fields if f.is_unknown]
    final_requires_review = len(unknown_fields) > 0 or any(getattr(field, "review_status", "UNKNOWN") == "REVIEW" for field in fields) or confidence < 0.95

    logger.info(
        "extract.pipeline.complete",
        document_id=document_id,
        doc_type=doc_type_str,
        page_count=evidence.get("page_count", 1),
        segment_count=len(segments),
        total_fields=len(fields),
        unknown_count=len(unknown_fields),
        requires_human_review=final_requires_review,
    )

    return {
        "document_id": document_id,
        "status": "SUCCESS",
        "page_count": evidence.get("page_count", 1),
        "classification": classification,
        "segments": segments,
        "segment_extractions": segment_extractions,
        "logical_document_groups": logical_groups,
        "pages": [
            {
                "page_number": p.get("page_number"),
                "classification": p.get("classification"),
                "signature_detection": p.get("signature_detection"),
                "line_count": len(p.get("lines", [])),
                "quality": p.get("quality"),
                "engine_used": p.get("engine_used", ""),
                "fallback_used": p.get("fallback_used", False),
                # Geometry of the canonical page image the normalized bboxes refer to.
                # Local filesystem path is intentionally not exposed.
                "page_image": {
                    k: (p.get("page_image") or {}).get(k)
                    for k in ("width", "height", "dpi", "rotation_applied")
                },
                "coord_space": p.get("coord_space", "page_image_normalized"),
            }
            for p in evidence.get("pages", [])
        ],
        # Keep the complete Phase 5 evidence available to persistence/audit
        # consumers. The compact pages projection above remains for clients
        # that only need page-level metadata.
        "evidence": {
            "pages": evidence.get("pages", []),
            "table_groups": evidence.get("table_groups", []),
            "processing_metadata": evidence.get("processing_metadata", {}),
        },
        "extracted_fields": [f.model_dump() for f in fields],
        "ml_extraction": ml_extraction,
        "warnings": [f"ml_extraction:{item.get('document_type')}:{(item.get('ml_extraction') or {}).get('status', 'unknown')}" for item in segment_extractions if (item.get("ml_extraction") or {}).get("status") not in ("ok", "disabled")],
        "arithmetic_validation": arithmetic_validation,
        "table_groups": evidence.get("table_groups", []),
        "logical_extractions": logical_extractions,
        "extraction_attempted": True,
        "unknown_fields": unknown_fields,
        "requires_human_review": final_requires_review,
        "pipeline_version": "phase6_v3_segment_aware",
    }



@router.post("/{document_type}", response_model=ExtractionResponse)
def extract_document(document_type: DocumentType, file: UploadFile) -> ExtractionResponse:
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

    purge_old_pages()
    document_id = str(uuid.uuid4())
    file_bytes = read_upload(file)
    logger.info(
        "extract.typed.request",
        document_id=document_id,
        document_type=document_type.value,
        filename=file.filename,
    )

    # Phase 5 pipeline
    pipeline = get_pipeline()
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
