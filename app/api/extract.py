import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.logging import get_logger
from app.schemas.document_type import DocumentType
from app.schemas.extraction import ExtractionResponse
from app.services.extraction.registry import get_extractor
from app.services.ocr.engine import OCREngine
from app.services.preprocessing.preprocessor import ImagePreprocessor

router = APIRouter(prefix="/v1/extract", tags=["extraction"])
logger = get_logger(__name__)


@router.post("/{document_type}", response_model=ExtractionResponse)
async def extract_document(document_type: DocumentType, file: UploadFile) -> ExtractionResponse:
    if document_type == DocumentType.UNKNOWN:
        raise HTTPException(
            status_code=422,
            detail="Cannot extract UNKNOWN document type; route to human review first (PRD §20).",
        )

    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    logger.info(
        "extract.request",
        document_id=document_id,
        document_type=document_type.value,
        filename=file.filename,
    )

    preprocessor = ImagePreprocessor()
    variants = preprocessor.generate_variants(file_bytes)

    ocr_engine = OCREngine()
    ocr_result = ocr_engine.run(variants["original"])

    extractor = get_extractor(document_type)
    fields = extractor.extract(document_id, ocr_result)

    return ExtractionResponse(
        document_id=document_id,
        document_type=document_type,
        fields=fields,
        ocr_engine_used=ocr_result.get("engine_used", "pymupdf_vector"),
        fallback_triggered=ocr_result.get("fallback_triggered", False),
        status="SUCCESS",
        requires_human_review=True,
    )


@router.post("/understand/evidence")
async def understand_document_evidence(file: UploadFile) -> dict:
    """Phase 5 Document Understanding Endpoint: Returns full structured evidence with bboxes and regions."""
    from app.services.pipeline import Phase5DocumentPipeline
    document_id = str(uuid.uuid4())
    file_bytes = await file.read()
    pipeline = Phase5DocumentPipeline()
    return pipeline.process_document(file_bytes, filename=file.filename or "doc.pdf", document_id=document_id)

