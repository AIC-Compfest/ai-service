import uuid

from fastapi import APIRouter, UploadFile

from app.core.logging import get_logger
from app.core.runtime import read_upload
from app.schemas.extraction import SignatureDetectionResponse
from app.services.signature.detector import SignatureDetector

router = APIRouter(prefix="/v1/signature", tags=["signature"])
logger = get_logger(__name__)


@router.post("/detect", response_model=SignatureDetectionResponse)
def detect_signature(file: UploadFile) -> SignatureDetectionResponse:
    document_id = str(uuid.uuid4())
    file_bytes = read_upload(file)
    logger.info("signature.request", document_id=document_id, filename=file.filename)
    detector = SignatureDetector()
    return detector.detect(document_id, file_bytes)
