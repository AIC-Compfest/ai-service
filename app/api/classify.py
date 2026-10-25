import uuid

from fastapi import APIRouter, UploadFile

from app.core.logging import get_logger
from app.core.runtime import read_upload
from app.schemas.classify import ClassifyResponse
from app.services.classification.classifier import get_classifier

router = APIRouter(prefix="/v1/classify", tags=["classification"])
logger = get_logger(__name__)


@router.post("", response_model=ClassifyResponse)
def classify_document(file: UploadFile) -> ClassifyResponse:
    document_id = str(uuid.uuid4())
    file_bytes = read_upload(file)
    logger.info("classify.request", document_id=document_id, filename=file.filename)
    classifier = get_classifier()
    return classifier.classify(document_id, file_bytes)
