import asyncio
import uuid
from functools import wraps
from typing import Callable, TypeVar

from fastapi import APIRouter, UploadFile, HTTPException

from app.core.logging import get_logger
from app.core.runtime import read_upload
from app.schemas.classify import ClassifyResponse
from app.services.classification.classifier import get_classifier

router = APIRouter(prefix="/v1/classify", tags=["classification"])
logger = get_logger(__name__)

F = TypeVar("F", bound=Callable)

def timeout_wrapper(timeout_seconds: int = 25):
    """Wrap async function with timeout. Must complete before Appwrite's 30s hard limit."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} exceeded {timeout_seconds}s timeout")
                raise HTTPException(status_code=408, detail=f"Request timeout (>{timeout_seconds}s)")
        return wrapper
    return decorator


@router.post("", response_model=ClassifyResponse)
@timeout_wrapper(timeout_seconds=25)
async def classify_document(file: UploadFile) -> ClassifyResponse:
    try:
        document_id = str(uuid.uuid4())
        file_bytes = read_upload(file)
        logger.info("classify.request", document_id=document_id, filename=file.filename)
        classifier = get_classifier()
        # Run classifier in thread pool to prevent blocking
        return await asyncio.to_thread(classifier.classify, document_id, file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("classify.error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
