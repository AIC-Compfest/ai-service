import asyncio
import uuid
from functools import wraps
from typing import Callable, TypeVar

from fastapi import APIRouter, UploadFile, HTTPException

from app.core.logging import get_logger
from app.core.runtime import read_upload
from app.schemas.extraction import SignatureDetectionResponse
from app.services.signature.detector import SignatureDetector

router = APIRouter(prefix="/v1/signature", tags=["signature"])
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


@router.post("/detect", response_model=SignatureDetectionResponse)
@timeout_wrapper(timeout_seconds=25)
async def detect_signature(file: UploadFile) -> SignatureDetectionResponse:
    try:
        document_id = str(uuid.uuid4())
        file_bytes = read_upload(file)
        logger.info("signature.request", document_id=document_id, filename=file.filename)
        detector = SignatureDetector()
        # Run detector in thread pool to prevent blocking
        return await asyncio.to_thread(detector.detect, document_id, file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("signature.error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Signature detection failed: {str(e)}")
