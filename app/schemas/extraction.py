from pydantic import BaseModel

from app.schemas.document_type import DocumentType
from app.schemas.evidence import ExtractedField


class ExtractionResponse(BaseModel):
    document_id: str
    document_type: DocumentType
    fields: list[ExtractedField]
    ocr_engine_used: str
    fallback_triggered: bool = False
    status: str = "SUCCESS"  # SUCCESS | FAILED
    error_code: str | None = None
    requires_human_review: bool = False


class SignatureDetectionResponse(BaseModel):
    """PRD §33 — hanya presence detection, bukan verifikasi identitas."""

    document_id: str
    signature_present: bool
    confidence: float
    bbox: dict[str, float] | None = None
