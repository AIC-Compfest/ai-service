from pydantic import BaseModel

from app.schemas.document_type import DocumentType
from app.schemas.evidence import ExtractedField, BoundingBox


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
    """
    PRD §33, §70 — Visual Presence Detection for POD/Surat Jalan.
    Strictly answers: 'Apakah terdapat tanda tangan dan/atau stempel fisik pada area dokumen?'
    Never asserts identity, authenticity, ownership, or biometrics.
    """

    document_id: str
    signature_present: bool
    confidence: float
    bbox: BoundingBox | dict[str, float] | None = None
    stamp_present: bool = False
    stamp_confidence: float = 0.0
    stamp_bbox: BoundingBox | dict[str, float] | None = None
    requires_human_review: bool = False
    is_unknown: bool = False

