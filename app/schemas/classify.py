from pydantic import BaseModel

from app.schemas.document_type import DocumentType


class ClassifyResponse(BaseModel):
    document_id: str
    document_type: DocumentType
    confidence: float
    requires_human_review: bool
    # PRD §20: classification_confidence < threshold -> human review.
    # Threshold pemilik keputusan tetap backend-api; service ini hanya
    # melaporkan confidence apa adanya.
