from pydantic import BaseModel
from typing import Dict, Optional

from app.schemas.document_type import DocumentType


class ClassifyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    document_id: str
    document_type: DocumentType
    confidence: float
    requires_human_review: bool
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = "doc-classifier-v1"
    classification_method: Optional[str] = "ml_tfidf_logistic_regression"

