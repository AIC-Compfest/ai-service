"""
Document classifier — PRD §20.

STUB: belum ada model terlatih. Untuk sekarang selalu mengembalikan UNKNOWN
dengan confidence 0.0 sehingga secara eksplisit memaksa human review,
bukan menebak (PRD §60 — no hallucination rule berlaku juga untuk stub).
"""

from app.core.logging import get_logger
from app.schemas.classify import ClassifyResponse
from app.schemas.document_type import DocumentType

logger = get_logger(__name__)


class DocumentClassifier:
    def classify(self, document_id: str, file_bytes: bytes) -> ClassifyResponse:
        logger.info("classify.stub_called", document_id=document_id)
        return ClassifyResponse(
            document_id=document_id,
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            requires_human_review=True,
        )


def get_classifier() -> DocumentClassifier:
    return DocumentClassifier()
