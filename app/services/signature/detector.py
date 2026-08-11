"""
Signature presence detector — PRD §33.

Hanya menjawab: "Apakah terdapat tanda tangan pada area dokumen yang
relevan?" TIDAK boleh mengklaim signature authenticity, identity, atau
ownership (PRD §33, §70 poin 3-4).

Baseline model direkomendasikan custom detector (bukan RT-DETR/YOLO
langsung, perhatikan licensing — PRD §40, §70 poin 15).
"""

from app.core.logging import get_logger
from app.schemas.extraction import SignatureDetectionResponse

logger = get_logger(__name__)


class SignatureDetector:
    def detect(self, document_id: str, image_bytes: bytes) -> SignatureDetectionResponse:
        logger.info("signature_detector.stub_called", document_id=document_id)
        return SignatureDetectionResponse(
            document_id=document_id,
            signature_present=False,
            confidence=0.0,
            bbox=None,
        )
