"""
PODExtractor — PRD §15. Minimum fields:
pod_number, shipment_id, awb, delivery_date, receiver_name, receiver_company,
quantity_received, condition, signature_present, signature_bbox,
signature_confidence, stamp_present.

signature_present di sini HANYA presence — bukan signature_authentic atau
signature_owner (PRD §15, §33).
"""

from app.core.logging import get_logger
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.base import BaseExtractor

logger = get_logger(__name__)

POD_REQUIRED_FIELDS = [
    "pod_number",
    "shipment_id",
    "awb",
    "delivery_date",
    "receiver_name",
    "receiver_company",
    "quantity_received",
    "condition",
    "stamp_present",
]


class PODExtractor(BaseExtractor):
    document_type = "POD"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        logger.info("pod_extractor.stub_called", document_id=document_id)
        # signature_present/signature_bbox/signature_confidence ditangani
        # terpisah oleh services/signature (lihat detector.py), bukan di sini.
        return [
            ExtractedField(
                field=name,
                value=None,
                confidence=0.0,
                source=EvidenceSource(document_id=document_id, page=1),
                method=ExtractionMethod.OCR,
                is_unknown=True,
            )
            for name in POD_REQUIRED_FIELDS
        ]
