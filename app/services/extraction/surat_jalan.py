"""
SuratJalanExtractor — PRD §14. Minimum fields:
surat_jalan_number, shipment_id, awb, shipment_date, sender, receiver,
origin, destination, item_description, quantity, weight.
"""

from app.core.logging import get_logger
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.base import BaseExtractor

logger = get_logger(__name__)

SURAT_JALAN_REQUIRED_FIELDS = [
    "surat_jalan_number",
    "shipment_id",
    "awb",
    "shipment_date",
    "sender",
    "receiver",
    "origin",
    "destination",
    "item_description",
    "quantity",
    "weight",
]


class SuratJalanExtractor(BaseExtractor):
    document_type = "SURAT_JALAN"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        logger.info("surat_jalan_extractor.stub_called", document_id=document_id)
        return [
            ExtractedField(
                field=name,
                value=None,
                confidence=0.0,
                source=EvidenceSource(document_id=document_id, page=1),
                method=ExtractionMethod.OCR,
                is_unknown=True,
            )
            for name in SURAT_JALAN_REQUIRED_FIELDS
        ]
