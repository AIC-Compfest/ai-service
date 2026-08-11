"""
InvoiceExtractor — PRD §13. Minimum fields:
invoice_number, invoice_date, vendor, customer, billing_period, currency,
subtotal, tax, grand_total, line_items[] (shipment_id, awb, service_type,
origin, destination, weight, quantity, base_rate, fuel_surcharge,
accessorial_charge, line_total).

STUB: belum terhubung ke OCR sungguhan. Lihat ROADMAP.md Fase 4-5.
"""

from app.core.logging import get_logger
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.base import BaseExtractor

logger = get_logger(__name__)

INVOICE_REQUIRED_FIELDS = [
    "invoice_number",
    "invoice_date",
    "vendor",
    "customer",
    "billing_period",
    "currency",
    "subtotal",
    "tax",
    "grand_total",
]


class InvoiceExtractor(BaseExtractor):
    document_type = "INVOICE"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        logger.info("invoice_extractor.stub_called", document_id=document_id)
        return [
            ExtractedField(
                field=name,
                value=None,
                confidence=0.0,
                source=EvidenceSource(document_id=document_id, page=1),
                method=ExtractionMethod.OCR,
                is_unknown=True,
            )
            for name in INVOICE_REQUIRED_FIELDS
        ]
