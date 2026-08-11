"""
RateAgreementExtractor — PRD §16. Minimum fields:
agreement_id, vendor_id, effective_from, effective_to, rate_rules[],
surcharge_rules[], minimum_charge, dispute_window_days.

PENTING: hasil ekstraksi di sini masih RAW. Konversi menjadi executable
pricing rules adalah tanggung jawab Contract Rule Compiler di backend-api
(PRD §28), BUKAN service ini. Jangan simpan kontrak hanya sebagai raw text
atau embedding (PRD §16).
"""

from app.core.logging import get_logger
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.base import BaseExtractor

logger = get_logger(__name__)

RATE_AGREEMENT_REQUIRED_FIELDS = [
    "agreement_id",
    "vendor_id",
    "effective_from",
    "effective_to",
    "minimum_charge",
    "dispute_window_days",
    # rate_rules[] dan surcharge_rules[] ditangani sebagai structured
    # sub-extraction terpisah setelah OCR table extraction (Fase 5+)
]


class RateAgreementExtractor(BaseExtractor):
    document_type = "RATE_AGREEMENT"

    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        logger.info("rate_agreement_extractor.stub_called", document_id=document_id)
        return [
            ExtractedField(
                field=name,
                value=None,
                confidence=0.0,
                source=EvidenceSource(document_id=document_id, page=1),
                method=ExtractionMethod.OCR,
                is_unknown=True,
            )
            for name in RATE_AGREEMENT_REQUIRED_FIELDS
        ]
