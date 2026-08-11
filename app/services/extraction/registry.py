"""
Registry yang memetakan DocumentType -> extractor spesifik.
Ini BUKAN fungsi extract_document() generik (PRD §23 melarang itu) —
registry hanya routing ke extractor yang tepat.
"""

from app.schemas.document_type import DocumentType
from app.services.extraction.base import BaseExtractor
from app.services.extraction.invoice import InvoiceExtractor
from app.services.extraction.pod import PODExtractor
from app.services.extraction.rate_agreement import RateAgreementExtractor
from app.services.extraction.surat_jalan import SuratJalanExtractor

_REGISTRY: dict[DocumentType, type[BaseExtractor]] = {
    DocumentType.INVOICE: InvoiceExtractor,
    DocumentType.POD: PODExtractor,
    DocumentType.SURAT_JALAN: SuratJalanExtractor,
    DocumentType.RATE_AGREEMENT: RateAgreementExtractor,
}


def get_extractor(document_type: DocumentType) -> BaseExtractor:
    extractor_cls = _REGISTRY.get(document_type)
    if extractor_cls is None:
        raise ValueError(
            f"No extractor registered for {document_type}. "
            "UNKNOWN document types must be routed to human review, "
            "not extraction (PRD §20)."
        )
    return extractor_cls()
