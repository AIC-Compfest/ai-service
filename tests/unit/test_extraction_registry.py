import pytest

from app.schemas.document_type import DocumentType
from app.services.extraction.invoice import InvoiceExtractor
from app.services.extraction.pod import PODExtractor
from app.services.extraction.rate_agreement import RateAgreementExtractor
from app.services.extraction.registry import get_extractor
from app.services.extraction.surat_jalan import SuratJalanExtractor


@pytest.mark.parametrize(
    "doc_type,expected_cls",
    [
        (DocumentType.INVOICE, InvoiceExtractor),
        (DocumentType.POD, PODExtractor),
        (DocumentType.SURAT_JALAN, SuratJalanExtractor),
        (DocumentType.RATE_AGREEMENT, RateAgreementExtractor),
    ],
)
def test_registry_routes_to_correct_extractor(doc_type, expected_cls):
    extractor = get_extractor(doc_type)
    assert isinstance(extractor, expected_cls)


def test_registry_raises_for_unknown():
    with pytest.raises(ValueError):
        get_extractor(DocumentType.UNKNOWN)


def test_invoice_extractor_returns_unknown_fields_when_no_ocr_data():
    extractor = InvoiceExtractor()
    fields = extractor.extract("doc_001", ocr_result={})
    assert len(fields) > 0
    assert all(f.is_unknown for f in fields)
    # No hallucination rule (PRD §60): value harus None jika tidak ada evidence
    assert all(f.value is None for f in fields)
