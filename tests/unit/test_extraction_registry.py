import unittest

from app.schemas.document_type import DocumentType
from app.services.extraction.invoice import InvoiceExtractor
from app.services.extraction.pod import PODExtractor
from app.services.extraction.rate_agreement import RateAgreementExtractor
from app.services.extraction.registry import get_extractor
from app.services.extraction.surat_jalan import SuratJalanExtractor


class TestExtractionRegistry(unittest.TestCase):
    def test_registry_routes_to_correct_extractor(self):
        cases = [
            (DocumentType.INVOICE, InvoiceExtractor),
            (DocumentType.POD, PODExtractor),
            (DocumentType.SURAT_JALAN, SuratJalanExtractor),
            (DocumentType.RATE_AGREEMENT, RateAgreementExtractor),
        ]
        for doc_type, expected_cls in cases:
            with self.subTest(doc_type=doc_type):
                extractor = get_extractor(doc_type)
                self.assertIsInstance(extractor, expected_cls)

    def test_registry_raises_for_unknown(self):
        with self.assertRaises(ValueError):
            get_extractor(DocumentType.UNKNOWN)

    def test_invoice_extractor_returns_unknown_fields_when_no_ocr_data(self):
        extractor = InvoiceExtractor()
        fields = extractor.extract("doc_001", ocr_result={})
        self.assertGreater(len(fields), 0)
        self.assertTrue(all(f.is_unknown for f in fields))
        # No hallucination rule (PRD §60): value harus None jika tidak ada evidence
        self.assertTrue(all(f.value is None for f in fields))
