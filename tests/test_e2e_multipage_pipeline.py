"""
test_e2e_multipage_pipeline.py

End-to-End Multi-Page PDF Pipeline Verification Tests.
Validates:
1. Ingestion of multi-page container PDF (3+ pages)
2. Per-page rendering, quality evaluation, OCR & bounding box localization
3. Per-page ML Classification (Invoice, Surat Jalan, POD)
4. Per-page Signature & Stamp Detection
5. Logical Document Grouping across pages
6. Multi-document structured extraction with Evidence Source
"""

import unittest
from pathlib import Path

from app.services.pipeline import Phase5DocumentPipeline
from app.api.extract import get_extractor
from app.schemas.document_type import DocumentType


class TestE2EMultiPagePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = Phase5DocumentPipeline()
        cls.sample_dir = Path("d:/AIC/sample_datasets")
        cls.bundle_clean = cls.sample_dir / "01_CLEAN_MATCH_Cepat_Logistik_Nusantara.pdf"
        cls.bundle_overcharge = cls.sample_dir / "02_RATE_OVERCHARGE_Trans_Express_Indonesia.pdf"
        cls.bundle_exception = cls.sample_dir / "03_WEIGHT_FUEL_EXCEPTION_Kargo_Andalan.pdf"

    def test_bundle_1_clean_match_multipage_understanding(self):
        """Test 3-page container PDF: Page 1=Invoice, Page 2=Surat Jalan, Page 3=POD."""
        if not self.bundle_clean.exists():
            self.skipTest("Sample PDF bundle not generated")

        evidence = self.pipeline.process_document(
            self.bundle_clean, filename=self.bundle_clean.name, document_id="doc-e2e-001"
        )

        self.assertEqual(evidence["document_id"], "doc-e2e-001")
        self.assertEqual(evidence["page_count"], 3)
        self.assertIn("pages", evidence)
        self.assertEqual(len(evidence["pages"]), 3)

        pages = evidence["pages"]

        # Page 1: Invoice
        p1 = pages[0]
        self.assertEqual(p1["page_number"], 1)
        self.assertEqual(p1["classification"]["document_type"], "INVOICE")
        self.assertGreater(p1["classification"]["confidence"], 0.70)
        self.assertIn("lines", p1)
        self.assertGreater(len(p1["lines"]), 0)

        # Page 2: Surat Jalan
        p2 = pages[1]
        self.assertEqual(p2["page_number"], 2)
        self.assertEqual(p2["classification"]["document_type"], "SURAT_JALAN")
        self.assertGreater(p2["classification"]["confidence"], 0.70)

        # Page 3: POD
        p3 = pages[2]
        self.assertEqual(p3["page_number"], 3)
        self.assertEqual(p3["classification"]["document_type"], "POD")
        self.assertGreater(p3["classification"]["confidence"], 0.70)

        # Signature Detection on POD page
        self.assertTrue(p3["signature_detection"]["signature_present"])

        # Logical Document Groups
        groups = evidence["logical_document_groups"]
        self.assertIn(1, groups.get("INVOICE", []))
        self.assertIn(2, groups.get("SURAT_JALAN", []))
        self.assertIn(3, groups.get("POD", []))

    def test_bundle_1_extraction_all_logical_types(self):
        """Test extraction across all logical documents in the container."""
        if not self.bundle_clean.exists():
            self.skipTest("Sample PDF bundle not generated")

        evidence = self.pipeline.process_document(
            self.bundle_clean, filename=self.bundle_clean.name, document_id="doc-e2e-001"
        )

        # 1. Extract Invoice
        inv_extractor = get_extractor(DocumentType.INVOICE)
        inv_evidence = dict(evidence)
        inv_evidence["pages"] = [pages for pages in evidence["pages"] if pages["page_number"] == 1]
        inv_fields = inv_extractor.extract("doc-e2e-001", inv_evidence)
        inv_map = {f.field: f.value for f in inv_fields}

        self.assertIn("invoice_number", inv_map)
        self.assertIn("grand_total", inv_map)
        self.assertEqual(inv_map["vendor_id"], "3PL-LOG-001")

        # 2. Extract Surat Jalan
        sj_extractor = get_extractor(DocumentType.SURAT_JALAN)
        sj_evidence = dict(evidence)
        sj_evidence["pages"] = [pages for pages in evidence["pages"] if pages["page_number"] == 2]
        sj_fields = sj_extractor.extract("doc-e2e-001", sj_evidence)
        sj_map = {f.field: f.value for f in sj_fields}

        self.assertIn("surat_jalan_number", sj_map)
        self.assertIn("actual_weight_kg", sj_map)

        # 3. Extract POD
        pod_extractor = get_extractor(DocumentType.POD)
        pod_evidence = dict(evidence)
        pod_evidence["pages"] = [pages for pages in evidence["pages"] if pages["page_number"] == 3]
        pod_fields = pod_extractor.extract("doc-e2e-001", pod_evidence)
        pod_map = {f.field: f.value for f in pod_fields}

        self.assertIn("pod_number", pod_map)
        self.assertIn("signature_present", pod_map)

    def test_bundle_1_segments_verification(self):
        """Test that Phase5DocumentPipeline generates segments for multi-page PDF."""
        if not self.bundle_clean.exists():
            self.skipTest("Sample PDF bundle not generated")

        evidence = self.pipeline.process_document(
            self.bundle_clean, filename=self.bundle_clean.name, document_id="doc-e2e-seg-001"
        )

        self.assertIn("segments", evidence)
        segments = evidence["segments"]
        self.assertGreaterEqual(len(segments), 3)

        seg_types = [s["document_type"] for s in segments]
        self.assertIn("INVOICE", seg_types)
        self.assertIn("SURAT_JALAN", seg_types)
        self.assertIn("POD", seg_types)

        for s in segments:
            self.assertGreaterEqual(s["segment_number"], 1)
            self.assertGreaterEqual(s["start_page"], 1)
            self.assertGreaterEqual(s["end_page"], s["start_page"])
            self.assertGreaterEqual(s["page_count"], 1)
            self.assertGreaterEqual(s["classification_confidence"], 0.0)

    def test_10page_synthetic_fixture_segmentation(self):
        """Test segmentation on the 10-page synthetic test fixture."""
        fixture_path = Path("d:/AIC/synthetic-data/test_fixtures/synthetic_3pl_test_10pages.pdf")
        if not fixture_path.exists():
            self.skipTest("10-page fixture PDF not found")

        evidence = self.pipeline.process_document(
            fixture_path, filename=fixture_path.name, document_id="doc-10p-001"
        )

        self.assertEqual(evidence["page_count"], 10)
        self.assertIn("segments", evidence)
        segments = evidence["segments"]
        self.assertGreaterEqual(len(segments), 3)

        # Check total pages covered by segments equals page count
        total_seg_pages = sum(s["page_count"] for s in segments)
        self.assertEqual(total_seg_pages, 10)


if __name__ == "__main__":
    unittest.main()

