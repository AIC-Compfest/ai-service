"""
test_phase5_components.py

Comprehensive Unit Tests for Phase 5 Document Understanding Components.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

from app.services.preprocessing.document_ingestion import DocumentIngestionService
from app.services.preprocessing.image_preprocessor import ImagePreprocessor
from app.services.classification.classifier import DocumentClassifier
from app.services.ocr.engine import OCREngine
from app.services.signature.detector import SignatureDetector
from app.services.pipeline import Phase5DocumentPipeline
from app.schemas.document_type import DocumentType
from tests.support.paths import SYNTHETIC_DATA_ROOT, require


class TestPhase5Components(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate synthetic dataset sample files (external to this repository; skip when absent)
        cls.dataset_dir = SYNTHETIC_DATA_ROOT / "dataset" / "v4_pilot_20"
        if not cls.dataset_dir.exists():
            cls.dataset_dir = SYNTHETIC_DATA_ROOT / "dataset" / "v3_pilot_20"
        require(cls.dataset_dir, "synthetic-data/dataset/v4_pilot_20")

        def first(pattern: str) -> Path:
            matches = sorted(cls.dataset_dir.glob(pattern))
            if not matches:
                raise unittest.SkipTest(f"No fixture matching {pattern} in {cls.dataset_dir.name}")
            return matches[0]

        cls.sample_invoice = first("invoices/*.pdf")
        cls.sample_sj = first("transactions/*/surat_jalan.pdf")
        cls.sample_pod = first("transactions/*/pod.jpg")
        cls.sample_ra = first("transactions/*/rate_agreement.pdf")

    def test_document_ingestion_pdf(self):
        ingested = DocumentIngestionService.ingest_file(self.sample_invoice)
        self.assertIsNotNone(ingested.document_id)
        self.assertEqual(len(ingested.source_sha256), 64)
        self.assertEqual(ingested.mime_type, "application/pdf")
        self.assertGreaterEqual(len(ingested.pages), 1)
        self.assertTrue(ingested.pages[0].is_pdf_page)
        self.assertGreater(ingested.pages[0].width, 0)
        self.assertGreater(ingested.pages[0].height, 0)

    def test_document_ingestion_jpg(self):
        ingested = DocumentIngestionService.ingest_file(self.sample_pod)
        # The source is a JPEG, but ingestion deliberately re-encodes every non-PDF image to the
        # canonical PNG (document_ingestion.py), so the evidence reports the canonical type.
        self.assertEqual(self.sample_pod.suffix.lower(), ".jpg")
        self.assertEqual(ingested.mime_type, "image/png")
        self.assertEqual(len(ingested.pages), 1)
        self.assertFalse(ingested.pages[0].is_pdf_page)

    def test_image_preprocessor_quality_metrics(self):
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        # Draw some dark text/rectangles
        cv2.rectangle(img, (50, 50), (200, 100), (20, 20, 20), -1)
        quality = ImagePreprocessor.estimate_quality(img)
        self.assertIn("blur_score", quality)
        self.assertIn("brightness", quality)
        self.assertIn("contrast", quality)
        self.assertIn("skew_angle", quality)
        self.assertFalse(quality["is_low_contrast"])

    def test_classifier_accuracy(self):
        classifier = DocumentClassifier()

        # 1. Invoice Test
        inv_ingested = DocumentIngestionService.ingest_file(self.sample_invoice)
        inv_res = classifier.classify(inv_ingested.document_id, inv_ingested.raw_bytes)
        self.assertEqual(inv_res.document_type, DocumentType.INVOICE)
        self.assertGreaterEqual(inv_res.confidence, 0.70)

        # 2. Surat Jalan Test
        sj_ingested = DocumentIngestionService.ingest_file(self.sample_sj)
        sj_res = classifier.classify(sj_ingested.document_id, sj_ingested.raw_bytes)
        self.assertEqual(sj_res.document_type, DocumentType.SURAT_JALAN)
        # Synthetic SJ is out-of-distribution for the model trained on DATASET/Dataset.
        self.assertGreaterEqual(sj_res.confidence, 0.60)

        # 3. Rate Agreement Test
        ra_ingested = DocumentIngestionService.ingest_file(self.sample_ra)
        ra_res = classifier.classify(ra_ingested.document_id, ra_ingested.raw_bytes)
        self.assertEqual(ra_res.document_type, DocumentType.RATE_AGREEMENT)
        self.assertGreaterEqual(ra_res.confidence, 0.70)

        # 4. Unknown Document Test
        unk_res = classifier.classify_text("Random document content with no logistics terms whatsoever.")
        self.assertEqual(unk_res.document_type, DocumentType.UNKNOWN)
        self.assertTrue(unk_res.requires_human_review)

    def test_ocr_and_layout_understanding(self):
        pipeline = Phase5DocumentPipeline()
        evidence = pipeline.process_document(self.sample_invoice)

        self.assertEqual(evidence["classification"]["document_type"], "INVOICE")
        self.assertGreater(evidence["page_count"], 0)
        p1 = evidence["pages"][0]
        self.assertGreater(len(p1["lines"]), 5)
        self.assertGreater(len(p1["regions"]), 1)
        self.assertEqual(p1["engine_used"], "pymupdf_vector")
        self.assertFalse(p1["fallback_used"])

        # Check that bounding boxes are normalized to [0, 1000]
        for line in p1["lines"]:
            bx1, by1, bx2, by2 = line["bbox_norm"]
            self.assertTrue(0 <= bx1 <= 1000)
            self.assertTrue(0 <= by1 <= 1000)
            self.assertTrue(0 <= bx2 <= 1000)
            self.assertTrue(0 <= by2 <= 1000)

    def test_signature_presence_detection(self):
        detector = SignatureDetector()
        pod_bytes = self.sample_pod.read_bytes()
        sig_res = detector.detect("DOC-POD-001", pod_bytes)
        self.assertTrue(sig_res.signature_present)
        self.assertGreater(sig_res.confidence, 0.70)

    def test_pipeline_end_to_end(self):
        pipeline = Phase5DocumentPipeline()
        evidence = pipeline.process_document(self.sample_sj)
        self.assertEqual(evidence["classification"]["document_type"], "SURAT_JALAN")
        self.assertIn("processing_time_ms", evidence["processing_metadata"])
        self.assertGreater(evidence["processing_metadata"]["processing_time_ms"], 0)


if __name__ == "__main__":
    unittest.main()
