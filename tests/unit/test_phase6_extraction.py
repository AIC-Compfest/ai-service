"""
test_phase6_extraction.py

Comprehensive Unit Tests for Phase 6: Structured Field Extraction.
Tests all four extractors against:
  1. Synthetic OCR evidence dicts (unit tests — no real files needed).
  2. Real Phase 5 pipeline output on dataset v4_pilot_20 (integration tests).

PRD References: §13, §14, §15, §16, §18, §23, §49, §60.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.schemas.document_type import DocumentType
from app.schemas.evidence import ExtractionMethod
from app.services.extraction.invoice import InvoiceExtractor
from app.services.extraction.pod import PODExtractor
from app.services.extraction.rate_agreement import RateAgreementExtractor
from app.services.extraction.surat_jalan import SuratJalanExtractor
from app.services.extraction.utils import (
    find_key_value_horizontal,
    find_key_value_vertical,
    find_amount_near_keyword,
    extract_regex_with_bbox,
    parse_date_string,
    parse_amount,
    parse_weight,
    parse_table_columns,
)


# ---------------------------------------------------------------------------
# Synthetic OCR Evidence Fixtures
# ---------------------------------------------------------------------------

def _make_line(text: str, bbox_norm: list[float], confidence: float = 0.99) -> dict:
    """Build a minimal OCR line dict identical to Phase 5 output format."""
    words = text.split()
    tokens = []
    if len(bbox_norm) == 4:
        x1, y1, x2, y2 = bbox_norm
        step = (x2 - x1) / max(len(words), 1)
        for i, w in enumerate(words):
            tokens.append({
                "text": w,
                "confidence": confidence,
                "bbox_norm": [x1 + i * step, y1, x1 + (i + 1) * step, y2],
            })
    return {
        "text": text,
        "confidence": confidence,
        "bbox_norm": bbox_norm,
        "tokens": tokens,
    }


SYNTHETIC_INVOICE_EVIDENCE = {
    "document_id": "TEST-INV-001",
    "classification": {"document_type": "INVOICE", "confidence": 0.95},
    "signature_detection": {"signature_present": False, "confidence": 0.88, "bbox": None},
    "pages": [{
        "page_number": 1,
        "width": 595, "height": 841,
        "engine_used": "pymupdf_vector",
        "average_confidence": 0.99,
        "lines": [
            _make_line("PT Cepat Logistik Nusantara [VENDOR ID: 3PL-LOG-001]", [50, 40, 940, 60]),
            _make_line("TAX INVOICE No: INV-2026-00089", [50, 80, 600, 100]),
            _make_line("Invoice Date: 10 Agustus 2026", [50, 110, 400, 130]),
            _make_line("Bill To: PT Retailindo Sukses Makmur", [50, 140, 500, 160]),
            _make_line("Billing Period: 01 Juli 2026 - 31 Juli 2026", [50, 165, 600, 185]),
            _make_line("Due Date: 10 September 2026", [50, 190, 400, 210]),
            _make_line("Currency: IDR", [50, 220, 250, 240]),
            _make_line("Subtotal IDR 1.250.000", [400, 720, 940, 740]),
            _make_line("PPN 11% IDR 137.500", [400, 750, 940, 770]),
            _make_line("Grand Total IDR 1.387.500", [400, 780, 940, 800]),
        ],
        "tables": [{
            "bbox_norm": [50, 350, 940, 680],
            "headers": ["No", "Shipment ID", "AWB", "Origin", "Destination",
                        "Chargeable Weight", "Base Rate", "Fuel Surcharge", "Total"],
            "rows": [
                ["1", "SHP-100001", "AWB77829391", "Jakarta", "Bandung", "15.0",
                 "75000", "10500", "85500"],
                ["2", "SHP-100002", "AWB77829392", "Jakarta", "Surabaya", "20.0",
                 "120000", "16800", "136800"],
            ],
            "confidence": 0.95,
        }],
        "regions": [
            {"region_type": "header", "bbox_norm": [50, 40, 940, 190], "confidence": 0.95},
            {"region_type": "table", "bbox_norm": [50, 350, 940, 680], "confidence": 0.92},
            {"region_type": "totals", "bbox_norm": [400, 720, 940, 810], "confidence": 0.95},
        ],
        "quality": {"blur_score": 1500.0, "brightness": 240.0, "contrast": 45.0,
                    "skew_angle": 0.0, "is_blurry": False, "is_low_contrast": False},
    }],
}

SYNTHETIC_SJ_EVIDENCE = {
    "document_id": "TEST-SJ-001",
    "classification": {"document_type": "SURAT_JALAN", "confidence": 0.98},
    "signature_detection": {"signature_present": True, "confidence": 0.90, "bbox": None},
    "pages": [{
        "page_number": 1,
        "width": 595, "height": 841,
        "engine_used": "pymupdf_vector",
        "average_confidence": 0.99,
        "lines": [
            _make_line("SURAT JALAN", [50, 40, 300, 60]),
            _make_line("No. Surat Jalan: SJ-2026-00089", [50, 80, 500, 100]),
            _make_line("Shipment ID: SHP-100001", [50, 110, 400, 130]),
            _make_line("AWB No: AWB77829391", [50, 140, 350, 160]),
            _make_line("Tanggal Kirim: 10 Agustus 2026", [50, 170, 500, 190]),
            _make_line("Pengirim: PT Retailindo Sukses Makmur", [50, 210, 600, 230]),
            _make_line("Penerima: Toko Maju Jaya", [50, 250, 500, 270]),
            _make_line("Asal: Jakarta Barat", [50, 290, 400, 310]),
            _make_line("Tujuan: Bandung Selatan", [50, 320, 450, 340]),
            _make_line("Koli: 10", [50, 360, 250, 380]),
            _make_line("Berat Aktual: 15 KG", [50, 390, 350, 410]),
            _make_line("Deskripsi Barang: Elektronik Konsumen", [50, 430, 600, 450]),
        ],
        "tables": [],
        "regions": [],
        "quality": {"blur_score": 1500.0, "brightness": 240.0, "contrast": 45.0,
                    "skew_angle": 0.0, "is_blurry": False, "is_low_contrast": False},
    }],
}

SYNTHETIC_POD_EVIDENCE = {
    "document_id": "TEST-POD-001",
    "classification": {"document_type": "POD", "confidence": 0.92},
    "signature_detection": {"signature_present": True, "confidence": 0.94,
                             "bbox": {"x1": 650.0, "y1": 750.0, "x2": 900.0, "y2": 850.0}},
    "pages": [{
        "page_number": 1,
        "width": 595, "height": 841,
        "engine_used": "raster_ocr",
        "average_confidence": 0.87,
        "lines": [
            _make_line("PROOF OF DELIVERY", [50, 40, 400, 60]),
            _make_line("POD No: POD-2026-00089", [50, 80, 450, 100]),
            _make_line("AWB No: AWB77829391", [50, 110, 400, 130]),
            _make_line("Received Date: 12 Agustus 2026", [50, 150, 500, 170]),
            _make_line("Arrival Time: 14:30 WIB", [50, 180, 400, 200]),
            _make_line("Diterima Oleh: Budi Santoso", [50, 220, 500, 240]),
            _make_line("Perusahaan Penerima: Toko Maju Jaya", [50, 250, 600, 270]),
            _make_line("Delivered Qty: 10", [50, 290, 350, 310]),
            _make_line("Kondisi Barang: GOOD", [50, 330, 400, 350]),
            _make_line("STEMPEL", [600, 720, 850, 760], confidence=0.80),
        ],
        "tables": [],
        "regions": [],
        "quality": {"blur_score": 800.0, "brightness": 200.0, "contrast": 50.0,
                    "skew_angle": 2.5, "is_blurry": False, "is_low_contrast": False},
    }],
}

SYNTHETIC_RA_EVIDENCE = {
    "document_id": "TEST-RA-001",
    "classification": {"document_type": "RATE_AGREEMENT", "confidence": 0.96},
    "signature_detection": {"signature_present": False, "confidence": 0.88, "bbox": None},
    "pages": [{
        "page_number": 1,
        "width": 595, "height": 841,
        "engine_used": "pymupdf_vector",
        "average_confidence": 0.99,
        "lines": [
            _make_line("PT Cepat Logistik Nusantara [VENDOR ID: 3PL-LOG-001]", [50, 40, 940, 60]),
            _make_line("PERJANJIAN KERJASAMA TARIF LOGISTIK", [50, 80, 800, 100]),
            _make_line("Kontrak No: AGR-2026-001", [50, 110, 500, 130]),
            _make_line("Berlaku Mulai: 01 Januari 2026", [50, 150, 500, 170]),
            _make_line("Berlaku Sampai: 31 Desember 2026", [50, 180, 550, 200]),
            _make_line("Biaya Minimum: Rp 25.000", [50, 220, 500, 240]),
            _make_line("Batas Waktu Klaim: 45 hari", [50, 250, 450, 270]),
            _make_line("Fuel Surcharge Index: 14%", [50, 290, 500, 310]),
        ],
        "tables": [{
            "bbox_norm": [50, 350, 940, 650],
            "headers": ["Service", "Origin", "Destination", "Min Weight",
                        "Max Weight", "Base Rate"],
            "rows": [
                ["REGULAR", "Jakarta", "Bandung", "0", "10", "60000"],
                ["REGULAR", "Jakarta", "Bandung", "10", "20", "75000"],
                ["REGULAR", "Jakarta", "Bandung", "20", "50", "100000"],
                ["REGULAR", "Jakarta", "Surabaya", "0", "10", "90000"],
            ],
            "confidence": 0.96,
        }],
        "regions": [],
        "quality": {"blur_score": 1500.0, "brightness": 240.0, "contrast": 45.0,
                    "skew_angle": 0.0, "is_blurry": False, "is_low_contrast": False},
    }],
}


# ---------------------------------------------------------------------------
# Unit Tests — Utility Functions
# ---------------------------------------------------------------------------

class TestExtractionUtils(unittest.TestCase):
    """Tests for shared utility functions in utils.py (PRD §18, §24, §60)."""

    def setUp(self):
        self.lines = SYNTHETIC_INVOICE_EVIDENCE["pages"][0]["lines"]

    def test_find_key_value_horizontal_found(self):
        val, conf, bbox = find_key_value_horizontal(self.lines, ["invoice date"])
        self.assertIsNotNone(val)
        self.assertIn("2026", val)
        self.assertGreater(conf, 0.0)
        self.assertIsNotNone(bbox)

    def test_find_key_value_horizontal_not_found(self):
        val, conf, bbox = find_key_value_horizontal(self.lines, ["nonexistent_keyword_xyz"])
        self.assertIsNone(val)
        self.assertEqual(conf, 0.0)
        self.assertIsNone(bbox)

    def test_find_amount_near_keyword(self):
        amount, conf, bbox = find_amount_near_keyword(self.lines, ["grand total", "total"])
        self.assertIsNotNone(amount)
        self.assertGreater(amount, 0)

    def test_extract_regex_with_bbox_awb(self):
        sj_lines = SYNTHETIC_SJ_EVIDENCE["pages"][0]["lines"]
        val, conf, bbox = extract_regex_with_bbox(sj_lines, r"\bAWB\d{7,10}\b")
        self.assertIsNotNone(val)
        self.assertIn("AWB", val)

    def test_parse_date_string_iso(self):
        self.assertEqual(parse_date_string("2026-08-10"), "2026-08-10")

    def test_parse_date_string_indonesian(self):
        result = parse_date_string("10 Agustus 2026")
        self.assertEqual(result, "2026-08-10")

    def test_parse_date_string_dmy_slash(self):
        result = parse_date_string("10/08/2026")
        self.assertEqual(result, "2026-08-10")

    def test_parse_date_string_invalid(self):
        self.assertIsNone(parse_date_string("not a date"))

    def test_parse_amount_idr(self):
        self.assertEqual(parse_amount("Rp 75.000"), 75000.0)
        self.assertEqual(parse_amount("IDR 137.500"), 137500.0)

    def test_parse_amount_none(self):
        self.assertIsNone(parse_amount(None))

    def test_parse_table_columns_fuzzy_match(self):
        tables = SYNTHETIC_INVOICE_EVIDENCE["pages"][0]["tables"]
        from app.services.extraction.invoice import INVOICE_TABLE_COLUMNS
        rows = parse_table_columns(tables, INVOICE_TABLE_COLUMNS)
        self.assertGreater(len(rows), 0)
        # AWB column should be matched
        self.assertIn("awb_number", rows[0])
        self.assertIsNotNone(rows[0]["awb_number"])


# ---------------------------------------------------------------------------
# Unit Tests — InvoiceExtractor
# ---------------------------------------------------------------------------

class TestInvoiceExtractor(unittest.TestCase):
    """Tests for InvoiceExtractor (PRD §13, §18, §23, §49, §60)."""

    def setUp(self):
        self.extractor = InvoiceExtractor()
        self.fields = self.extractor.extract("TEST-INV-001", SYNTHETIC_INVOICE_EVIDENCE)
        self.field_map = {f.field: f for f in self.fields}

    # PRD §18: Every field must have an EvidenceSource
    def test_all_fields_have_evidence_source(self):
        for field in self.fields:
            self.assertIsNotNone(field.source, f"Field {field.field} missing EvidenceSource")
            self.assertEqual(field.source.document_id, "TEST-INV-001")

    # PRD §60: Missing fields must be is_unknown=True, value=None
    def test_no_hallucination_rule(self):
        for field in self.fields:
            if field.is_unknown:
                self.assertIsNone(field.value, f"Field {field.field}: is_unknown=True but value is not None")
                self.assertEqual(field.confidence, 0.0, f"Field {field.field}: is_unknown=True but confidence > 0")

    def test_invoice_number_extracted(self):
        f = self.field_map.get("invoice_number")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("INV", f.value)

    def test_vendor_name_extracted(self):
        f = self.field_map.get("vendor_name")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("Cepat Logistik", f.value)

    def test_vendor_id_extracted(self):
        f = self.field_map.get("vendor_id")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertEqual(f.value, "3PL-LOG-001")

    def test_invoice_date_parsed_to_iso(self):
        f = self.field_map.get("invoice_date")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertRegex(f.value, r"\d{4}-\d{2}-\d{2}")

    def test_currency_default_idr(self):
        f = self.field_map.get("currency")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertEqual(f.value, "IDR")

    def test_grand_total_extracted(self):
        f = self.field_map.get("grand_total")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertGreater(f.value, 0)

    def test_line_items_extracted(self):
        f = self.field_map.get("line_items")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertIsInstance(f.value, list)
            self.assertGreater(len(f.value), 0)
            # Check each line item has shipment_id
            for item in f.value:
                self.assertIn("shipment_id", item)

    def test_empty_ocr_result_returns_all_unknown(self):
        fields = self.extractor.extract("EMPTY-DOC", {})
        self.assertTrue(all(f.is_unknown for f in fields))
        self.assertTrue(all(f.value is None for f in fields))


# ---------------------------------------------------------------------------
# Unit Tests — SuratJalanExtractor
# ---------------------------------------------------------------------------

class TestSuratJalanExtractor(unittest.TestCase):
    """Tests for SuratJalanExtractor (PRD §14, §18, §23)."""

    def setUp(self):
        self.extractor = SuratJalanExtractor()
        self.fields = self.extractor.extract("TEST-SJ-001", SYNTHETIC_SJ_EVIDENCE)
        self.field_map = {f.field: f for f in self.fields}

    def test_all_fields_have_evidence_source(self):
        for field in self.fields:
            self.assertIsNotNone(field.source)

    def test_no_hallucination_rule(self):
        for field in self.fields:
            if field.is_unknown:
                self.assertIsNone(field.value)

    def test_surat_jalan_number_extracted(self):
        f = self.field_map.get("surat_jalan_number")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("SJ", f.value)

    def test_awb_extracted(self):
        f = self.field_map.get("awb")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("AWB", f.value)

    def test_actual_weight_extracted_critical(self):
        """actual_weight_kg is critical for pricing reconciliation (PRD §27)."""
        f = self.field_map.get("actual_weight_kg")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown, "actual_weight_kg should be extracted from synthetic data")
        self.assertIsInstance(f.value, float)
        self.assertEqual(f.value, 15.0)

    def test_origin_destination_extracted(self):
        origin = self.field_map.get("origin")
        dest = self.field_map.get("destination")
        self.assertIsNotNone(origin)
        self.assertIsNotNone(dest)

    def test_shipment_date_iso_format(self):
        f = self.field_map.get("shipment_date")
        if f and not f.is_unknown:
            self.assertRegex(f.value, r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# Unit Tests — PODExtractor
# ---------------------------------------------------------------------------

class TestPODExtractor(unittest.TestCase):
    """Tests for PODExtractor (PRD §15, §18, §33)."""

    def setUp(self):
        self.extractor = PODExtractor()
        self.fields = self.extractor.extract("TEST-POD-001", SYNTHETIC_POD_EVIDENCE)
        self.field_map = {f.field: f for f in self.fields}

    def test_all_fields_have_evidence_source(self):
        for field in self.fields:
            self.assertIsNotNone(field.source)

    def test_no_hallucination_rule(self):
        for field in self.fields:
            if field.is_unknown:
                self.assertIsNone(field.value)

    def test_pod_number_extracted(self):
        f = self.field_map.get("pod_number")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("POD", f.value)

    def test_receiver_name_extracted(self):
        f = self.field_map.get("receiver_name")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)

    def test_goods_condition_normalised(self):
        f = self.field_map.get("goods_condition")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertIn(f.value, ["GOOD", "DAMAGED", "SHORTAGE"])

    def test_signature_present_from_phase5_detector(self):
        """
        signature_present must come from Phase 5 SignatureDetector (PRD §33),
        NOT be inferred from OCR text.
        """
        f = self.field_map.get("signature_present")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertTrue(f.value)  # Synthetic evidence has signature_present=True

    def test_signature_bbox_present_when_signature_detected(self):
        f = self.field_map.get("signature_bbox")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertIsInstance(f.value, list)
            self.assertEqual(len(f.value), 4)

    def test_signature_semantics_not_identity(self):
        """
        PRD §33, §70: signature_present=True does NOT mean authenticated identity.
        No field should claim "signature_authentic" or "signature_owner".
        """
        field_names = [f.field for f in self.fields]
        self.assertNotIn("signature_authentic", field_names)
        self.assertNotIn("signature_owner", field_names)
        self.assertNotIn("signature_identity", field_names)


# ---------------------------------------------------------------------------
# Unit Tests — RateAgreementExtractor
# ---------------------------------------------------------------------------

class TestRateAgreementExtractor(unittest.TestCase):
    """Tests for RateAgreementExtractor (PRD §16, §18, §28)."""

    def setUp(self):
        self.extractor = RateAgreementExtractor()
        self.fields = self.extractor.extract("TEST-RA-001", SYNTHETIC_RA_EVIDENCE)
        self.field_map = {f.field: f for f in self.fields}

    def test_all_fields_have_evidence_source(self):
        for field in self.fields:
            self.assertIsNotNone(field.source)

    def test_no_hallucination_rule(self):
        for field in self.fields:
            if field.is_unknown:
                self.assertIsNone(field.value)

    def test_agreement_id_extracted(self):
        f = self.field_map.get("agreement_id")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertIn("AGR", f.value)

    def test_vendor_id_extracted(self):
        f = self.field_map.get("vendor_id")
        self.assertIsNotNone(f)
        self.assertFalse(f.is_unknown)
        self.assertEqual(f.value, "3PL-LOG-001")

    def test_effective_dates_extracted(self):
        eff_from = self.field_map.get("effective_from")
        eff_to = self.field_map.get("effective_to")
        if eff_from and not eff_from.is_unknown:
            self.assertRegex(eff_from.value, r"\d{4}-\d{2}-\d{2}")
        if eff_to and not eff_to.is_unknown:
            self.assertRegex(eff_to.value, r"\d{4}-\d{2}-\d{2}")

    def test_minimum_charge_extracted(self):
        f = self.field_map.get("minimum_charge")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertEqual(f.value, 25000.0)

    def test_dispute_window_extracted(self):
        f = self.field_map.get("dispute_window_days")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertEqual(f.value, 45)

    def test_rate_matrix_extracted(self):
        f = self.field_map.get("rate_matrix")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertIsInstance(f.value, list)
            self.assertGreater(len(f.value), 0)
            # Check structure of first rate rule
            rule = f.value[0]
            self.assertIn("service_type", rule)
            self.assertIn("base_rate_idr", rule)

    def test_surcharge_rules_extracted(self):
        f = self.field_map.get("surcharge_rules")
        self.assertIsNotNone(f)
        if not f.is_unknown:
            self.assertIsInstance(f.value, list)
            types = [s["type"] for s in f.value]
            self.assertIn("fuel_surcharge_percent", types)
            # Verify fuel surcharge value
            fuel = next(s for s in f.value if s["type"] == "fuel_surcharge_percent")
            self.assertEqual(fuel["value"], 14.0)

    def test_raw_extraction_no_pricing_calculation(self):
        """
        PRD §16, §28: RateAgreementExtractor MUST NOT calculate prices.
        It returns raw base_rate values — pricing is computed in backend-api.
        """
        rate_matrix_field = self.field_map.get("rate_matrix")
        if rate_matrix_field and not rate_matrix_field.is_unknown:
            for row in rate_matrix_field.value:
                # Raw numbers only — no computed "total" or "charged" fields
                self.assertNotIn("computed_charge", row)
                self.assertNotIn("expected_total", row)


# ---------------------------------------------------------------------------
# Integration Tests — Against Real Dataset Files
# ---------------------------------------------------------------------------

class TestPhase6Integration(unittest.TestCase):
    """
    Integration tests: run Phase 5 pipeline on real dataset files
    then assert Phase 6 extraction produces non-empty structured results.
    Requires: synthetic-data/dataset/v4_pilot_20 to be present.
    """

    @classmethod
    def setUpClass(cls):
        from tests.support.paths import SYNTHETIC_DATA_ROOT

        cls.dataset_dir = SYNTHETIC_DATA_ROOT / "dataset" / "v4_pilot_20"
        cls.skip = not cls.dataset_dir.exists()

    def _run_pipeline(self, file_path: Path) -> dict:
        from app.services.pipeline import Phase5DocumentPipeline
        pipeline = Phase5DocumentPipeline()
        return pipeline.process_document(file_path)

    def test_invoice_extraction_integration(self):
        if self.skip:
            self.skipTest("Dataset v4_pilot_20 not found")
        invoices = sorted(self.dataset_dir.glob("invoices/*.pdf"))
        if not invoices:
            self.skipTest("No invoice PDFs found")
        evidence = self._run_pipeline(invoices[0])
        extractor = InvoiceExtractor()
        fields = extractor.extract(evidence["document_id"], evidence)
        field_map = {f.field: f for f in fields}

        # All fields must have EvidenceSource (PRD §18)
        for f in fields:
            self.assertIsNotNone(f.source)

        # No hallucination (PRD §60)
        for f in fields:
            if f.is_unknown:
                self.assertIsNone(f.value)

        # At least some critical fields must be non-unknown on a real PDF
        known_fields = [f for f in fields if not f.is_unknown]
        self.assertGreater(len(known_fields), 3,
                           "Expected at least 4 fields to be extracted from a real invoice PDF")

    def test_surat_jalan_extraction_integration(self):
        if self.skip:
            self.skipTest("Dataset v4_pilot_20 not found")
        sj_files = sorted(self.dataset_dir.glob("transactions/*/surat_jalan.pdf"))
        if not sj_files:
            self.skipTest("No surat_jalan.pdf found")
        evidence = self._run_pipeline(sj_files[0])
        extractor = SuratJalanExtractor()
        fields = extractor.extract(evidence["document_id"], evidence)
        for f in fields:
            self.assertIsNotNone(f.source)
            if f.is_unknown:
                self.assertIsNone(f.value)

    def test_pod_extraction_integration(self):
        if self.skip:
            self.skipTest("Dataset v4_pilot_20 not found")
        pod_files = sorted(self.dataset_dir.glob("transactions/*/pod.jpg"))
        if not pod_files:
            self.skipTest("No pod.jpg found")
        evidence = self._run_pipeline(pod_files[0])
        extractor = PODExtractor()
        fields = extractor.extract(evidence["document_id"], evidence)
        field_map = {f.field: f for f in fields}
        # Signature fields should always be present (even if value=False)
        self.assertIn("signature_present", field_map)
        self.assertIn("signature_confidence", field_map)

    def test_rate_agreement_extraction_integration(self):
        if self.skip:
            self.skipTest("Dataset v4_pilot_20 not found")
        ra_files = sorted(self.dataset_dir.glob("transactions/*/rate_agreement.pdf"))
        if not ra_files:
            self.skipTest("No rate_agreement.pdf found")
        evidence = self._run_pipeline(ra_files[0])
        extractor = RateAgreementExtractor()
        fields = extractor.extract(evidence["document_id"], evidence)
        for f in fields:
            self.assertIsNotNone(f.source)
            if f.is_unknown:
                self.assertIsNone(f.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
