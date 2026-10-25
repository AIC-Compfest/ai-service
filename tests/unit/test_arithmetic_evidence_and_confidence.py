"""Fase 5: arithmetic evidence on the extraction response and the source of confidences."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.confidence import LINE_SCAN_CONSTANT, MAX_TABLE_DERIVED, table_derived_confidence
from app.services.extraction.invoice import InvoiceExtractor
from app.services.validation.document import validate_extraction
from app.tools.table_eval.synthetic import build_fixture


def _field(name, value):
    return ExtractedField(field=name, value=value, confidence=0.9, source=EvidenceSource(document_id="d", page=1), method=ExtractionMethod.RULE_BASED, is_unknown=value is None)


def _page(rows, headers=("No", "Item", "Qty", "Unit Price", "Amount")):
    return {"page_number": 1, "tables": [{"table_id": "t1", "role": "line_items", "headers": list(headers), "rows": rows}]}


def test_extraction_evidence_checks_rows_subtotal_and_grand_total():
    pages = [_page([["1", "a", "2", "1.000", "2.000"], ["2", "b", "1", "500", "500"]])]
    fields = [_field("subtotal", 2500), _field("tax_amount", 275), _field("grand_total", 2775), _field("currency", "IDR")]
    result = validate_extraction(fields, pages)
    assert result["status"] == "ok" and result["coverage"] == {"tables": 1, "tables_with_rules": 1, "rows_total": 2, "rows_evaluated": 2}
    assert [c["status"] for c in result["totals"]["checks"]] == ["PASS", "PASS"]
    assert result["failed_rows_total"] == 0 and result["caveats"]


def test_a_wrong_row_and_a_wrong_subtotal_are_reported_without_touching_values():
    pages = [_page([["1", "a", "2", "1.000", "2.100"]])]
    fields = [_field("subtotal", 9999), _field("currency", "IDR")]
    result = validate_extraction(fields, pages)
    assert result["failed_rows_total"] == 1 and result["failed_rows"][0]["expected"] == "2000"
    assert result["totals"]["checks"][0]["status"] == "FAIL"
    assert fields[0].value == 9999  # input untouched


def test_other_roles_and_missing_columns_are_not_applicable_not_failed():
    info = {"page_number": 1, "tables": [{"table_id": "i", "role": "info", "headers": ["Qty", "Rate", "Total"], "rows": [["1", "1", "9"]]}]}
    assert validate_extraction([], [info])["status"] == "not_applicable"
    no_rules = validate_extraction([], [_page([["1", "x"]], headers=("No", "Item"))])
    assert no_rules["coverage"]["tables_with_rules"] == 0 and no_rules["coverage"]["rows_evaluated"] == 0


def test_evidence_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_arithmetic_evidence", False)
    assert validate_extraction([], [_page([["1", "a", "2", "1.000", "2.000"]])]) == {"status": "disabled"}


def test_table_derived_confidence_uses_the_weakest_structural_score_and_never_claims_certainty():
    tables = [{"validation": {"structural": {"score": 0.97}}}, {"validation": {"structural": {"score": 0.71}}}]
    assert table_derived_confidence(tables, True) == (0.71, "table_structural_score")
    assert table_derived_confidence([{"validation": {"structural": {"score": 1.0}}}], True)[0] == MAX_TABLE_DERIVED
    assert table_derived_confidence([], True) == (LINE_SCAN_CONSTANT, "line_scan_heuristic")
    assert table_derived_confidence(tables, False) == (0.0, "no_value")


def test_invoice_line_items_carry_a_confidence_source():
    page = {"page_number": 1, "lines": [{"text": "INVOICE", "bbox_norm": [0.1, 0.01, 0.4, 0.03], "tokens": []}],
            "tables": [{"role": "line_items", "headers": ["No", "Shipment ID", "Total"], "rows": [["1", "SHP-0001-X", "1000"]], "bbox_norm": [0.1, 0.3, 0.9, 0.4], "validation": {"structural": {"score": 0.8}}}]}
    fields = {f.field: f for f in InvoiceExtractor().extract("d", {"pages": [page]})}
    assert fields["line_items"].confidence_source == "table_structural_score" and fields["line_items"].confidence == 0.8


def test_table_evidence_reports_the_structural_score_as_confidence():
    from app.services.pipeline import Phase5DocumentPipeline

    settings.enable_table_extractor = True
    evidence = Phase5DocumentPipeline().process_document(build_fixture("single_ruled").pdf, filename="s.pdf", document_id="it-conf")
    (table,) = [t for t in evidence["pages"][0]["tables"] if t["role"] == "line_items"]
    assert table["confidence"] > 0 and table["confidence_source"] == "structural_score"


def test_full_pipeline_response_has_arithmetic_validation():
    pdf = build_fixture("single_ruled").pdf
    body = TestClient(app).post("/v1/extract/pipeline/full", files={"file": ("s.pdf", pdf, "application/pdf")}).json()
    assert "arithmetic_validation" in body and "status" in body["arithmetic_validation"]
