from pathlib import Path

from app.services.labeling.grounding import ground_truth_field
from app.services.labeling.schema import FieldStatus, TruthField, TruthRecord
from app.services.labeling.store import LabelStore


def test_label_store_upsert_is_audited_and_idempotent(tmp_path):
    store = LabelStore(tmp_path)
    truth = TruthRecord("D1", "POD", fields={"x": TruthField(FieldStatus.PRESENT, "ABC")})
    label = ground_truth_field(truth, "x", "text", [{"m_candidate_id": "c1", "m_parsed_norm": "abc", "m_raw_text": "ABC", "m_page": "1"}])
    store.upsert(label)
    store.upsert(label)
    assert len(store.list()) == 1
    assert len(store.audit_path.read_text(encoding="utf-8").splitlines()) == 1


def test_grounding_exact_and_missed_statuses():
    truth = TruthRecord("D1", "INVOICE", fields={"invoice_number": TruthField(FieldStatus.PRESENT, "INV-001"), "due_date": TruthField(FieldStatus.PRESENT, "01 Januari 2026")})
    exact = ground_truth_field(truth, "invoice_number", "id", [{"m_candidate_id": "c1", "m_parsed_norm": "inv-001", "m_raw_text": "INV-001", "m_page": "1"}])
    missed = ground_truth_field(truth, "due_date", "date", [{"m_candidate_id": "c2", "m_parsed_norm": "2026-02-01", "m_raw_text": "01 Februari 2026", "m_page": "1"}])
    assert exact.positive_candidate_ids == "c1"
    assert exact.grounding_method == "exact"
    assert missed.status == FieldStatus.MISSED