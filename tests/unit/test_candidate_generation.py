from decimal import Decimal

from app.services.candidates.parsers import parse_date, parse_number, parse_time
from app.services.candidates.pipeline import CandidatePipeline
from app.tools.build_candidate_dataset import restrict_pages


def _line(text, x1, y1, x2, y2):
    x1, y1, x2, y2 = (value / 1000 for value in (x1, y1, x2, y2))
    words = text.split()
    width = (x2 - x1) / len(words)
    return {"text": text, "confidence": 0.95, "bbox_norm": [x1, y1, x2, y2], "tokens": [
        {"text": word, "confidence": 0.95, "bbox_norm": [x1 + i * width, y1, x1 + (i + 1) * width, y2]}
        for i, word in enumerate(words)
    ]}


def test_number_vectors():
    assert parse_number("Rp1.250.000").value == Decimal("1250000")
    assert parse_number("1.250").value == Decimal("1250")
    assert parse_number("1.250").value_alt == Decimal("1.250")
    assert parse_number("1.250.000,50").value == Decimal("1250000.50")
    assert parse_number("(1.250.000)").value == Decimal("-1250000")
    assert parse_number("1.250.000.5").value is None


def test_date_and_time_vectors():
    assert parse_date("05 Juli 2026").iso == "2026-07-05"
    assert parse_date("05/07/2026").ambiguous_dm
    assert parse_date("31/02/2026").iso is None
    assert parse_time("14:30 WIB")["value"] == "14:30"


def test_invoice_candidate_contract_and_context(tmp_path):
    evidence = {"pages": [{"page_number": 1, "engine_used": "pymupdf_vector", "lines": [
        _line("PT Cepat Logistik [VENDOR ID: 3PL-LOG-001]", 60, 30, 560, 50),
        _line("INVOICE", 400, 90, 600, 120),
        _line("No. Invoice : INV-2026-000123", 60, 160, 460, 180),
        _line("Tanggal : 05 Juli 2026", 60, 190, 400, 210),
        _line("Jatuh Tempo : 04 Agustus 2026", 60, 220, 460, 240),
        _line("Kepada Yth : PT Maju Jaya", 60, 270, 420, 290),
        _line("Subtotal Rp 1.250.000", 500, 760, 940, 780),
        _line("PPN 11% Rp 137.500", 500, 790, 940, 810),
        _line("Total Tagihan Rp 1.387.500", 500, 820, 940, 840),
    ]}]}
    rows = CandidatePipeline("config/candidates").generate("D1", "INVOICE", evidence, "run")
    grouped = {kind: [row for row in rows if row["m_cand_type"] == kind] for kind in {row["m_cand_type"] for row in rows}}
    assert len(grouped["amount"]) == 3
    assert {row["m_parsed_value"] for row in grouped["date"]} == {"2026-07-05", "2026-08-04"}
    assert any(row["m_raw_text"] == "1.250.000" and row["ctx_left"] == "Subtotal Rp" for row in grouped["amount"])
    assert all(row["label__subtotal"] == "" for row in grouped["amount"])
    assert all(0 <= row["m_x1"] < row["m_x2"] <= 1 for row in rows)


def test_shared_evidence_can_be_restricted_to_logical_segment():
    evidence = {"pages": [{"page_number": 1}, {"page_number": 2}, {"page_number": 3}]}
    selected = restrict_pages(evidence, "2", "3")
    assert [page["page_number"] for page in selected["pages"]] == [2, 3]
    assert [page["page_number"] for page in evidence["pages"]] == [1, 2, 3]


def test_rate_agreement_emits_configured_percent_and_days_candidates():
    evidence = {"pages": [{"page_number": 1, "engine_used": "paddleocr", "lines": [
        _line("Surcharge 11%", 100, 400, 400, 420),
        _line("Payment term 30 days", 100, 440, 500, 460),
    ]}]}

    rows = CandidatePipeline("config/candidates").generate("R1", "RATE_AGREEMENT", evidence, "run")

    assert any(row["m_cand_type"] == "percent" and row["m_parsed_value"] == "11" for row in rows)
    assert any(row["m_cand_type"] == "days" and row["m_parsed_value"] == "30" for row in rows)


def test_multitoken_amount_and_numeric_id_overlap():
    evidence = {"pages": [{"page_number": 1, "engine_used": "paddleocr", "lines": [
        _line("Total Due Rp 25 ,000 ,000", 100, 700, 800, 730),
        _line("Invoice 0309002", 100, 100, 500, 130),
    ]}]}
    rows = CandidatePipeline("config/candidates").generate("D2", "INVOICE", evidence)
    amount = next(row for row in rows if row["m_cand_type"] == "amount" and row["m_parsed_value"] == "25000000")
    assert amount["m_raw_text"] == "Rp 25 ,000 ,000"
    assert amount["a_n_tokens"] == 4 and amount["a_has_currency_sym"] == 1
    numeric = [row for row in rows if row["m_raw_text"] == "0309002"]
    assert {row["m_cand_type"] for row in numeric} == {"amount", "id"}
    assert all(row["m_overlap_types"] == "amount|id" for row in numeric)


def test_fuzzy_month_date_is_not_an_id():
    evidence = {"pages": [{"page_number": 1, "lines": [_line("Date 12-Har-09", 100, 100, 500, 130)]}]}
    rows = CandidatePipeline("config/candidates").generate("D3", "INVOICE", evidence)
    dates = [row for row in rows if row["m_cand_type"] == "date"]
    assert any(row["m_parsed_value"] == "2009-03-12" and row["dt_month_fuzzy"] == 1 for row in dates)
    assert not any(row["m_cand_type"] == "id" and "12-Har-09" in row["m_raw_text"] for row in rows)


def test_caps_are_applied_and_reported():
    evidence = {"pages": [{"page_number": 1, "lines": [_line(f"Total Rp {i}.000", 100, 100 + i * 20, 500, 120 + i * 20) for i in range(1, 5)]}]}
    pipeline = CandidatePipeline("config/candidates")
    pipeline.caps["amount"] = 2
    rows = pipeline.generate("D4", "INVOICE", evidence)
    assert len([row for row in rows if row["m_cand_type"] == "amount"]) == 2
    assert pipeline.last_run_report["cap_overflow"] == {"1": {"amount": 2}}
