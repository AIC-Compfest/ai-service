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
    evidence = {"pages": [{"page_number": 1, "engine_used": "easyocr_gpu", "lines": [
        _line("Surcharge 11%", 100, 400, 400, 420),
        _line("Payment term 30 days", 100, 440, 500, 460),
    ]}]}

    rows = CandidatePipeline("config/candidates").generate("R1", "RATE_AGREEMENT", evidence, "run")

    assert any(row["m_cand_type"] == "percent" and row["m_parsed_value"] == "11" for row in rows)
    assert any(row["m_cand_type"] == "days" and row["m_parsed_value"] == "30" for row in rows)
