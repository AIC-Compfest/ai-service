from app.services.candidates.pipeline import CandidatePipeline
from app.services.evidence.docling_tables import normalize_docling_evidence


def _table():
    rows = [["Jasa Desain Website", "1", "12000000", "12000000"], ["Hosting & Domain", "1", "2400000", "2400000"], ["Maintenance Bulanan", "3", "500000", "1500000"]]
    return {"table_id": "D-p1-t1", "bbox_norm": [.05, .4, .95, .7], "headers": ["Jenis Jasa Service Type", "Qty / Pg Jmlh / Lbr", "Unit Price Hrga Lmbr", "Jumlah Harga Total Price"], "columns": [{"idx": i, "bbox_norm": [0, 0, 1, 1], "header": ""} for i in range(4)], "cells": [{"row": r, "col": c, "text": rows[r][c], "bbox_norm": [0, 0, 1, 1]} for r in range(3) for c in range(4)], "rows": rows, "row_count": 3, "col_count": 4, "summary_rows": [["", "", "Subtotal", "15900000"], ["", "", "Pajak", "1749000"]], "confidence": .9, "provider": "docling", "reason": "docling:text", "validation": {"page_kind": "DIGITAL"}, "role": "line_items", "role_reason": "numeric_grid"}


def test_docling_adapter_and_one_table_candidate():
    evidence = {"pages": [{"page_number": 1, "coord_space": "page_image_normalized", "engine_used": "docling_vector", "lines": [], "tables": [_table()]}]}
    normalized, warnings = normalize_docling_evidence(evidence)
    assert not warnings
    row = next(row for row in CandidatePipeline("config/candidates").generate("D", "INVOICE", normalized) if row["m_cand_type"] == "table")
    assert row["m_parsed_norm"] == "3x4"
    assert row["m_line_idx"] == ""
    assert row["m_evidence_role"] == "line_items"
    assert row["tb_n_rows"] == 3
    assert row["tb_has_total_summary"] == 1
    assert row["tb_summary_matches_colsum"] == 1


def test_legacy_coordinates_are_normalized_and_unknown_space_rejected():
    table = _table()
    table["bbox_norm"] = [50, 400, 950, 700]
    evidence = {"pages": [{"page_number": 1, "coord_space": "legacy_1000", "tables": [table]}]}
    normalized, _ = normalize_docling_evidence(evidence)
    assert normalized["pages"][0]["tables"][0]["bbox_norm"] == [.05, .4, .95, .7]
    bad = {"pages": [{"page_number": 1, "coord_space": "pixels", "tables": []}]}
    try:
        normalize_docling_evidence(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown coordinate space must be rejected")
