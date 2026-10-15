from pathlib import Path

from app.services.candidates.context import build_page
from app.services.candidates.schema import Cell, Column, Table, TableRow
from app.services.candidates.tables.router import TableExtractor
from app.services.candidates.pipeline import table_evidence
from app.services.candidates.pipeline import CandidatePipeline
from app.core.config import settings


def _table(source, filled=2, count=1):
    columns = [Column(0, 0, .5, "Item"), Column(1, .5, 1, "Qty")]
    rows = [TableRow(i, [Cell("x" if n < filled else "", (0, .1, .5, .2), col_idx=0), Cell("1", (.5, .1, 1, .2), col_idx=1)], (0, .1, 1, .2)) for i, n in enumerate(range(count))]
    return Table("t", 1, 1, "columnar", source, columns, rows, (0, .1, 1, .2))


class Provider:
    def __init__(self, name, result): self.name, self.result = name, result
    def tables(self, page, doc_type, document_id="document"): return self.result


def test_router_falls_back_when_first_provider_has_no_tables():
    page = build_page({"page_number": 1, "lines": []})
    assert TableExtractor([Provider("empty", []), Provider("fallback", [_table("fallback")])]).ordered(["empty", "fallback"]).extract(page, "INVOICE")[0].source == "fallback"


def test_router_selects_structural_score_and_preserves_multiple_tables():
    page = build_page({"page_number": 1, "lines": []})
    selected = TableExtractor([Provider("weak", [_table("weak", 0)]), Provider("strong", [_table("strong"), _table("strong", count=2)])]).ordered(["weak", "strong"]).extract(page, "INVOICE")
    assert len(selected) == 2 and all(table.source == "strong" for table in selected)


def test_table_evidence_keeps_legacy_fields_and_adds_metadata():
    payload = table_evidence(_table("token_grid"))
    assert {"bbox_norm", "headers", "rows", "row_count", "col_count", "confidence"} <= payload.keys()
    assert {"provider", "reason", "validation", "columns", "cells"} <= payload.keys()


def test_candidate_runtime_emits_router_diagnostics_only_when_flagged(monkeypatch):
    monkeypatch.setattr(settings, "enable_table_extractor", True)
    evidence = {"pages": [{"page_number": 1, "lines": [
        {"text": "No Koli Berat", "bbox_norm": [.05, .2, .9, .22], "tokens": [
            {"text": "No", "bbox_norm": [.05, .2, .1, .22]}, {"text": "Koli", "bbox_norm": [.4, .2, .5, .22]}, {"text": "Berat", "bbox_norm": [.75, .2, .9, .22]}]},
        {"text": "1 10 25", "bbox_norm": [.05, .25, .85, .27], "tokens": [
            {"text": "1", "bbox_norm": [.05, .25, .1, .27]}, {"text": "10", "bbox_norm": [.4, .25, .5, .27]}, {"text": "25", "bbox_norm": [.75, .25, .85, .27]}]},
    ]}]}
    pipeline = CandidatePipeline(Path("config/candidates"))
    pipeline.generate("D1", "SURAT_JALAN", evidence)
    assert pipeline.last_report["tables"][0]["provider"] == "token_grid"
