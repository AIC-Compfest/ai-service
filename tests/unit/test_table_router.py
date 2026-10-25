from pathlib import Path

from app.services.candidates.context import build_page
from app.services.candidates.schema import Cell, Column, Table, TableRow
from app.services.candidates.tables.router import TableExtractor
from app.services.candidates.pipeline import table_evidence, stitch_tables
from app.services.candidates.pipeline import CandidatePipeline
from app.core.config import settings


def _table(source, filled=2, count=1, top=.1):
    columns = [Column(0, 0, .5, "Item"), Column(1, .5, 1, "Qty")]
    rows = [TableRow(i, [Cell("x" if n < filled else "", (0, top, .5, top + .1), col_idx=0), Cell("1", (.5, top, 1, top + .1), col_idx=1)], (0, top, 1, top + .1)) for i, n in enumerate(range(count))]
    return Table("t", 1, 1, "columnar", source, columns, rows, (0, top, 1, top + .1))


class Provider:
    def __init__(self, name, result): self.name, self.result = name, result
    def tables(self, page, doc_type, document_id="document"): return self.result


def test_router_falls_back_when_first_provider_has_no_tables():
    page = build_page({"page_number": 1, "lines": []})
    assert TableExtractor([Provider("empty", []), Provider("fallback", [_table("fallback")])]).ordered(["empty", "fallback"]).extract(page, "INVOICE")[0].source == "fallback"


def test_router_keeps_the_best_candidate_of_each_overlapping_group_and_all_distinct_tables():
    """Arbitration is per table: the weak duplicate loses, the two distinct strong tables both survive."""
    page = build_page({"page_number": 1, "lines": []})
    weak = _table("weak", 0)
    strong_a, strong_b = _table("strong"), _table("strong", count=2, top=.5)
    selected = TableExtractor([Provider("weak", [weak]), Provider("strong", [strong_a, strong_b])]).ordered(["weak", "strong"]).extract(page, "INVOICE")
    assert len(selected) == 2 and all(table.source == "strong" for table in selected)
    assert [table.table_id for table in selected] == ["document-p1-t1", "document-p1-t2"]


def test_router_mixes_providers_when_they_find_different_tables():
    page = build_page({"page_number": 1, "lines": []})
    selected = TableExtractor([Provider("a", [_table("a", top=.1)]), Provider("b", [_table("b", top=.6)])]).ordered(["a", "b"]).extract(page, None)
    assert sorted(table.source for table in selected) == ["a", "b"]


def test_router_assigns_a_role_with_a_reason():
    page = build_page({"page_number": 1, "lines": []})
    (table,) = TableExtractor([Provider("a", [_table("a")])]).ordered(["a"]).extract(page, None)
    assert table.role in {"line_items", "summary", "info", "other"} and table.role_reason


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
    pipeline.table_extractor = TableExtractor([Provider("docling", [_table("docling")])]).ordered(["docling"])  # no Docling models in unit tests
    pipeline.generate("D1", "SURAT_JALAN", evidence)
    assert pipeline.last_report["tables"][0]["provider"] == "docling"


def test_stitching_joins_only_consecutive_tables_with_same_headers():
    first, second = _table("token_grid"), _table("token_grid")
    first.page_start = first.page_end = 1; first.stitched_pages = [1]
    second.page_start = second.page_end = 2; second.stitched_pages = [2]
    result = stitch_tables([first, second])
    assert len(result) == 1
    assert result[0].stitched_pages == [1, 2]
    assert [row.idx for row in result[0].rows] == [0, 1]


def test_stitching_rejects_header_or_page_gaps():
    first, second = _table("token_grid"), _table("token_grid")
    second.page_start = second.page_end = 3
    second.columns[0].header_text = "Different"
    assert len(stitch_tables([first, second])) == 2
