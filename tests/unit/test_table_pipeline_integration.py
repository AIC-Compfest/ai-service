"""Type-aware table discovery wired into Phase5DocumentPipeline (K1), continuation and stitching (F11)."""
from __future__ import annotations

import copy

import pymupdf
import pytest

from app.core.config import settings
from app.services.candidates.pipeline import stitch_tables
from app.services.candidates.schema import Cell, Column, Table, TableRow
from app.services.extraction.invoice import InvoiceExtractor
from app.services.ocr.engine import OCREngine
from app.services.pipeline import Phase5DocumentPipeline
from app.tools.table_eval.synthetic import build_fixture

PIPELINE = Phase5DocumentPipeline()


def _run(name: str, flag: bool, monkeypatch) -> dict:
    monkeypatch.setattr(settings, "enable_table_extractor", flag)
    return PIPELINE.process_document(build_fixture(name).pdf, filename=f"{name}.pdf", document_id=f"it-{name}-{flag}")


def test_flag_off_keeps_tables_empty_and_groups_empty(monkeypatch):
    evidence = _run("single_unruled", False, monkeypatch)
    assert all(page["tables"] == [] for page in evidence["pages"])
    assert evidence["table_groups"] == []


def test_flag_on_fills_tables_with_roles_and_additive_fields(monkeypatch):
    evidence = _run("single_unruled", True, monkeypatch)
    (table,) = [t for t in evidence["pages"][0]["tables"] if t["role"] == "line_items"]
    assert table["headers"] == ["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"]
    assert table["row_count"] == 6
    assert {"role", "role_reason", "summary_rows", "table_id", "provider", "columns", "cells"} <= table.keys()
    # legacy fields are untouched
    assert {"bbox_norm", "headers", "rows", "row_count", "col_count", "confidence"} <= table.keys()


def test_ocr_engine_skips_its_typeless_discovery_when_the_pipeline_owns_it(monkeypatch):
    monkeypatch.setattr(settings, "enable_table_extractor", True)
    document = pymupdf.open(stream=build_fixture("single_ruled").pdf, filetype="pdf")
    engine = OCREngine()
    assert engine.process_pdf_page(document[0], 1, {}, discover_tables=False).tables == []
    assert len(engine.process_pdf_page(document[0], 1, {}).tables) == 1


def test_multipage_table_with_repeated_header_forms_one_group(monkeypatch):
    evidence = _run("multipage_repeated_header", True, monkeypatch)
    (group,) = evidence["table_groups"]
    assert group["pages"] == [1, 2] and group["row_count"] == 24
    assert group["summary_rows"]  # the Total row of the last page


def test_multipage_table_without_repeated_header_is_continued_by_geometry(monkeypatch):
    """Page 2 has no header row at all: rows are recovered with the columns of page 1 (F11)."""
    evidence = _run("multipage_no_header", True, monkeypatch)
    (group,) = evidence["table_groups"]
    assert group["pages"] == [1, 2] and group["row_count"] == 24
    page2 = [t for t in evidence["pages"][1]["tables"] if t["role"] == "line_items"]
    assert page2 and page2[0]["reason"] == "continuation" and page2[0]["continued_from"]
    assert page2[0]["headers"] == ["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"]


def test_extractor_ignores_tables_that_are_not_line_items():
    def table(role, header):
        return {"role": role, "headers": [header, "Total"], "rows": [["SHP-0001 x", "1000"]], "bbox_norm": [0.1, 0.1, 0.9, 0.2]}

    evidence = {"pages": [{"page_number": 1, "lines": [{"text": "INVOICE", "bbox_norm": [0.1, 0.01, 0.4, 0.03], "tokens": []}], "tables": [table("info", "Shipment id"), table("line_items", "Shipment id")]}]}
    fields = {f.field: f for f in InvoiceExtractor().extract("d", evidence)}
    assert len(fields["line_items"].value) == 1


# ---- stitch_tables

def _table(table_id, page, rows, headers=("Item", "Qty"), summary=0, reason="", validation=None):
    columns = [Column(i, i / 2, (i + 1) / 2, h) for i, h in enumerate(headers)]
    body = [TableRow(i, [Cell(f"{table_id}-{i}-{c}", (0, 0, 1, 1), row_idx=i, col_idx=c) for c in range(len(headers))], (0, 0, 1, 1)) for i in range(rows)]
    total = [TableRow(i, [Cell("Total", (0, 0, 1, 1)), Cell("9", (0, 0, 1, 1))], (0, 0, 1, 1), kind_hint="total_like") for i in range(summary)]
    return Table(table_id, page, page, "columnar", "token_grid", columns, body, (0, 0.5, 1, 0.9), stitched_pages=[page], summary_rows=total, reason=reason, validation=validation or {})


def test_stitching_does_not_mutate_its_inputs():
    first, second = _table("a", 1, 2), _table("b", 2, 3)
    snapshot = copy.deepcopy((first, second))
    (joined,) = stitch_tables([first, second])
    assert len(joined.rows) == 5 and [row.idx for row in joined.rows] == [0, 1, 2, 3, 4]
    assert (first.rows[0].idx, len(first.rows), len(second.rows), second.rows[0].idx) == (0, 2, 3, 0)
    assert [r.idx for r in second.rows] == [r.idx for r in snapshot[1].rows]


def test_a_table_that_ended_with_a_total_row_is_not_joined_to_the_next_page():
    assert len(stitch_tables([_table("a", 1, 2, summary=1), _table("b", 2, 3)])) == 2


def test_continuation_joins_only_the_table_it_continues():
    first = _table("a", 1, 2)
    cont = _table("c", 2, 2, reason="continuation", validation={"continued_from": "a"})
    stranger = _table("c2", 2, 2, reason="continuation", validation={"continued_from": "zzz"})
    assert len(stitch_tables([first, cont])) == 1
    assert len(stitch_tables([first, stranger])) == 2


def test_tables_on_one_page_are_ordered_naturally_not_as_strings():
    ids = ["d-p1-t10", "d-p1-t2", "d-p1-t1"]
    result = stitch_tables([_table(i, 1, 1, headers=(i, "Qty")) for i in ids])
    assert [t.table_id for t in result] == ["d-p1-t1", "d-p1-t2", "d-p1-t10"]


# ---- continuation guard (found on rate-card documents: key-value blocks were "continued")

def test_only_the_lowest_data_table_of_a_page_can_be_continued():
    from app.services.candidates.table_runtime import _can_continue

    rules = {"min_tail_rows": 3, "min_tail_columns": 3, "bottom_tolerance": 0.02}
    data = _table("a", 1, 5, headers=("Item", "Qty", "Price"))
    below = _table("b", 1, 5, headers=("Item", "Qty", "Price"))
    below.bbox = (0, 0.92, 1, 0.97)
    assert _can_continue(data, [data], rules)
    assert not _can_continue(data, [data, below], rules)  # another table sits lower on the page
    assert not _can_continue(_table("k", 1, 1, headers=("Key", "Value", "Note")), [], rules)  # a one-row key-value block
    assert not _can_continue(_table("t", 1, 5, summary=1, headers=("Item", "Qty", "Price")), [], rules)  # it already ended
