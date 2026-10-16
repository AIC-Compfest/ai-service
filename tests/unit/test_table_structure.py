"""Header, column, row, role and arbitration behaviour of the geometry-first token_grid provider (Fase 3)."""
from __future__ import annotations

import pymupdf
import pytest

from app.services.candidates.context import build_page
from app.services.candidates.tables import TableExtractor, TokenGridTableProvider
from app.services.candidates.tables.scoring import arbitrate, classify_role, overlap_ratio
from app.tools.table_eval.metrics import evaluate_doc
from app.tools.table_eval.model import gold_from_json, prediction_from_evidence
from app.tools.table_eval.runner import predict_fixture
from app.tools.table_eval.synthetic import build_all, build_fixture, to_evidence_page
from tests.support.paths import AI_SERVICE_ROOT

CONFIG = AI_SERVICE_ROOT / "config" / "candidates"
PROVIDER = TokenGridTableProvider(CONFIG / "line_item_aliases.yaml", CONFIG / "abbreviations.yaml")
ROUTER = TableExtractor.from_config(CONFIG, ["pymupdf_find_tables", "token_grid"])


def _page(rows, char_w=0.011, h=0.012):
    """rows: [(y, [(x, text), ...])] -> candidate Page with 0..1 boxes (text width = chars * char_w)."""
    lines = []
    for y, cells in rows:
        tokens = []
        for x, text in cells:
            tokens.append({"text": text, "confidence": 0.95, "bbox_norm": [x, y, min(x + char_w * len(text), 0.99), y + h]})
        lines.append({"text": " ".join(t["text"] for t in tokens), "bbox_norm": [0, y, 1, y + h], "tokens": tokens})
    return build_page({"page_number": 1, "lines": lines, "width": 595, "height": 842})


def _tables(rows):
    return PROVIDER.tables(_page(rows), None, "d")


ITEM_ROWS = [
    (0.20, [(0.05, "No"), (0.15, "Deskripsi"), (0.55, "Qty"), (0.70, "Harga"), (0.85, "Jumlah")]),
    (0.23, [(0.05, "1"), (0.15, "Pengiriman Jakarta"), (0.57, "2"), (0.70, "1.250.000"), (0.85, "2.500.000")]),
    (0.26, [(0.05, "2"), (0.15, "Asuransi Muatan"), (0.57, "1"), (0.70, "275.000"), (0.85, "275.000")]),
]


# ---- header (F1)

def test_header_is_the_row_above_numeric_rows_not_the_first_row_with_two_cells():
    (table,) = _tables(ITEM_ROWS)
    assert [c.header_text for c in table.columns] == ["No", "Deskripsi", "Qty", "Harga", "Jumlah"]
    assert len(table.rows) == 2


def test_label_value_lines_are_not_headers():
    kv = [(0.10, [(0.05, "Invoice No :"), (0.30, "INV/2026/001")]), (0.12, [(0.05, "Tanggal :"), (0.30, "12 Maret 2026")]), (0.14, [(0.05, "Pelanggan :"), (0.30, "PT Maju")])]
    assert _tables(kv) == []


def test_running_text_is_not_a_header():
    prose = [
        (0.10, [(0.05, "Kami"), (0.13, "dari"), (0.20, "PT"), (0.26, "Bima"), (0.34, "bermaksud"), (0.48, "melakukan"), (0.65, "penagihan")]),
        (0.13, [(0.05, "sebagai"), (0.16, "berikut"), (0.30, "dengan"), (0.45, "rincian")]),
    ]
    assert _tables(prose) == []


def test_first_convincing_header_opens_the_table_and_a_later_one_is_a_second_table():
    rows = ITEM_ROWS + [
        (0.40, [(0.05, "No"), (0.15, "Metode"), (0.55, "Tanggal"), (0.70, "Bank"), (0.85, "Nominal")]),
        (0.43, [(0.05, "1"), (0.15, "Transfer"), (0.55, "12/03/2026"), (0.70, "BCA"), (0.85, "300.000")]),
    ]
    tables = _tables(rows)
    assert [t.columns[1].header_text for t in tables] == ["Deskripsi", "Metode"]
    assert [len(t.rows) for t in tables] == [2, 1]


def test_two_line_header_is_joined_and_the_first_body_row_is_not_swallowed():
    rows = [
        (0.20, [(0.05, "No"), (0.15, "Nama"), (0.55, "Berat"), (0.80, "Jumlah")]),
        (0.215, [(0.15, "Barang"), (0.55, "(Kg)"), (0.80, "Koli")]),
        (0.25, [(0.05, "1"), (0.15, "Pupuk Organik"), (0.57, "50"), (0.82, "200")]),
        (0.28, [(0.05, "2"), (0.15, "Benih Padi"), (0.57, "5"), (0.82, "120")]),
    ]
    (table,) = _tables(rows)
    assert [c.header_text for c in table.columns] == ["No", "Nama Barang", "Berat (Kg)", "Jumlah Koli"]
    assert len(table.rows) == 2


# ---- rows (F3, F9, F10)

def test_total_row_with_empty_first_column_is_a_summary_row_not_a_continuation():
    rows = ITEM_ROWS + [(0.285, [(0.70, "Total"), (0.85, "2.775.000")])]
    (table,) = _tables(rows)
    assert len(table.rows) == 2
    assert [r.cells[3].text for r in table.summary_rows] == ["Total"]
    assert table.summary_rows[0].kind_hint == "total_like"


def test_item_whose_description_starts_with_total_stays_an_item():
    rows = [ITEM_ROWS[0], (0.23, [(0.05, "1"), (0.15, "Total Logistic Service"), (0.57, "2"), (0.70, "1.250.000"), (0.85, "2.500.000")]), ITEM_ROWS[2]]
    (table,) = _tables(rows)
    assert len(table.rows) == 2 and not table.summary_rows


def test_wrapped_description_is_one_row():
    rows = ITEM_ROWS[:2] + [(0.245, [(0.15, "via Semarang")])] + [(0.28, ITEM_ROWS[2][1])]
    (table,) = _tables(rows)
    assert len(table.rows) == 2
    assert table.rows[0].cells[1].text == "Pengiriman Jakarta via Semarang"
    assert table.rows[0].continuation_rows == 1


def test_notes_and_footer_below_the_table_do_not_become_rows_or_summary():
    rows = ITEM_ROWS + [(0.34, [(0.05, "Catatan:")]), (0.36, [(0.05, "1. Pembayaran paling lambat 14 hari setelah invoice")])]
    (table,) = _tables(rows)
    assert len(table.rows) == 2 and not table.summary_rows


# ---- columns (F2)

def test_columns_come_from_data_extent_not_from_page_edges():
    (table,) = _tables(ITEM_ROWS)
    assert table.columns[0].x1 > 0.0 and table.columns[-1].x2 < 1.0
    assert table.bbox[0] >= 0.04 and table.bbox[2] <= 0.99


def test_data_under_no_header_creates_a_headerless_column():
    rows = [ITEM_ROWS[0], (0.23, ITEM_ROWS[1][1] + [(0.40, "pcs")]), (0.26, ITEM_ROWS[2][1] + [(0.40, "box")])]
    (table,) = _tables(rows)
    assert len(table.columns) == 6
    assert any(c.header_text == "" for c in table.columns)


# ---- multi-table (F4, F5)

@pytest.mark.parametrize("mode", ["vector", "phrase"])
def test_side_by_side_tables_stay_separate_and_each_is_complete(mode):
    document = pymupdf.open(stream=build_fixture("side_by_side").pdf, filetype="pdf")
    tables = PROVIDER.tables(build_page(to_evidence_page(document[0], 1, mode)), None, "d")
    assert len(tables) == 2
    left, right = sorted(tables, key=lambda t: t.bbox[0])
    assert left.bbox[2] < right.bbox[0]
    assert len(left.rows) == len(right.rows) == 3


# ---- roles (F6) and arbitration

def test_roles_are_assigned_from_structure():
    (items,) = _tables(ITEM_ROWS)
    assert classify_role(items, None)[0] == "line_items"


def test_router_gives_every_table_a_role_and_prose_pages_have_no_item_table():
    document = pymupdf.open(stream=build_fixture("paragraph_only").pdf, filetype="pdf")
    tables = ROUTER.extract(build_page({**to_evidence_page(document[0], 1, "vector"), "_source_page": document[0]}), None, "d")
    assert all(t.role for t in tables)
    assert not [t for t in tables if t.role == "line_items"]


def test_arbitration_keeps_best_of_overlapping_candidates():
    (a,) = _tables(ITEM_ROWS)
    (b,) = _tables(ITEM_ROWS)
    a.structural_score, b.structural_score = 0.9, 0.5
    assert arbitrate([b, a]) == [a]
    assert overlap_ratio(a.bbox, b.bbox) == pytest.approx(1.0)


# ---- end-to-end regression on the constructed fixtures

ALL_FIXTURES = {f.name for f in build_all()}
# Documented limits (see REPORT_FASE_3): info_block_beside has a 10 pt gutter (adversarial); multipage_no_header needs
# geometry stitching (Fase 4); in phrase mode adjacent cells of blank_column / header_two_line / side_by_side are merged
# into ONE OCR phrase box and cannot be separated from geometry alone.
KNOWN_LIMITS = {"vector": {"info_block_beside", "multipage_no_header"}, "phrase": {"info_block_beside", "multipage_no_header", "blank_column", "header_two_line", "side_by_side"}}


def _failures(fixture, mode, view="line_items"):
    prediction = predict_fixture(fixture.pdf, "router_none", mode, f"constructed/{fixture.name}")
    gold = gold_from_json(fixture.gold, allow_unreviewed=True)
    return dict(evaluate_doc(gold, prediction_from_evidence(prediction, gold.document_id), view=view).failures)


@pytest.mark.parametrize("mode", ["vector", "phrase"])
def test_constructed_fixtures_are_exact_in_the_line_items_view(mode):
    broken = {f.name: _failures(f, mode) for f in build_all() if f.name not in KNOWN_LIMITS[mode]}
    assert {name: codes for name, codes in broken.items() if codes} == {}


def test_known_limits_are_still_limits_so_the_list_cannot_go_stale():
    still_failing = {name for name in KNOWN_LIMITS["vector"] if _failures(build_fixture(name), "vector")}
    assert still_failing == KNOWN_LIMITS["vector"]


def test_multi_table_pages_are_fully_correct_on_vector_fixtures():
    for name in ("stacked_two_tables", "side_by_side", "two_tables_ruled_stacked", "three_tables_mixed"):
        assert _failures(build_fixture(name), "vector", view="any") == {}, name
