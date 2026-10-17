"""Clean-up of PyMuPDF tables: trailing summary rows and header-less tables (found with the stress set)."""
from __future__ import annotations

from app.services.candidates.schema import Cell, Column, Table, TableRow
from app.services.candidates.tables.refine import headerless_to_data, refine_pymupdf_table, split_summary_rows
from app.services.candidates.tables.rows import summary_pattern

PATTERN = summary_pattern(["total", "subtotal", "ppn", "diskon"])


def _table(headers: list[str], rows: list[list[str]]) -> Table:
    n = len(headers)
    columns = [Column(i, i / n, (i + 1) / n, h, [h]) for i, h in enumerate(headers)]
    body = []
    for r, texts in enumerate(rows):
        top = 0.2 + r * 0.02
        cells = [Cell(text, (c / n, top, (c + 1) / n, top + 0.02), row_idx=r, col_idx=c) for c, text in enumerate(texts)]
        body.append(TableRow(r, cells, (0, top, 1, top + 0.02)))
    return Table("t", 1, 1, "grid", "pymupdf_find_tables", columns, body, (0.0, 0.18, 1.0, 0.2 + len(rows) * 0.02))


ITEMS = [["1", "Kardus", "2", "50.000", "100.000"], ["2", "Lakban", "5", "10.000", "50.000"], ["3", "Palet", "1", "75.000", "75.000"]]


def test_trailing_rows_with_a_known_label_become_summary_rows():
    table = _table(["No", "Item", "Qty", "Harga", "Jumlah"], ITEMS + [["", "", "", "Subtotal", "225.000"], ["", "", "", "PPN 11%", "24.750"]])
    assert split_summary_rows(table, PATTERN) == 2
    assert len(table.rows) == 3 and [r.cells[3].text for r in table.summary_rows] == ["Subtotal", "PPN 11%"]
    assert table.bbox[3] == max(c.bbox[3] for r in table.rows for c in r.cells)  # the box ends with the last ITEM row


def test_an_unknown_label_next_to_an_amount_is_a_summary_row_by_structure():
    table = _table(["No", "Item", "Qty", "Harga", "Jumlah"], ITEMS + [["", "", "", "Biaya Kirim", "8.620.000"]])
    assert split_summary_rows(table, PATTERN) == 1 and len(table.rows) == 3


def test_an_item_whose_description_starts_with_total_stays_an_item():
    rows = ITEMS[:2] + [["3", "Total Logistic Service Surabaya", "1", "75.000", "75.000"]]
    table = _table(["No", "Item", "Qty", "Harga", "Jumlah"], rows)
    assert split_summary_rows(table, PATTERN) == 0 and len(table.rows) == 3


def test_a_full_row_without_a_label_is_never_taken_as_summary():
    table = _table(["No", "Item", "Qty", "Harga", "Jumlah"], ITEMS)
    assert split_summary_rows(table, None) == 0


def test_at_least_one_item_row_always_remains():
    table = _table(["No", "Item", "Harga", "Jumlah"], [["", "", "Subtotal", "1.000"], ["", "", "Total", "1.000"]])
    split_summary_rows(table, PATTERN)
    assert len(table.rows) >= 1


def test_a_data_looking_header_becomes_the_first_row_and_the_headers_are_blank():
    table = _table(["1", "INV6012283", "Pengemasan Kayu", "31.7", "2.848.000"], [["2", "RS2163558", "Customs", "33.7", "3.800.000"], ["3", "INV5658866", "Customs", "52.2", "823.000"]])
    assert headerless_to_data(table) is True
    assert [r.cells[1].text for r in table.rows] == ["INV6012283", "RS2163558", "INV5658866"]
    assert all(c.header_text == "" for c in table.columns) and [r.idx for r in table.rows] == [0, 1, 2]


def test_a_real_header_is_left_alone():
    table = _table(["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"], ITEMS)
    assert headerless_to_data(table) is False and table.columns[1].header_text == "Deskripsi" and len(table.rows) == 3


def test_a_header_with_units_and_a_year_is_not_mistaken_for_data():
    table = _table(["Tgl 2026", "Berat (Kg)", "Rp"], ITEMS[:2])
    assert headerless_to_data(table) is False


def test_refine_applies_both_steps():
    table = _table(["1", "INV1", "x", "2.000", "10.000"], [["2", "INV2", "y", "3.000", "20.000"], ["", "", "", "Total", "30.000"]])
    refine_pymupdf_table(table, PATTERN)
    assert len(table.rows) == 2 and len(table.summary_rows) == 1 and table.columns[0].header_text == ""
