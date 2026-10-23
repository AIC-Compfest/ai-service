"""Docling integration: adapter, missed-table check and router flow. No Docling install or model download needed."""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace as NS

from app.services.candidates.schema import Cell, Column, Page, Table, TableRow, Token
from app.services.candidates.tables.missed import suspect_missed_tables
from app.services.candidates.tables.router import TableExtractor
from app.services.docling.adapter import docling_table_to_internal, normalize_box

W, H = 600.0, 800.0


class Origin(Enum):
    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"


def box(l, t, r, b, origin=Origin.TOPLEFT):
    return NS(l=l, t=t, r=r, b=b, coord_origin=origin)


def cell(r, c, text, bbox=None, header=False, r2=None, c2=None):
    return NS(start_row_offset_idx=r, end_row_offset_idx=r2 or r + 1, start_col_offset_idx=c, end_col_offset_idx=c2 or c + 1, text=text, column_header=header, row_header=False, bbox=bbox)


def item(cells, rows, cols, bbox):
    return NS(data=NS(table_cells=cells, num_rows=rows, num_cols=cols), prov=[NS(page_no=1, bbox=bbox)])


def make_item():
    # Table box is BOTTOMLEFT (t above b); cells are TOPLEFT -- the mix seen in real Docling output.
    table_box = box(60, 700, 540, 500, Origin.BOTTOMLEFT)  # -> y from 100 to 300 in top-left terms
    cells = [
        cell(0, 0, "Deskripsi", box(60, 100, 300, 130), True),
        cell(0, 1, "Qty", box(300, 100, 400, 130), True),
        cell(0, 2, "Jumlah", box(400, 100, 540, 130), True),
        cell(1, 0, "Pengiriman  Jakarta", box(60, 130, 300, 160)),
        cell(1, 1, "2", box(300, 130, 400, 160)),
        cell(1, 2, "100.000", box(400, 130, 540, 160)),
        cell(2, 0, "Total", box(60, 160, 300, 190)),
        cell(2, 2, "200.000", box(400, 160, 540, 190)),
    ]
    return item(cells, 3, 3, table_box)


def test_normalize_box_handles_both_origins():
    top_left = normalize_box(box(60, 100, 540, 300), W, H)
    bottom_left = normalize_box(box(60, 700, 540, 500, Origin.BOTTOMLEFT), W, H)
    assert top_left == bottom_left == (0.1, 0.125, 0.9, 0.375)
    assert normalize_box(None, W, H) is None
    assert normalize_box(box(10, 10, 10, 20), W, H) is None  # zero width


def test_adapter_builds_internal_table_with_header_and_summary_split():
    table = docling_table_to_internal(make_item(), page_width=W, page_height=H, page_number=2, document_id="D", ordinal=1)
    assert table is not None
    assert (table.table_id, table.page_start, table.source, table.layout_kind) == ("D-p2-t1", 2, "docling", "grid")
    assert [c.header_text for c in table.columns] == ["Deskripsi", "Qty", "Jumlah"]
    assert table.rows[0].cells[0].text == "Pengiriman Jakarta"  # whitespace collapsed
    assert table.bbox == (0.1, 0.125, 0.9, 0.375)
    assert all(0 <= v <= 1 for row in table.rows for c in row.cells for v in c.bbox)
    assert round(table.columns[1].x1, 3) == 0.5


def test_adapter_rejects_single_column_blocks_and_empty_tables():
    one_col = item([cell(0, 0, "Invoice Number : 1", box(0, 0, 50, 10), True), cell(1, 0, "Tanggal", box(0, 10, 50, 20))], 2, 1, box(0, 0, 50, 20))
    assert docling_table_to_internal(one_col, page_width=W, page_height=H, page_number=1, document_id="D", ordinal=1) is None
    assert docling_table_to_internal(item([], 0, 0, None), page_width=W, page_height=H, page_number=1, document_id="D", ordinal=1) is None


def test_adapter_headerless_table_and_missing_cell_boxes():
    cells = [cell(0, 0, "Barang A", None), cell(0, 1, "3", None), cell(1, 0, "Barang B", None), cell(1, 1, "4", None)]
    table = docling_table_to_internal(item(cells, 2, 2, box(60, 100, 540, 200)), page_width=W, page_height=H, page_number=1, document_id="D", ordinal=1)
    assert table is not None and [c.header_text for c in table.columns] == ["", ""]
    assert len(table.rows) == 2
    assert table.rows[1].cells[1].bbox[1] >= table.rows[0].cells[1].bbox[1]  # derived row geometry keeps reading order


def tok(text, x1, y1, x2, y2, line_idx):
    return Token(text, (x1, y1, x2, y2), 0.9, line_idx, 0)


def page_with_numeric_lines(n, y0=0.5):
    lines = []
    for i in range(n):
        y = y0 + i * 0.03
        lines.append([tok(t, 0.1 + j * 0.2, y, 0.25 + j * 0.2, y + 0.02, i) for j, t in enumerate(["Item", "2", "100.000", "200.000"])])
    return Page(1, [t for line in lines for t in line], lines)


def internal_table(y1, y2):
    columns = [Column(i, 0.1 * i, 0.1 * i + 0.1, f"h{i}", [f"h{i}"]) for i in range(3)]
    rows = [TableRow(0, [Cell("x", (0.1, y1, 0.2, y2))], (0.1, y1, 0.2, y2))]
    return Table("T", 1, 1, "grid", "docling", columns, rows, (0.05, y1, 0.95, y2))


def test_missed_check_flags_uncovered_numeric_block_and_passes_when_covered():
    page = page_with_numeric_lines(4)
    suspect, why = suspect_missed_tables(page, [])
    assert suspect and why.startswith("uncovered_numeric_block")
    suspect, _ = suspect_missed_tables(page, [internal_table(0.45, 0.7)])
    assert suspect is False
    short = page_with_numeric_lines(2)  # totals-like fragments are below min_uncovered_lines
    assert suspect_missed_tables(short, [])[0] is False


class FakeProvider:
    def __init__(self, name, tables):
        self.name, self._tables, self.calls = name, tables, 0

    def tables(self, page, doc_type, document_id="document"):
        self.calls += 1
        return list(self._tables)


def good_table(source, y1=0.45, y2=0.7):
    t = docling_table_to_internal(make_item(), page_width=W, page_height=H, page_number=1, document_id="D", ordinal=1)
    t.source, t.bbox = source, (0.05, y1, 0.95, y2)
    return t



def test_router_from_config_registers_only_docling():
    from app.core.config import candidates_config_dir

    extractor = TableExtractor.from_config(candidates_config_dir(), ["docling"])
    assert list(extractor.providers) == ["docling"] and extractor.order == ["docling"]


def test_router_uses_the_docling_provider_only():
    page = page_with_numeric_lines(4)
    docling = FakeProvider("docling", [good_table("docling")])
    tables = TableExtractor([docling], docling={}).ordered(["docling"]).extract(page, "INVOICE", "D")
    assert docling.calls == 1 and [t.source for t in tables] == ["docling"]
    assert tables[0].table_id == "D-p1-t1"
    empty = FakeProvider("docling", [])
    assert TableExtractor([empty], docling={}).ordered(["docling"]).extract(page, None, "D") == []


def test_provider_routes_by_pdf_bytes():
    from app.services.candidates.tables.docling_provider import DoclingTableProvider

    provider = DoclingTableProvider()
    pdf_page = Page(1, [], [], pdf_bytes=b"%PDF-one-page")
    assert provider.route(pdf_page) == ("DIGITAL", b"%PDF-one-page", True, False)
    kind, payload, is_pdf, ocr = provider.route(Page(1, [], []))  # image upload without a readable PNG
    assert (kind, payload, is_pdf, ocr) == ("SCAN", None, False, True)
    assert provider.tables(Page(1, [], []), None) == []


def test_build_page_preserves_docling_inputs():
    from app.services.candidates.context import build_page

    page = build_page({
        "page_number": 1,
        "width": 1000,
        "height": 2000,
        "page_image": {"path": "page.png"},
        "lines": [],
        "_pdf_bytes": b"%PDF-table-input",
    })

    assert page.pdf_bytes == b"%PDF-table-input"
    assert page.image_path == "page.png"
    assert page.aspect == 0.5
