"""Items made of two text lines with the numbers centred between them (found on invoice_13) stay ONE table."""
from __future__ import annotations

import pymupdf

from app.services.candidates.context import build_page
from app.services.candidates.tables.geometry import split_blocks
from app.services.candidates.tables.token_grid import TokenGridTableProvider
from app.tools.table_eval.synthetic import to_evidence_page
from app.core.config import candidates_config_dir

ITEMS = [("Kertas A3 (BOX)", "Ukuran A3, 80 gsm", "85,000", "6", "510,000"), ("Tinta Printer (BOTOL)", "Warna Hitam", "45,000", "12", "540,000"), ("Stapler Kenko (PCS)", "Ukuran Besar", "32,000", "5", "160,000"), ("Map Plastik (PCS)", "Warna Biru", "9,000", "10", "90,000")]


def _page(spacing: float):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 359.0
    page.insert_text((50, 60), "INVOICE", fontsize=14)
    for x, text in ((50, "Keterangan"), (300, "Harga Unit"), (390, "Kuantitas"), (490, "Total")):
        page.insert_text((x, y), text, fontsize=10)
    y = 393.0
    for first, second, price, qty, total in ITEMS[:3]:
        page.insert_text((50, y), first, fontsize=10)
        page.insert_text((50, y + 15), second, fontsize=10)
        page.insert_text((300, y + 7), f"Rp {price}", fontsize=10)  # numbers centred between the two description lines
        page.insert_text((400, y + 7), qty, fontsize=10)
        page.insert_text((480, y + 7), f"Rp {total}", fontsize=10)
        y += spacing
    page.insert_text((50, y + 40), "Terima kasih atas kepercayaan Anda.", fontsize=10)
    return page


def _provider() -> TokenGridTableProvider:
    config = candidates_config_dir()
    return TokenGridTableProvider(config / "line_item_aliases.yaml", config / "abbreviations.yaml")


def test_regularly_spaced_two_line_items_form_one_table():
    page = _page(spacing=54)  # layout measured on invoice_13: lines 0, +7, +15 pt; rows 54 pt apart
    tables = _provider().tables(build_page({**to_evidence_page(page, 1, "vector"), "_source_page": page}), "INVOICE", "d")
    items = [t for t in tables if len(t.rows) >= 3]
    assert len(items) == 1
    (table,) = items
    assert [c.header_text for c in table.columns][:1] == ["Keterangan"] and len(table.rows) == 3
    assert [row.cells[-1].text for row in table.rows] == ["Rp 510,000", "Rp 540,000", "Rp 160,000"]


def test_a_single_large_gap_is_still_a_block_break():
    from app.services.candidates.tables.geometry import Line, Seg

    def line(cy):
        return Line([Seg([])], cy)

    ys = [0.10, 0.12, 0.14, 0.16, 0.45, 0.47, 0.49]  # one large gap only
    blocks = split_blocks([line(y) for y in ys], height=0.02)
    assert [len(b.lines) for b in blocks] == [4, 3]


def test_repeated_large_gaps_are_row_spacing():
    from app.services.candidates.tables.geometry import Line, Seg

    def line(cy):
        return Line([Seg([])], cy)

    ys = [0.10, 0.108, 0.116, 0.164, 0.172, 0.18, 0.228, 0.236, 0.244, 0.292, 0.30, 0.308]
    blocks = split_blocks([line(y) for y in ys], height=0.012)
    assert len(blocks) == 1
    assert len(split_blocks([line(y) for y in ys], height=0.012, options={"block_periodic_min": 0})) > 1  # switch disables it
