"""Deterministic constructed table fixtures with EXACT gold (no human review, no OCR).

Pages are drawn with PyMuPDF so that geometry is known by construction. Every fixture returns a PDF
(bytes) plus a GOLD_REVIEW-shaped gold dict (``source_label="constructed_exact"``, ``reviewed=false`` because
no human reviewed it; evaluate with ``--allow-unreviewed``).

``to_evidence_page`` converts a drawn page into the OCR evidence line format, either with exact word
tokens (``vector``) or with phrase-level boxes, estimated tokens, jitter and a small skew that imitate a
raster OCR engine (``phrase``). Both are deterministic for a given seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import pymupdf

PAGE_W, PAGE_H = 595.0, 842.0
FONT, FONT_BOLD = "helv", "hebo"


def text_width(text: str, size: float, bold: bool = False) -> float:
    return pymupdf.get_text_length(text, fontname=FONT_BOLD if bold else FONT, fontsize=size)


@dataclass
class TableSpec:
    x: float
    y: float
    col_widths: list[float]
    header: list[str] | None
    rows: list[list[str]]
    summary: list[list[str]] = field(default_factory=list)
    aligns: list[str] | None = None  # per column: "l", "r", "c"
    ruled: bool = True
    font_size: float = 9.0
    line_height: float = 11.0
    role: str = "line_items"
    header_groups: list[tuple[str, int, int]] = field(default_factory=list)  # (label, first_col, last_col)
    logical_id: str | None = None
    draw_header: bool = True
    table_id: str = ""


@dataclass
class Fixture:
    name: str
    pdf: bytes
    gold: dict
    description: str = ""

    @property
    def page_count(self) -> int:
        return len(self.gold["pages"])


def _norm(box: tuple[float, float, float, float]) -> list[float]:
    return [round(box[0] / PAGE_W, 6), round(box[1] / PAGE_H, 6), round(box[2] / PAGE_W, 6), round(box[3] / PAGE_H, 6)]


def _join(text: str) -> str:
    return " ".join(part.strip() for part in text.split("\n") if part.strip())


class Canvas:
    """One PDF document under construction plus the matching gold."""

    def __init__(self, name: str):
        self.name = name
        self.doc = pymupdf.open()
        self.pages: list[dict] = []
        self._table_seq = 0

    def new_page(self) -> pymupdf.Page:
        page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.pages.append({"page_number": len(self.pages) + 1, "tables": []})
        return page

    @property
    def page(self) -> pymupdf.Page:
        return self.doc[len(self.doc) - 1]

    def text(self, x: float, y: float, text: str, size: float = 10.0, bold: bool = False) -> float:
        """Draw a free text line (baseline at ``y``); returns the bottom y."""
        self.page.insert_text((x, y), text, fontsize=size, fontname=FONT_BOLD if bold else FONT)
        return y + size * 1.4

    def paragraph(self, x: float, y: float, lines: list[str], size: float = 9.0) -> float:
        for line in lines:
            y = self.text(x, y, line, size)
        return y

    def table(self, spec: TableSpec) -> float:
        """Draw a table and register its gold. Returns the y coordinate below the last drawn row."""
        self._table_seq += 1
        page = self.page
        fs, lh = spec.font_size, spec.line_height
        aligns = spec.aligns or ["l"] * len(spec.col_widths)
        xs = [spec.x]
        for width in spec.col_widths:
            xs.append(xs[-1] + width)
        y = spec.y

        def draw_row(cells: list[str], top: float, bold: bool) -> float:
            lines = [cell.split("\n") for cell in cells]
            height = max(len(parts) for parts in lines) * lh + 6
            for c, parts in enumerate(lines):
                if spec.ruled:
                    page.draw_rect(pymupdf.Rect(xs[c], top, xs[c + 1], top + height), color=(0, 0, 0), width=0.6)
                for i, part in enumerate(parts):
                    if not part:
                        continue
                    w = text_width(part, fs, bold)
                    if aligns[c] == "r":
                        tx = xs[c + 1] - 4 - w
                    elif aligns[c] == "c":
                        tx = xs[c] + (spec.col_widths[c] - w) / 2
                    else:
                        tx = xs[c] + 4
                    page.insert_text((tx, top + 3 + fs + i * lh), part, fontsize=fs, fontname=FONT_BOLD if bold else FONT)
            return height

        header_cells: list[dict] = []
        table_top = y
        if spec.header is not None and spec.draw_header:
            if spec.header_groups:
                group_h = lh + 6
                for label, c0, c1 in spec.header_groups:
                    gw = xs[c1 + 1] - xs[c0]
                    if spec.ruled:
                        page.draw_rect(pymupdf.Rect(xs[c0], y, xs[c1 + 1], y + group_h), color=(0, 0, 0), width=0.6)
                    page.insert_text((xs[c0] + (gw - text_width(label, fs, True)) / 2, y + 3 + fs), label, fontsize=fs, fontname=FONT_BOLD)
                y += group_h
            header_top = y
            h = draw_row(spec.header, y, True)
            for c, cell in enumerate(spec.header):
                text = _join(cell)
                for label, c0, c1 in spec.header_groups:
                    if c0 <= c <= c1:
                        text = f"{label} {text}"
                header_cells.append({"column_index": c, "raw_text": text, "normalized_text": text, "bbox_norm": _norm((xs[c], header_top, xs[c + 1], header_top + h)), "legibility": "clear"})
            y += h
        rows_gold, last_item_bottom = [], y
        for r, row in enumerate(spec.rows):
            h = draw_row(row, y, False)
            cells = [{"column_index": c, "raw_text": _join(cell), "normalized_text": _join(cell), "bbox_norm": _norm((xs[c], y, xs[c + 1], y + h)), "legibility": "clear"} for c, cell in enumerate(row)]
            rows_gold.append({"row_id": f"r{r + 1}", "kind": "item", "cells": cells})
            y += h
            last_item_bottom = y
        summary_gold = []
        for r, row in enumerate(spec.summary):
            h = draw_row(row, y, True)
            cells = [{"column_index": c, "raw_text": _join(cell), "normalized_text": _join(cell), "bbox_norm": _norm((xs[c], y, xs[c + 1], y + h)), "legibility": "clear"} for c, cell in enumerate(row)]
            summary_gold.append({"row_id": f"s{r + 1}", "kind": "summary", "cells": cells})
            y += h
        page_index = len(self.pages) - 1
        table_id = spec.table_id or f"{self.name}-p{page_index + 1}-t{len(self.pages[page_index]['tables']) + 1}"
        self.pages[page_index]["tables"].append(
            {
                "table_id": table_id,
                "role": spec.role,
                "bbox_norm": _norm((xs[0], table_top, xs[-1], last_item_bottom)),
                "headers": header_cells,
                "rows": rows_gold,
                "summary_rows": summary_gold,
                "logical_id": spec.logical_id,
                "quality": {"header_found": bool(header_cells)},
            }
        )
        return y

    def finish(self, description: str = "") -> Fixture:
        gold = {
            "document_id": f"constructed/{self.name}",
            "source": {"relative_path": f"constructed/{self.name}.pdf", "sha256": "", "source_kind": "pdf"},
            "document_type": "CONSTRUCTED",
            "source_label": "constructed_exact",
            "reviewed": False,
            "review": {"schema_version": "1", "prepared_by": "table_eval.synthetic", "prepared_at_utc": "", "approved_by": "", "approved_at_utc": "", "notes": description},
            "pages_exhaustive": True,
            "pages": self.pages,
        }
        return Fixture(self.name, self.doc.tobytes(), gold, description)


# ---------------------------------------------------------------- fixture builders

ITEMS = [
    ["1", "Pengiriman Jakarta - Surabaya", "2", "1.250.000", "2.500.000"],
    ["2", "Pengiriman Bandung - Medan", "1", "3.400.000", "3.400.000"],
    ["3", "Biaya Penanganan Khusus", "4", "150.000", "600.000"],
    ["4", "Asuransi Muatan", "1", "275.000", "275.000"],
    ["5", "Pengemasan Kayu", "3", "420.000", "1.260.000"],
    ["6", "Bongkar Muat", "2", "310.000", "620.000"],
]
HEADER = ["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"]
WIDTHS = [30, 210, 50, 110, 110]
ALIGNS = ["c", "l", "c", "r", "r"]
SUMMARY = [["", "", "", "Subtotal", "8.755.000"], ["", "", "", "PPN 11%", "963.050"], ["", "", "", "Total", "9.718.050"]]


def _title(c: Canvas, title: str) -> float:
    return c.text(50, 60, title, 14, True)


def _fx_single_ruled() -> Fixture:
    c = Canvas("single_ruled")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, ITEMS, aligns=ALIGNS))
    return c.finish("One ruled table.")


def _fx_single_unruled() -> Fixture:
    c = Canvas("single_unruled")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, ITEMS, aligns=ALIGNS, ruled=False))
    return c.finish("One table without ruling lines.")


def _fx_items_with_summary_rows_empty_first_col() -> Fixture:
    c = Canvas("items_total_rows_inside")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, ITEMS, SUMMARY, aligns=ALIGNS))
    return c.finish("Item rows followed by Subtotal/PPN/Total rows whose first column is empty.")


def _fx_item_named_total() -> Fixture:
    c = Canvas("item_named_total")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    rows = [r[:] for r in ITEMS[:4]]
    rows[1][1] = "Total Logistic Service Surabaya"
    rows[2][1] = "Jumlah Muatan Tambahan"
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, rows, SUMMARY[2:], aligns=ALIGNS, ruled=False))
    return c.finish("Item descriptions starting with 'Total' / 'Jumlah' must stay items.")


def _fx_stacked_two_tables() -> Fixture:
    c = Canvas("stacked_two_tables")
    c.new_page()
    y = _title(c, "RINCIAN BIAYA") + 20
    y = c.table(TableSpec(50, y + 10, WIDTHS, HEADER, ITEMS[:3], aligns=ALIGNS, ruled=False))
    y = c.text(50, y + 40, "Rincian Biaya Tambahan", 11, True) + 6
    c.table(TableSpec(50, y + 4, [30, 210, 50, 110, 110], ["No", "Keterangan", "Qty", "Tarif", "Jumlah"], ITEMS[3:], aligns=ALIGNS, ruled=False))
    return c.finish("Two independent item tables stacked vertically with a heading between them.")


def _fx_side_by_side() -> Fixture:
    c = Canvas("side_by_side")
    c.new_page()
    _title(c, "PERBANDINGAN TARIF")
    left = TableSpec(40, 110, [24, 100, 56, 70], ["No", "Tujuan", "Qty", "Tarif"], [["1", "Surabaya", "2", "1.250.000"], ["2", "Medan", "1", "3.400.000"], ["3", "Makassar", "4", "2.150.000"]], aligns=["c", "l", "c", "r"], ruled=False)
    right = TableSpec(330, 110, [24, 100, 56, 70], ["No", "Asal", "Qty", "Tarif"], [["1", "Bandung", "3", "950.000"], ["2", "Semarang", "2", "1.100.000"], ["3", "Denpasar", "5", "2.800.000"]], aligns=["c", "l", "c", "r"], ruled=False)
    c.table(left)
    c.table(right)
    return c.finish("Two tables at the same height, left and right.")


def _fx_items_plus_summary_table() -> Fixture:
    c = Canvas("items_plus_summary_table")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    y = c.table(TableSpec(50, y + 20, WIDTHS, HEADER, ITEMS[:4], aligns=ALIGNS, ruled=False))
    c.table(TableSpec(335, y + 24, [80, 130], None, [["Subtotal", "5.050.000"], ["PPN 11%", "555.500"], ["Total", "5.605.500"]], aligns=["l", "r"], ruled=False, role="summary", draw_header=False))
    return c.finish("Item table with a separate summary block bottom-right (role summary).")


def _fx_info_block_beside() -> Fixture:
    c = Canvas("info_block_beside")
    c.new_page()
    y = _title(c, "INVOICE") + 14
    c.paragraph(380, y + 20, ["Tanggal : 12 Maret 2026", "Jatuh Tempo : 11 April 2026", "No. PO : PO/2026/117", "Pelanggan : PT Sinar Makmur"], 9)
    c.table(TableSpec(50, y + 20, [24, 120, 36, 70, 70], ["No", "Deskripsi", "Qty", "Harga", "Jumlah"], [["1", "Layanan Kurir", "2", "250.000", "500.000"], ["2", "Pengemasan", "1", "150.000", "150.000"], ["3", "Asuransi", "1", "75.000", "75.000"]], aligns=["c", "l", "c", "r", "r"], ruled=False))
    return c.finish("Key-value block beside the table (must not become a table).")


def _fx_header_two_line() -> Fixture:
    c = Canvas("header_two_line")
    c.new_page()
    y = _title(c, "SURAT JALAN") + 20
    c.table(TableSpec(50, y + 20, [30, 180, 60, 80, 110], ["No", "Nama\nBarang", "Berat\n(Kg)", "Jumlah\nKoli", "Keterangan"], [["1", "Pupuk Organik", "50", "200", "Siap kirim"], ["2", "Pupuk Kompos", "25", "350", "-"], ["3", "Benih Padi", "5", "120", "Siap tanam"]], aligns=["c", "l", "r", "r", "l"], ruled=False))
    return c.finish("Header cells wrapped on two lines.")


def _fx_header_grouped() -> Fixture:
    c = Canvas("header_grouped")
    c.new_page()
    y = _title(c, "RATE CARD") + 20
    c.table(TableSpec(50, y + 20, [30, 150, 80, 80, 100], ["No", "Tujuan", "Per Kg", "Minimum", "Transit"], [["1", "Surabaya", "3.500", "250.000", "2 hari"], ["2", "Medan", "9.800", "500.000", "4 hari"], ["3", "Makassar", "7.200", "400.000", "3 hari"]], aligns=["c", "l", "r", "r", "c"], header_groups=[("Tarif", 2, 3)]))
    return c.finish("Grouped header: 'Tarif' spans 'Per Kg' and 'Minimum'.")


def _fx_wrapped_cells() -> Fixture:
    c = Canvas("wrapped_cells")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    rows = [
        ["1", "Pengiriman Jakarta - Surabaya\nmelalui jalur darat via Semarang", "2", "1.250.000", "2.500.000"],
        ["2", "Pengiriman Bandung - Medan", "1", "3.400.000", "3.400.000"],
        ["3", "Biaya Penanganan Khusus\nbarang pecah belah", "4", "150.000", "600.000"],
    ]
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, rows, aligns=ALIGNS, ruled=False))
    return c.finish("Description cells wrapped on two lines (continuation rows).")


def _fx_blank_column() -> Fixture:
    c = Canvas("blank_column")
    c.new_page()
    y = _title(c, "DAFTAR BARANG") + 20
    c.table(TableSpec(50, y + 20, [30, 190, 70, 70, 120], ["No", "Nama Barang", "Satuan", "Qty", "Keterangan"], [["1", "Kardus Besar", "pcs", "200", ""], ["2", "Lakban", "roll", "50", "Warna bening"], ["3", "Plastik Wrap", "roll", "30", ""], ["4", "Lem Kayu", "kaleng", "12", ""]], aligns=["c", "l", "c", "r", "l"], ruled=False))
    return c.finish("Table with empty cells and a mostly empty last column.")


def _fx_notes_footer_below() -> Fixture:
    c = Canvas("notes_footer_below")
    c.new_page()
    y = _title(c, "INVOICE") + 20
    y = c.table(TableSpec(50, y + 20, WIDTHS, HEADER, ITEMS[:4], SUMMARY[2:], aligns=ALIGNS, ruled=False))
    y = c.paragraph(50, y + 22, ["Catatan:", "1. Pembayaran paling lambat 14 hari setelah invoice diterima.", "2. Transfer ke BCA 066.1234.567 a.n. PT Sinar Makmur", "3. Barang yang sudah dikirim tidak dapat ditukar."], 9)
    c.text(400, y + 50, "Hormat kami,", 9)
    c.text(400, y + 110, "(Andi Prasetyo)", 9)
    return c.finish("Notes, bank info and signature under the table (must not become rows).")


def _fx_paragraph_only() -> Fixture:
    c = Canvas("paragraph_only")
    c.new_page()
    y = _title(c, "SYARAT DAN KETENTUAN") + 20
    lines = [
        "Pengirim menyatakan bahwa barang yang dikirim bukan barang terlarang menurut peraturan yang berlaku.",
        "Perusahaan tidak bertanggung jawab atas kerusakan akibat kemasan yang tidak memadai atau force majeure.",
        "Klaim kehilangan atau kerusakan wajib diajukan secara tertulis paling lambat 3 hari setelah barang diterima.",
        "Nilai ganti rugi maksimal sebesar 10 kali biaya pengiriman kecuali barang diasuransikan.",
        "Segala perselisihan akan diselesaikan secara musyawarah, dan bila tidak tercapai melalui pengadilan negeri.",
    ] * 3
    c.paragraph(50, y + 20, lines, 9)
    return c.finish("A page of paragraphs. No table exists; any predicted table is a false positive.")


def _fx_multipage_repeated_header() -> Fixture:
    c = Canvas("multipage_repeated_header")
    rows = [[str(i), f"Pengiriman rute {i}", str(i % 4 + 1), "100.000", f"{(i % 4 + 1) * 100000:,}".replace(",", ".")] for i in range(1, 25)]
    c.new_page()
    y = _title(c, "INVOICE") + 20
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, rows[:12], aligns=ALIGNS, ruled=False, logical_id="mp1"))
    c.new_page()
    c.table(TableSpec(50, 80, WIDTHS, HEADER, rows[12:], SUMMARY[2:], aligns=ALIGNS, ruled=False, logical_id="mp1"))
    return c.finish("One logical table over two pages with the header repeated.")


def _fx_multipage_no_header() -> Fixture:
    c = Canvas("multipage_no_header")
    rows = [[str(i), f"Pengiriman rute {i}", str(i % 4 + 1), "100.000", f"{(i % 4 + 1) * 100000:,}".replace(",", ".")] for i in range(1, 25)]
    c.new_page()
    y = _title(c, "INVOICE") + 20
    c.table(TableSpec(50, y + 20, WIDTHS, HEADER, rows[:12], aligns=ALIGNS, ruled=False, logical_id="mp2"))
    c.new_page()
    c.table(TableSpec(50, 80, WIDTHS, None, rows[12:], SUMMARY[2:], aligns=ALIGNS, ruled=False, logical_id="mp2", draw_header=False))
    return c.finish("Continuation page without a header row.")


def _fx_two_tables_with_notes_between() -> Fixture:
    c = Canvas("two_tables_ruled_stacked")
    c.new_page()
    y = _title(c, "LAPORAN") + 20
    y = c.table(TableSpec(50, y + 10, [30, 160, 90, 90, 100], ["No", "Keterangan", "Qty", "Tarif", "Jumlah"], [["1", "Layanan A", "2", "100.000", "200.000"], ["2", "Layanan B", "3", "150.000", "450.000"]], aligns=["c", "l", "c", "r", "r"]))
    y = c.text(50, y + 30, "Rincian Pembayaran", 11, True)
    c.table(TableSpec(50, y + 8, [30, 160, 90, 90, 100], ["No", "Metode", "Tanggal", "Bank", "Nominal"], [["1", "Transfer", "12/03/2026", "BCA", "300.000"], ["2", "Transfer", "20/03/2026", "BNI", "350.000"]], aligns=["c", "l", "c", "c", "r"]))
    return c.finish("Two ruled tables stacked with a heading between them.")


def _fx_three_tables_mixed() -> Fixture:
    c = Canvas("three_tables_mixed")
    c.new_page()
    y = _title(c, "RATE CONFIRMATION") + 20
    c.table(TableSpec(40, y + 10, [100, 120], ["Stop", "Alamat"], [["Pickup", "Gudang A Cikarang"], ["Delivery", "DC Surabaya"]], aligns=["l", "l"], role="other"))
    left_bottom = c.table(TableSpec(40, y + 110, [150, 80, 90], ["Deskripsi", "Unit", "Biaya (USD)"], [["Net Freight", "1", "3.425,00"], ["Fuel Surcharge", "1", "210,00"], ["Detention", "2", "150,00"]], aligns=["l", "c", "r"]))
    c.table(TableSpec(40, left_bottom + 30, [150, 80, 90], ["Item", "Perhitungan", "Subtotal (USD)"], [["Freight", "1 x 3.425,00", "3.425,00"], ["Fuel", "1 x 210,00", "210,00"]], aligns=["l", "c", "r"]))
    return c.finish("Three independent tables on one page (stop list, charges, payment).")


FIXTURE_BUILDERS: dict[str, Callable[[], Fixture]] = {
    "single_ruled": _fx_single_ruled,
    "single_unruled": _fx_single_unruled,
    "items_total_rows_inside": _fx_items_with_summary_rows_empty_first_col,
    "item_named_total": _fx_item_named_total,
    "stacked_two_tables": _fx_stacked_two_tables,
    "side_by_side": _fx_side_by_side,
    "items_plus_summary_table": _fx_items_plus_summary_table,
    "info_block_beside": _fx_info_block_beside,
    "header_two_line": _fx_header_two_line,
    "header_grouped": _fx_header_grouped,
    "wrapped_cells": _fx_wrapped_cells,
    "blank_column": _fx_blank_column,
    "notes_footer_below": _fx_notes_footer_below,
    "paragraph_only": _fx_paragraph_only,
    "multipage_repeated_header": _fx_multipage_repeated_header,
    "multipage_no_header": _fx_multipage_no_header,
    "two_tables_ruled_stacked": _fx_two_tables_with_notes_between,
    "three_tables_mixed": _fx_three_tables_mixed,
}

# Fixtures that contain several tables on one page (used for the multi-table acceptance criterion).
MULTI_TABLE_FIXTURES = ("stacked_two_tables", "side_by_side", "items_plus_summary_table", "two_tables_ruled_stacked", "three_tables_mixed")


def build_fixture(name: str) -> Fixture:
    return FIXTURE_BUILDERS[name]()


def build_all() -> list[Fixture]:
    return [builder() for builder in FIXTURE_BUILDERS.values()]


# ---------------------------------------------------------------- evidence conversion

def to_evidence_page(page: pymupdf.Page, page_number: int, mode: str = "vector", seed: int = 0) -> dict:
    """Convert a drawn page into OCR-evidence lines (``bbox_norm`` in 0..1).

    ``vector``: exact word tokens grouped by (block, line) as the OCR engine does for digital PDFs.
    ``phrase``: raster-like. Words of a visual line are merged into phrase boxes (split at gaps larger than
    one em), tokens are ESTIMATED proportionally to character count, and every box is jittered in y
    (+-12% of the text height) and rotated by 0.4 degrees around the page center.
    """
    words = page.get_text("words")
    width, height = page.rect.width, page.rect.height
    rng = random.Random(seed)
    angle = math.radians(0.4)

    def norm_box(x0, y0, x1, y1, jitter=0.0):
        if mode == "phrase":
            cx, cy = width / 2, height / 2
            pts = []
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                dx, dy = px - cx, py - cy
                pts.append((cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle) + jitter))
            x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
            y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
        x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
        y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
        return [round(x0 / width, 6), round(y0 / height, 6), round(max(x1, x0 + 0.5) / width, 6), round(max(y1, y0 + 0.5) / height, 6)]

    lines: list[dict] = []
    if mode == "vector":
        grouped: dict[tuple[int, int], list] = {}
        for word in words:
            grouped.setdefault((word[5], word[6]), []).append(word)
        for key in sorted(grouped):
            items = sorted(grouped[key], key=lambda w: w[0])
            tokens = [{"text": w[4], "confidence": 1.0, "bbox_norm": norm_box(w[0], w[1], w[2], w[3]), "_estimated": False} for w in items]
            lines.append({"text": " ".join(w[4] for w in items), "confidence": 1.0, "bbox_norm": norm_box(min(w[0] for w in items), min(w[1] for w in items), max(w[2] for w in items), max(w[3] for w in items)), "tokens": tokens})
        return {"page_number": page_number, "lines": lines, "engine_used": "synthetic_vector", "width": width, "height": height}

    # phrase mode: cluster words into visual lines, then split into phrases at large gaps.
    heights = sorted(w[3] - w[1] for w in words) or [10.0]
    median_h = heights[len(heights) // 2]
    rows: list[list] = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (word[1] + word[3]) / 2
        row = next((r for r in rows if abs((r[0][1] + r[0][3]) / 2 - cy) <= median_h * 0.5), None)
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    for row in rows:
        row.sort(key=lambda w: w[0])
        phrases: list[list] = [[row[0]]]
        for word in row[1:]:
            if word[0] - phrases[-1][-1][2] > median_h * 1.0:
                phrases.append([word])
            else:
                phrases[-1].append(word)
        for phrase in phrases:
            x0, y0 = min(w[0] for w in phrase), min(w[1] for w in phrase)
            x1, y1 = max(w[2] for w in phrase), max(w[3] for w in phrase)
            jitter = rng.uniform(-0.12, 0.12) * (y1 - y0)
            text_words = [w[4] for w in phrase]
            total_chars = sum(len(t) for t in text_words) or 1
            cursor, tokens = x0, []
            for token_text in text_words:
                span = (x1 - x0) * len(token_text) / total_chars
                tokens.append({"text": token_text, "confidence": 0.9, "bbox_norm": norm_box(cursor, y0, cursor + span, y1, jitter), "_estimated": len(text_words) > 1})
                cursor += span
            lines.append({"text": " ".join(text_words), "confidence": 0.9, "bbox_norm": norm_box(x0, y0, x1, y1, jitter), "tokens": tokens})
    return {"page_number": page_number, "lines": lines, "engine_used": "synthetic_phrase", "width": width, "height": height}
