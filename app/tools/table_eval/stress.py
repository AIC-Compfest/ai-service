"""Randomised, seeded PDF pages with several tables and EXACT gold, to stress table extraction on varied layouts.

Each page mixes (all chosen from a seed): 1 to 4 tables stacked with captions/notes between, two tables side by side,
ruled / unruled / mixed ruling, 3 to 8 columns of different kinds (index, description, quantity, unit, weight, price,
discount, amount, date, reference, destination), Indonesian or English headers, two-level (grouped) headers, wrapped
header cells, wrapped description cells, header-less tables, summary rows inside or a separate summary block, key-value
info blocks and footer notes.  Everything is drawn with PyMuPDF so the gold is exact (``constructed_exact``).

This is VARIETY BY CONSTRUCTION, not a sample of real documents: it shows which layout features break an extractor, not
how often they occur in production.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .synthetic import PAGE_H, PAGE_W, Canvas, Fixture, TableSpec, text_width

KINDS = {
    "index": (["No", "No."], ["No"]),
    "desc": (["Deskripsi", "Keterangan", "Nama Barang", "Uraian", "Layanan"], ["Description", "Item", "Service"]),
    "qty": (["Qty", "Jumlah", "Kuantitas", "Koli"], ["Qty", "Quantity", "Pcs"]),
    "unit": (["Satuan", "Unit"], ["UOM", "Unit"]),
    "weight": (["Berat (Kg)", "Berat", "Berat Total"], ["Weight", "Weight (Kg)"]),
    "price": (["Harga Satuan", "Tarif", "Harga"], ["Unit Price", "Rate", "Price"]),
    "discount": (["Diskon", "Potongan"], ["Discount", "Disc"]),
    "amount": (["Jumlah", "Total", "Jumlah Harga"], ["Amount", "Total", "Line Total"]),
    "date": (["Tanggal", "Tgl Kirim"], ["Date", "Ship Date"]),
    "ref": (["No. Resi", "No. Referensi"], ["AWB", "Reference"]),
    "dest": (["Tujuan", "Asal"], ["Destination", "Origin"]),
}
OPTIONAL = ["qty", "unit", "weight", "price", "discount", "date", "ref", "dest"]
DESCRIPTIONS = [
    "Pengiriman Jakarta - Surabaya", "Biaya Penanganan Khusus", "Asuransi Muatan", "Pengemasan Kayu", "Bongkar Muat",
    "Sewa Gudang Bulanan", "Kardus Besar 60x40", "Lakban Bening", "Pupuk Organik 50 Kg", "Benih Padi Unggul", "Layanan Kurir Same Day",
    "Freight Charge Air", "Fuel Surcharge", "Customs Clearance", "Detention Fee", "Palet Kayu Standar",
]
CITIES = ["Jakarta", "Surabaya", "Medan", "Makassar", "Bandung", "Semarang", "Denpasar", "Balikpapan", "Palembang", "Yogyakarta"]
GROUP_LABELS = (["Tarif", "Biaya", "Rincian"], ["Rate", "Charges", "Details"])
CAPTIONS = (["Rincian Biaya", "Biaya Tambahan", "Daftar Barang", "Ringkasan Pengiriman"], ["Charges", "Additional Fees", "Shipment List"])
INFO_ID = ["Tanggal : 12 Maret 2026", "Jatuh Tempo : 11 April 2026", "No. PO : PO/2026/117", "Pelanggan : PT Sinar Makmur", "Alamat : Jl. Gajah Mada No 27"]
INFO_EN = ["Date : 12 March 2026", "Due Date : 11 April 2026", "PO No. : PO/2026/117", "Customer : Sinar Makmur Ltd", "Address : 27 Gajah Mada St"]
NOTES = [["Catatan:", "1. Pembayaran paling lambat 14 hari setelah invoice diterima.", "2. Transfer ke BCA 066.1234.567 a.n. PT Sinar Makmur"], ["Notes:", "Payment due within 14 days of invoice date.", "Bank transfer to account 066.1234.567"]]


@dataclass
class StressPage:
    fixture: Fixture
    tags: list[str]


def _money(rng: random.Random, lo: int, hi: int, style: str) -> str:
    value = rng.randrange(lo // 1000, hi // 1000) * 1000
    return f"{value:,}".replace(",", ".") if style == "id" else f"{value:,}.00"


def _cell(kind: str, i: int, rng: random.Random, style: str, wrap: bool) -> str:
    if kind == "index":
        return str(i + 1)
    if kind == "desc":
        text = rng.choice(DESCRIPTIONS)
        if wrap and " " in text and rng.random() < 0.6:
            words = text.split(" ")
            cut = max(1, len(words) // 2)
            text = " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
        return text
    if kind == "qty":
        return str(rng.randrange(1, 40))
    if kind == "unit":
        return rng.choice(["pcs", "box", "kg", "koli", "roll"])
    if kind == "weight":
        return f"{rng.randrange(5, 900) / 10:.1f}"
    if kind == "price":
        return _money(rng, 25_000, 4_000_000, style)
    if kind == "discount":
        return rng.choice(["0", "5%", "10%", _money(rng, 10_000, 200_000, style)])
    if kind == "amount":
        return _money(rng, 100_000, 90_000_000, style)
    if kind == "date":
        return f"{rng.randrange(1, 28):02d}/{rng.randrange(1, 12):02d}/2026"
    if kind == "ref":
        return f"{rng.choice(['AWB', 'RS', 'INV'])}{rng.randrange(1_000_000, 9_999_999)}"
    return rng.choice(CITIES)


def _make_table(rng: random.Random, x: float, y: float, max_width: float, max_height: float, ruled: bool, font: float, lang: int, style: str, small: bool = False):
    """Return ``(spec, tags)`` or ``None`` when nothing fits."""
    kinds = ["desc"] if rng.random() < 0.9 else ["dest"]
    optional = rng.sample(OPTIONAL, rng.randrange(1, 4 if small else 6))
    kinds += [k for k in optional if k not in kinds]
    if "index" not in kinds and rng.random() < (0.3 if small else 0.6):
        kinds.insert(0, "index")
    if rng.random() < 0.85:
        kinds.append("amount")
    order = {"index": 0, "date": 1, "ref": 1, "desc": 2, "dest": 2, "unit": 3, "weight": 3, "qty": 4, "price": 5, "discount": 6, "amount": 7}
    kinds.sort(key=lambda k: order[k] + rng.random() * 0.6)
    tags: list[str] = []
    wrap_header = rng.random() < 0.25
    wrap_rows = rng.random() < 0.3
    n_rows = rng.randrange(2, 5 if small else 9)
    headerless = rng.random() < 0.06
    header = []
    for kind in kinds:
        name = rng.choice(KINDS[kind][lang])
        if wrap_header and " " in name:
            name = name.replace(" ", "\n", 1)
        header.append(name)
    rows = [[_cell(kind, i, rng, style, wrap_rows) for kind in kinds] for i in range(n_rows)]

    def natural(cols: list[int]) -> list[float]:
        widths = []
        for c in cols:
            samples = [part for cell in [header[c]] + [r[c] for r in rows] for part in cell.split("\n")]
            widths.append(max(text_width(p, font, True) for p in samples) + 14)
        return widths

    cols = list(range(len(kinds)))
    widths = natural(cols)
    while sum(widths) > max_width and len(cols) > 3:
        drop = next((c for c in reversed(cols) if kinds[c] in OPTIONAL), cols[-1])
        cols.remove(drop)
        widths = natural(cols)
    if sum(widths) > max_width:
        return None
    kinds = [kinds[c] for c in cols]
    header = [header[c] for c in cols]
    rows = [[r[c] for c in cols] for r in rows]
    extra = max(0.0, min(max_width - sum(widths), rng.uniform(0, 90)))
    desc_cols = [i for i, k in enumerate(kinds) if k in ("desc", "dest")] or [0]
    for i in desc_cols:
        widths[i] += extra / len(desc_cols)
    aligns = ["c" if k in ("index", "qty", "unit", "date") else "r" if k in ("price", "amount", "weight", "discount") else "l" for k in kinds]
    summary = []
    if len(kinds) >= 3 and kinds[-1] in ("amount", "price") and rng.random() < 0.6:
        for label in rng.sample(["Subtotal", "PPN 11%", "Total", "Diskon", "Biaya Kirim"], rng.randrange(1, 4)):
            row = [""] * len(kinds)
            row[-2] = label
            row[-1] = _money(rng, 100_000, 90_000_000, style)
            summary.append(row)
        tags.append("summary_rows")
    groups = []
    if len(kinds) >= 4 and rng.random() < 0.15:
        c0 = rng.randrange(1, len(kinds) - 1)
        groups = [(rng.choice(GROUP_LABELS[lang]), c0, c0 + 1)]
        tags.append("grouped_header")
    if wrap_header and any("\n" in h for h in header):
        tags.append("wrapped_header")
    if wrap_rows and any("\n" in cell for r in rows for cell in r):
        tags.append("wrapped_rows")
    lh = font + 2
    height = (max(h.count("\n") + 1 for h in header) * lh + 6) + sum((max(c.count("\n") + 1 for c in r) * lh + 6) for r in rows + summary) + (lh + 6 if groups else 0)
    while height > max_height and len(rows) > 2:
        rows.pop()
        height = (max(h.count("\n") + 1 for h in header) * lh + 6) + sum((max(c.count("\n") + 1 for c in r) * lh + 6) for r in rows + summary) + (lh + 6 if groups else 0)
    if height > max_height:
        return None
    if headerless:
        tags.append("headerless")
    spec = TableSpec(x, y, widths, None if headerless else header, rows, summary, aligns=aligns, ruled=ruled, font_size=font, line_height=lh, header_groups=[] if headerless else groups, draw_header=not headerless)
    if headerless and "grouped_header" in tags:
        tags.remove("grouped_header")
    if headerless and "wrapped_header" in tags:
        tags.remove("wrapped_header")
    return spec, tags, height


def build_stress_page(seed: int) -> StressPage:
    rng = random.Random(1000 + seed)
    lang = 0 if rng.random() < 0.65 else 1
    style = "id" if lang == 0 or rng.random() < 0.3 else "en"
    font = rng.choice([8.0, 8.5, 9.0, 9.5, 10.0])
    ruling = rng.choice(["ruled", "unruled", "unruled", "mixed", "ruled"])
    c = Canvas(f"stress_{seed:04d}")
    c.new_page()
    tags = [ruling]
    y = c.text(50, 60, rng.choice(["INVOICE", "SURAT JALAN", "RATE CARD", "BUKTI TERIMA", "PACKING LIST", "LAPORAN PENGIRIMAN"]), 14, True) + 14
    if rng.random() < 0.6:
        info = INFO_ID if lang == 0 else INFO_EN
        y = c.paragraph(50, y + 6, rng.sample(info, rng.randrange(2, 5)), 9) + 8
        tags.append("info_block")

    placed = 0
    if rng.random() < 0.13:  # two tables side by side
        tags.append("side_by_side")
        left = _make_table(rng, 36, y + 10, 255, PAGE_H - y - 120, ruling == "ruled", font, lang, style, small=True)
        right = _make_table(rng, 312, y + 10, 250, PAGE_H - y - 120, ruling == "ruled", font, lang, style, small=True)
        bottoms = [y]
        for made in (left, right):
            if made:
                bottoms.append(c.table(made[0]))
                tags += made[1]
                placed += 1
        y = max(bottoms)  # whatever follows must clear BOTH tables
    else:
        wanted = rng.choices([1, 2, 3, 4], weights=[4, 3, 2, 1])[0]
        for index in range(wanted):
            room = PAGE_H - y - 150
            if room < 80:
                break
            ruled = ruling == "ruled" or (ruling == "mixed" and rng.random() < 0.5)
            if index and rng.random() < 0.7:
                y = c.text(50, y + 26, rng.choice(CAPTIONS[lang]), 11, True) + 6
            elif index:
                y += rng.uniform(22, 48)
            made = _make_table(rng, rng.choice([40, 50, 60]), y + 8, PAGE_W - 100, min(room, 330 if wanted > 2 else 520), ruled, font, lang, style)
            if made is None:
                continue
            spec, extra, _ = made
            y = c.table(spec)
            tags += extra
            placed += 1
            if index < wanted - 1 and rng.random() < 0.3:
                y = c.paragraph(50, y + 14, rng.choice(NOTES)[:2], 8.5)
        if placed and rng.random() < 0.15 and y < PAGE_H - 150:  # separate summary block bottom-right
            c.table(TableSpec(335, y + 18, [80, 130], None, [["Subtotal", "5.050.000"], ["PPN 11%", "555.500"], ["Total", "5.605.500"]], aligns=["l", "r"], ruled=False, role="summary", draw_header=False))
            tags.append("summary_block")
    if rng.random() < 0.5 and y < PAGE_H - 140:
        c.paragraph(50, y + 40, rng.choice(NOTES), 8.5)
    n_line = sum(1 for t in c.pages[0]["tables"] if t["role"] == "line_items")
    tags.append("tables=1" if n_line <= 1 else "tables=2" if n_line == 2 else "tables=3+")
    return StressPage(c.finish(f"stress seed {seed}"), sorted(set(tags)))


def build_stress(count: int, first_seed: int = 0) -> list[StressPage]:
    pages = []
    for seed in range(first_seed, first_seed + count):
        page = build_stress_page(seed)
        if page.fixture.gold["pages"][0]["tables"]:
            pages.append(page)
    return pages
