import fitz

from app.services.candidates.context import build_page
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider, _norm_bbox


def _grid(page, rect, header, row):
    shape = page.new_shape()
    x1, y1, x2, y2 = rect
    shape.draw_rect(fitz.Rect(rect))
    shape.draw_line((x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2))
    middle = (x1 + x2) / 2
    shape.draw_line((middle, y1), (middle, y2))
    shape.finish(color=(0, 0, 0)); shape.commit()
    page.insert_text((x1 + 8, y1 + 25), header[0]); page.insert_text((middle + 8, y1 + 25), header[1])
    page.insert_text((x1 + 8, y1 + 75), row[0]); page.insert_text((middle + 8, y1 + 75), row[1])


def test_pymupdf_provider_normalizes_multiple_digital_tables():
    document = fitz.open(); source = document.new_page(width=400, height=500)
    _grid(source, (30, 30, 250, 130), ("Item", "Qty"), ("Widget", "2"))
    _grid(source, (30, 200, 250, 300), ("Rate", "Total"), ("1000", "2000"))
    page = build_page({"page_number": 1, "lines": [], "_source_page": source})
    tables = PyMuPDFTableProvider().tables(page, "INVOICE", "D1")
    assert len(tables) == 2
    assert tables[0].source == "pymupdf_find_tables"
    assert tables[0].rows[0].cells[0].text == "Widget"
    assert all(0 <= value <= 1 for table in tables for value in table.bbox)


def test_pymupdf_provider_falls_back_cleanly_without_source_page():
    page = build_page({"page_number": 1, "lines": []})
    assert PyMuPDFTableProvider().tables(page, "INVOICE") == []


def test_pymupdf_provider_clips_slightly_out_of_page_geometry():
    assert _norm_bbox(10, -4, 50, 110, 100, 100) == (0.1, 0.0, 0.5, 1.0)
