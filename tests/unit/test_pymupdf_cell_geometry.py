"""Regression tests for PyMuPDF cell geometry (T-11): bbox must come from ``table.rows[r].cells[c]``."""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import pymupdf
import pytest

from app.services.candidates.context import build_page
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider, _normalize
from app.tools.table_eval.synthetic import build_fixture
from tests.support.paths import DATASET_ROOT

TOLERANCE = 0.005


def _provider_tables(page: pymupdf.Page):
    return PyMuPDFTableProvider().tables(build_page({"page_number": 1, "lines": [], "_source_page": page}), "INVOICE", "d")


def _strategy_tables(page: pymupdf.Page):
    for strategy in ("lines", "text"):
        found = page.find_tables(strategy=strategy).tables
        if found:
            return found
    return []


def _count_wrong_cells(page: pymupdf.Page) -> tuple[int, int]:
    width, height = page.rect.width, page.rect.height
    found = _strategy_tables(page)
    wrong = compared = 0
    for table in _provider_tables(page):
        reference = next(
            (f for f in found if all(abs(a - b) < 0.002 for a, b in zip((f.bbox[0] / width, f.bbox[1] / height, f.bbox[2] / width, f.bbox[3] / height), table.bbox))),
            None,
        )
        if reference is None:
            continue
        start = 0 if reference.header.external else 1
        texts = reference.extract()
        # The provider omits rows whose cells are all empty; mirror that when pairing rows.
        data_rows = [i for i in range(start, len(reference.rows)) if any(texts[i])]
        assert len(data_rows) == len(table.rows)
        for r, row in enumerate(table.rows):
            for c, cell in enumerate(row.cells):
                raw = reference.rows[data_rows[r]].cells[c]
                if raw is None:
                    continue
                compared += 1
                expected = (raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height)
                wrong += any(abs(a - b) > TOLERANCE for a, b in zip(expected, cell.bbox))
    return wrong, compared


def test_cells_match_pymupdf_row_cell_grid_on_constructed_ruled_table():
    document = pymupdf.open(stream=build_fixture("single_ruled").pdf, filetype="pdf")
    wrong, compared = _count_wrong_cells(document[0])
    assert compared >= 20
    assert wrong == 0


def test_cells_match_pymupdf_on_every_digital_invoice():
    root = DATASET_ROOT / "invoice" / "pdf"
    if not root.exists():
        pytest.skip("canonical dataset not found (set VERIFLOW_DATASET_ROOT)")
    wrong = compared = 0
    for pdf in sorted(root.glob("*.pdf")):
        for page in pymupdf.open(pdf):
            w, c = _count_wrong_cells(page)
            wrong, compared = wrong + w, compared + c
    assert compared > 1000, "expected the corpus to yield >1000 comparable cells"
    assert wrong == 0, f"{wrong}/{compared} cell bboxes disagree with table.rows[r].cells[c]"


# ---- fake PyMuPDF objects: missing cells and an external header

@dataclass
class _Row:
    cells: list
    bbox: tuple


@dataclass
class _Header:
    names: list
    external: bool


@dataclass
class _Found:
    bbox: tuple
    rows: list
    header: _Header
    cells: list = field(default_factory=list)
    texts: list = field(default_factory=list)

    def extract(self):
        return self.texts


class _FakePage:
    def __init__(self, found):
        self.rect = pymupdf.Rect(0, 0, 400, 500)
        self._found = found

    def find_tables(self, **_):
        return type("T", (), {"tables": self._found})()


def _page_with(found):
    fake = _FakePage(found)
    return build_page({"page_number": 1, "lines": [], "_source_page": fake})


def test_missing_cell_does_not_shift_following_cells():
    rows = [
        _Row([(10, 10, 110, 30), (110, 10, 210, 30), (210, 10, 310, 30)], (10, 10, 310, 30)),
        _Row([(10, 30, 110, 50), None, (210, 30, 310, 50)], (10, 30, 310, 50)),
        _Row([(10, 50, 110, 70), (110, 50, 210, 70), (210, 50, 310, 70)], (10, 50, 310, 70)),
    ]
    # The flat list is deliberately incomplete and unordered; a flat index would shift every later cell.
    flat = [rows[0].cells[0], rows[0].cells[1], rows[0].cells[2], rows[1].cells[0], rows[1].cells[2], rows[2].cells[0], rows[2].cells[1], rows[2].cells[2]]
    found = _Found((10, 10, 310, 70), rows, _Header(["A", "B", "C"], False), flat, [["A", "B", "C"], ["1", "", "3"], ["4", "5", "6"]])
    page = _page_with([found])
    table = _normalize([found], page, "d")[0]
    first, second = table.rows
    assert [cell.text for cell in first.cells] == ["1", "", "3"]
    assert first.cells[2].bbox == pytest.approx((210 / 400, 30 / 500, 310 / 400, 50 / 500))
    assert second.cells[1].bbox == pytest.approx((110 / 400, 50 / 500, 210 / 400, 70 / 500))
    # The None cell falls back to the column extent over the row extent, not to a neighbour's box.
    assert first.cells[1].bbox == pytest.approx((110 / 400, 30 / 500, 210 / 400, 50 / 500))


def test_external_header_keeps_all_rows_as_data_and_never_drops_a_data_row():
    rows = [
        _Row([(10, 40, 110, 60), (110, 40, 210, 60)], (10, 40, 210, 60)),
        _Row([(10, 60, 110, 80), (110, 60, 210, 80)], (10, 60, 210, 80)),
    ]
    found = _Found((10, 40, 210, 80), rows, _Header(["Item", "Qty"], True), [], [["Widget", "2"], ["Gadget", "3"]])
    table = _normalize([found], _page_with([found]), "d")[0]
    assert [c.header_text for c in table.columns] == ["Item", "Qty"]
    assert [[c.text for c in r.cells] for r in table.rows] == [["Widget", "2"], ["Gadget", "3"]]


def test_external_header_with_wrong_length_is_not_attached_to_columns():
    rows = [_Row([(10, 40, 110, 60), (110, 40, 210, 60)], (10, 40, 210, 60)), _Row([(10, 60, 110, 80), (110, 60, 210, 80)], (10, 60, 210, 80))]
    found = _Found((10, 40, 210, 80), rows, _Header(["One long title"], True), [], [["a", "1"], ["b", "2"]])
    table = _normalize([found], _page_with([found]), "d")[0]
    assert [c.header_text for c in table.columns] == ["", ""]
    assert len(table.rows) == 2


def test_internal_header_is_taken_from_first_row_even_when_names_were_renamed():
    rows = [_Row([(10, 10, 110, 30), (110, 10, 210, 30)], (10, 10, 210, 30)), _Row([(10, 30, 110, 50), (110, 30, 210, 50)], (10, 30, 210, 50))]
    # PyMuPDF renames empty/duplicate header names ("Col1"); the first extracted row is still the header.
    found = _Found((10, 10, 210, 50), rows, _Header(["Col1", "Qty"], False), [], [["", "Qty"], ["x", "2"]])
    table = _normalize([found], _page_with([found]), "d")[0]
    assert [c.header_text for c in table.columns] == ["", "Qty"]
    assert [[c.text for c in r.cells] for r in table.rows] == [["x", "2"]]


def test_column_extent_comes_from_cell_geometry_not_equal_split():
    rows = [
        _Row([(10, 10, 60, 30), (60, 10, 310, 30)], (10, 10, 310, 30)),
        _Row([(10, 30, 60, 50), (60, 30, 310, 50)], (10, 30, 310, 50)),
    ]
    found = _Found((10, 10, 310, 50), rows, _Header(["No", "Deskripsi"], False), [], [["No", "Deskripsi"], ["1", "Pengiriman"]])
    table = _normalize([found], _page_with([found]), "d")[0]
    assert table.columns[0].x2 == pytest.approx(60 / 400)
    assert table.columns[1].x1 == pytest.approx(60 / 400)
    assert table.columns[1].x2 == pytest.approx(310 / 400)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
