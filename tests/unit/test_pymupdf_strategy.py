"""PyMuPDF strategy selection and page-wide table rejection (T-10, F7)."""
from __future__ import annotations

import pymupdf
import yaml

from app.services.candidates.context import build_page
from app.services.candidates.tables import TableExtractor
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider
from app.tools.table_eval.metrics import area
from app.tools.table_eval.synthetic import build_fixture
from tests.support.paths import AI_SERVICE_ROOT

CONFIG_DIR = AI_SERVICE_ROOT / "config" / "candidates"


def _configured_provider() -> PyMuPDFTableProvider:
    return TableExtractor.from_config(CONFIG_DIR, ["pymupdf_find_tables"]).providers["pymupdf_find_tables"]


def _tables(provider, fixture_name: str, page_index: int = 0):
    document = pymupdf.open(stream=build_fixture(fixture_name).pdf, filetype="pdf")
    page = build_page({"page_number": page_index + 1, "lines": [], "_source_page": document[page_index]})
    return provider.tables(page, None, "d")


def test_configured_ratio_is_documented_and_loaded_from_yaml():
    options = yaml.safe_load((CONFIG_DIR / "table_resolution.yaml").read_text(encoding="utf-8"))["pymupdf"]
    assert options["max_page_area_ratio"] == 0.5
    assert _configured_provider().options["max_page_area_ratio"] == 0.5


def _full_page_text_grid() -> pymupdf.Page:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    for row in range(60):
        y = 40 + row * 13
        for col, x in enumerate((40, 190, 340, 470)):
            page.insert_text((x, y), f"Kolom{col}-{row}", fontsize=8)
    return page


def test_page_wide_text_table_is_rejected_by_configured_ratio():
    """Unrestricted, the 'text' strategy returns the whole page as one table (F7); the ratio removes it even if 'text' is enabled."""
    page = build_page({"page_number": 1, "lines": [], "_source_page": _full_page_text_grid()})
    unrestricted = PyMuPDFTableProvider().tables(page, None, "d")
    assert unrestricted and max(area(t.bbox) for t in unrestricted) > 0.5  # the failure being fixed
    assert PyMuPDFTableProvider({"strategies": ["lines", "text"], "max_page_area_ratio": 0.5}).tables(page, None, "d") == []
    assert _configured_provider().tables(page, None, "d") == []


def test_production_config_uses_lines_only_and_leaves_unruled_tables_to_token_geometry():
    assert _configured_provider().options["strategies"] == ["lines"]
    assert _tables(_configured_provider(), "single_unruled") == []


def test_text_strategy_is_still_available_when_configured():
    provider = PyMuPDFTableProvider({"strategies": ["lines", "text"], "max_page_area_ratio": 0.5})
    tables = _tables(provider, "single_unruled")
    assert len(tables) == 1 and area(tables[0].bbox) < 0.5 and len(tables[0].rows) == 6
    assert tables[0].reason == "pymupdf_find_tables:text"


def test_ruled_table_uses_lines_strategy():
    tables = _tables(_configured_provider(), "single_ruled")
    assert len(tables) == 1 and len(tables[0].rows) == 6
    assert [c.header_text for c in tables[0].columns] == ["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"]


def test_options_override_defaults_and_unknown_strategy_is_ignored():
    provider = PyMuPDFTableProvider({"strategies": ["nope", "lines"], "min_data_rows": 3})
    assert provider.options["min_data_rows"] == 3
    assert len(_tables(provider, "single_ruled")) == 1  # 'nope' raises inside PyMuPDF and is skipped
    assert _tables(PyMuPDFTableProvider({"strategies": ["lines"], "min_data_rows": 50}), "single_ruled") == []


def test_merge_strategies_keeps_non_overlapping_tables_only():
    merged = PyMuPDFTableProvider({"merge_strategies": True, "max_page_area_ratio": 0.5})
    tables = _tables(merged, "single_ruled")
    ids = [t.table_id for t in tables]
    assert len(ids) == len(set(ids))
    for i, a in enumerate(tables):
        for b in tables[i + 1:]:
            inter = area((max(a.bbox[0], b.bbox[0]), max(a.bbox[1], b.bbox[1]), min(a.bbox[2], b.bbox[2]), min(a.bbox[3], b.bbox[3])))
            assert inter / min(area(a.bbox), area(b.bbox)) <= 0.3
