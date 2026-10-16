"""Config resolution and caching (T-24, T-25): no CWD dependence, no YAML re-reads per page."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pymupdf

from app.core.config import PACKAGE_ROOT, Settings, candidates_config_dir, settings
from app.services.candidates.context import build_page
from app.services.candidates.tables import TableExtractor, TokenGridTableProvider
from app.services.candidates.tables import router as router_module
from app.services.candidates.tables import token_grid as token_grid_module
from app.tools.table_eval.synthetic import build_fixture

SCRIPT = """
import sys, pymupdf
from app.core.config import settings
settings.enable_table_extractor = True
from app.services.ocr.engine import OCREngine
doc = pymupdf.open(stream=open(sys.argv[1], "rb").read(), filetype="pdf")
evidence = OCREngine().process_pdf_page(doc[0], 1, {})
print("TABLES", len(evidence.tables), "CWD", __import__("os").getcwd())
"""


def test_default_config_dir_is_package_relative_and_overridable(tmp_path: Path):
    assert settings.config_dir == PACKAGE_ROOT / "config"
    assert candidates_config_dir().joinpath("line_item_aliases.yaml").exists()
    assert Settings(config_dir=tmp_path).config_dir == tmp_path


def test_table_discovery_works_from_a_different_working_directory(tmp_path: Path):
    """Before the fix this raised FileNotFoundError: config/candidates/table_resolution.yaml (path was CWD-relative)."""
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(build_fixture("single_ruled").pdf)
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT), "ENABLE_TABLE_EXTRACTOR": "true"}
    completed = subprocess.run([sys.executable, "-c", SCRIPT, str(pdf)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-800:]
    assert "TABLES 1" in completed.stdout
    assert str(tmp_path) in completed.stdout


def test_extractor_is_cached_and_yaml_is_not_reread_per_page(monkeypatch):
    router_module._cached_extractor.cache_clear()
    reads = []
    original = router_module.yaml.safe_load
    monkeypatch.setattr(router_module.yaml, "safe_load", lambda *a, **k: reads.append(1) or original(*a, **k))
    first = TableExtractor.cached(candidates_config_dir(), ["pymupdf_find_tables", "token_grid"])
    reads_after_first = len(reads)
    for _ in range(5):
        assert TableExtractor.cached(candidates_config_dir(), ["pymupdf_find_tables", "token_grid"]) is first
    assert len(reads) == reads_after_first >= 1


def test_ocr_engine_does_not_rebuild_the_extractor_per_page(monkeypatch):
    router_module._cached_extractor.cache_clear()
    builds = []
    original = TableExtractor.from_config.__func__
    monkeypatch.setattr(TableExtractor, "from_config", classmethod(lambda cls, *a, **k: builds.append(1) or original(cls, *a, **k)))
    from app.services.ocr.engine import OCREngine

    monkeypatch.setattr(settings, "enable_table_extractor", True)
    document = pymupdf.open(stream=build_fixture("single_ruled").pdf, filetype="pdf")
    engine = OCREngine()
    for _ in range(4):
        engine.process_pdf_page(document[0], 1, {})
    assert len(builds) == 1


def test_fuzzy_matcher_is_built_once_per_doc_type(monkeypatch):
    built = []
    original = token_grid_module.FuzzyFieldMatcher
    monkeypatch.setattr(token_grid_module, "FuzzyFieldMatcher", lambda *a, **k: built.append(a[0]) or original(*a, **k))
    config = candidates_config_dir()
    provider = TokenGridTableProvider(config / "line_item_aliases.yaml", config / "abbreviations.yaml")
    page = build_page({"page_number": 1, "lines": [{"text": "No Qty", "bbox_norm": [0.1, 0.1, 0.4, 0.12]}, {"text": "1 2", "bbox_norm": [0.1, 0.15, 0.4, 0.17]}]})
    for _ in range(3):
        provider.tables(page, "INVOICE", "d")
    provider.tables(page, "POD", "d")
    assert built == ["INVOICE", "POD"]
