"""Fase 7: shared pipeline, upload limit, page-image retention, working-directory independence."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core import runtime
from app.core.config import PACKAGE_ROOT, settings
from app.main import app


def test_the_pipeline_is_built_once_and_shared():
    runtime.reset_pipeline()
    first, second = runtime.get_pipeline(), runtime.get_pipeline()
    assert first is second


def test_uploads_above_the_limit_are_refused_before_any_processing(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    client = TestClient(app)
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1)
    for path in ("/v1/classify", "/v1/signature/detect", "/v1/extract/pipeline/full", "/v1/extract/understand/evidence", "/v1/extract/INVOICE"):
        response = client.post(path, files={"file": ("big.pdf", big, "application/pdf")})
        assert response.status_code == 413, path


def test_purge_removes_only_old_folders_under_the_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "page_output_dir", tmp_path)
    monkeypatch.setattr(settings, "page_retention_hours", 1)
    old, fresh = tmp_path / "old-doc", tmp_path / "fresh-doc"
    old.mkdir()
    fresh.mkdir()
    (old / "page_0001.png").write_bytes(b"x")
    keep_file = tmp_path / "notes.txt"
    keep_file.write_text("keep")
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    assert runtime.purge_old_pages(force=True) == 1
    assert not old.exists() and fresh.exists() and keep_file.exists()


def test_purge_is_disabled_by_zero_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "page_output_dir", tmp_path)
    monkeypatch.setattr(settings, "page_retention_hours", 0)
    (tmp_path / "doc").mkdir()
    os.utime(tmp_path / "doc", (0, 0))
    assert runtime.purge_old_pages(force=True) == 0 and (tmp_path / "doc").exists()


def test_page_images_go_to_the_configured_directory(tmp_path, monkeypatch):
    from app.services.preprocessing.document_ingestion import DocumentIngestionService
    from app.tools.table_eval.synthetic import build_fixture

    monkeypatch.setattr(settings, "page_output_dir", tmp_path / "pages")
    ingested = DocumentIngestionService.ingest_bytes(build_fixture("single_ruled").pdf, "x.pdf", document_id="doc-1")
    assert (tmp_path / "pages" / "doc-1" / "page_0001.png").exists()
    assert ingested.pages[0].page_image["path"].endswith("doc-1/page_0001.png")


def test_the_application_imports_and_finds_its_files_from_another_working_directory(tmp_path):
    """No path may depend on the CWD: import the app and read config/models from an unrelated directory."""
    code = (
        "from app.main import app; from app.core.config import settings, candidates_config_dir; "
        "from app.services.classification.classifier import MODELS_DIR; "
        "assert (candidates_config_dir() / 'field_registry.yaml').exists(); "
        "assert (settings.ml_model_set_dir).is_dir(); assert (MODELS_DIR / 'doc_classifier_v1.joblib').exists(); print('CWD_OK')"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}, timeout=300)
    assert "CWD_OK" in done.stdout, done.stderr[-800:]
