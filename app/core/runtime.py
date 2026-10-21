"""Process-wide runtime helpers: shared pipeline (K10), upload limit, page-image retention.

The pipeline holds the classifier, OCR engine and detectors; building one per request reloaded the ML models every time.
One instance is shared; per-call state is local to ``process_document`` and Docling conversions are serialised inside the Docling runner.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_LOCK = threading.Lock()
_PIPELINE = None
_LAST_PURGE = 0.0
PURGE_INTERVAL_SECONDS = 600


def get_pipeline():
    """The shared ``Phase5DocumentPipeline`` (created on first use)."""
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                from app.services.pipeline import Phase5DocumentPipeline  # heavy imports stay out of module import time

                _PIPELINE = Phase5DocumentPipeline()
    return _PIPELINE


def reset_pipeline() -> None:
    global _PIPELINE
    with _LOCK:
        _PIPELINE = None


def read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything above ``MAX_UPLOAD_SIZE_MB`` (HTTP 413) without buffering more than limit + 1 byte."""
    limit = int(settings.max_upload_size_mb) * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_size_mb} MB")
    return data


def purge_old_pages(now: float | None = None, force: bool = False) -> int:
    """Delete page-image folders older than ``PAGE_RETENTION_HOURS`` under ``PAGE_OUTPUT_DIR``; returns how many were removed.

    Only direct sub-folders of the configured output directory are touched. 0 hours disables the purge. Runs at most every
    ten minutes unless ``force``.
    """
    global _LAST_PURGE
    current = time.time() if now is None else now
    hours = int(settings.page_retention_hours)
    root = Path(settings.page_output_dir)
    if hours <= 0 or not root.is_dir() or (not force and current - _LAST_PURGE < PURGE_INTERVAL_SECONDS):
        return 0
    _LAST_PURGE = current
    removed = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and current - child.stat().st_mtime > hours * 3600:
                shutil.rmtree(child)
                removed += 1
        except OSError as error:  # a folder in use or already gone must not break a request
            logger.warning("page_purge_failed", folder=child.name, error=str(error))
    if removed:
        logger.info("page_purge_done", removed=removed, root=root.as_posix())
    return removed
