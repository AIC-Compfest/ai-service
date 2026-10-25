"""Shared test fixtures."""
import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def _never_purge_real_page_images(monkeypatch):
    """Endpoint tests must not delete the working tree's ``out/pages``; the purge tests set their own directory and hours."""
    monkeypatch.setattr(settings, "page_retention_hours", 0)
