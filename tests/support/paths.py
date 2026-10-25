"""Portable locations for fixtures that live outside the ai-service repository.

The workspace root defaults to the parent of ``ai-service`` and can be overridden with
``VERIFLOW_WORKSPACE_ROOT``. The canonical dataset root can be overridden with ``VERIFLOW_DATASET_ROOT``.
Tests must skip with a written reason (see ``require``) when an external fixture is missing.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VERIFLOW_WORKSPACE_ROOT", AI_SERVICE_ROOT.parent))
DATASET_ROOT = Path(os.environ.get("VERIFLOW_DATASET_ROOT", WORKSPACE_ROOT / "DATASET" / "Dataset"))
SYNTHETIC_DATA_ROOT = WORKSPACE_ROOT / "synthetic-data"
SAMPLE_DATASETS_ROOT = WORKSPACE_ROOT / "sample_datasets"


def require(path: Path, what: str) -> Path:
    """Return ``path`` or raise ``unittest.SkipTest`` naming the missing external fixture."""
    if not path.exists():
        raise unittest.SkipTest(f"External fixture not found: {what} ({path.name}); set VERIFLOW_WORKSPACE_ROOT")
    return path
