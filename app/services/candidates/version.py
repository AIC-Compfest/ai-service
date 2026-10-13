from __future__ import annotations

import hashlib
from pathlib import Path


def candidate_pipeline_version(config_dir: str | Path) -> str:
    """Hash candidate code and configuration so labels cannot silently outlive features."""
    root = Path(__file__).resolve().parent
    config = Path(config_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted((*root.rglob("*.py"), *config.rglob("*.yaml"), *config.rglob("*.csv"))):
        if path.name == "__pycache__":
            continue
        digest.update(str(path.relative_to(root.parent.parent.parent) if path.is_relative_to(root.parent.parent.parent) else path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]