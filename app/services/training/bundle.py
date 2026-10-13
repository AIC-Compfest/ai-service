from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib


def save_bundle(output_dir: str | Path, model_id: str, model: Any, metadata: dict[str, Any]) -> Path:
    output_dir = Path(output_dir) / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metadata = dict(metadata)
    metadata["model_id"] = model_id
    metadata["model_file"] = model_path.name
    fd, temp_name = tempfile.mkstemp(prefix="model.", suffix=".tmp", dir=output_dir)
    os.close(fd)
    try:
        joblib.dump(model, temp_name)
        os.replace(temp_name, model_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (output_dir / "bundle.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def load_bundle(path: str | Path, *, allow_mismatch: bool = False):
    path = Path(path)
    metadata = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256((path / metadata["model_file"]).read_bytes()).hexdigest()
    if actual != metadata["model_sha256"]:
        raise ValueError("model hash mismatch")
    return joblib.load(path / metadata["model_file"]), metadata