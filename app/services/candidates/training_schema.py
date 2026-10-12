from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import yaml


def load_policy(config_dir: str | Path) -> dict[str, Any]:
    path = Path(config_dir) / "training_policy.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def classify_column(name: str, policy: dict[str, Any]) -> tuple[str, str]:
    if name.startswith(tuple(policy.get("always_drop_prefixes", ["m_", "ctx_", "label_", "prelabel__"]))):
        return "meta_id" if name.startswith("m_") else "context_text" if name.startswith("ctx_") else "label", "always_drop"
    if name in policy.get("always_drop_names", []):
        return "raw_value", "always_drop"
    if name in policy.get("drop_default", []):
        return "value_echo", "drop_default"
    if name in policy.get("ablate", []):
        return "value_magnitude", "ablate"
    return "feature", "keep"


def select_training_columns(output_dir: str | Path, doc_type: str, field: str, include_ablate: bool = True) -> list[str]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    policy = yaml.safe_load((output_dir / "training_policy.yaml").read_text(encoding="utf-8")) if (output_dir / "training_policy.yaml").exists() else {}
    label = f"label__{field}"
    source = next((path for path in output_dir.glob(f"candidates_{doc_type}_*.csv") if label in next(csv.reader(path.open(encoding="utf-8", newline="")), [])), None)
    if source is None:
        return []
    source_columns = next(csv.reader(source.open(encoding="utf-8", newline="")))
    columns = []
    for name in source_columns:
        spec = manifest.get(name, {})
        role, train_policy = classify_column(name, policy)
        if train_policy == "keep" or (include_ablate and train_policy == "ablate"):
            if doc_type + "." + field in spec.get("applies_to", []) or "*" in spec.get("applies_to", []):
                if spec.get("dtype") in {"bool", "int", "float", "cat"}:
                    columns.append(name)
    return columns