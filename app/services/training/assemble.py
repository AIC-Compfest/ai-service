from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.candidates.training_schema import classify_column


@dataclass
class TrainingDataset:
    x: np.ndarray
    y: np.ndarray
    feature_columns: list[str]
    groups: np.ndarray
    documents: np.ndarray
    candidates: list[dict[str, Any]]
    candidate_type: str


def assemble(candidate_dir: str | Path, doc_type: str, field: str, *, include_ablate: bool = True) -> TrainingDataset:
    candidate_dir = Path(candidate_dir)
    manifest = json.loads((candidate_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads("{}")
    policy_path = candidate_dir / "training_policy.yaml"
    if policy_path.exists():
        import yaml
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    label = f"label__{field}"
    source = next((path for path in candidate_dir.glob(f"candidate_labels_{doc_type}_*.csv") if label in next(csv.reader(path.open(encoding="utf-8", newline="")), [])), None)
    if source is None:
        raise FileNotFoundError(f"no exported labels for {doc_type}.{field}")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    feature_columns = []
    for name, spec in manifest.items():
        role, train_policy = classify_column(name, policy)
        if train_policy in {"keep", "ablate"} and (include_ablate or train_policy != "ablate") and spec.get("dtype") in {"bool", "int", "float", "cat"} and (doc_type + "." + field in spec.get("applies_to", []) or "*" in spec.get("applies_to", [])):
            if name in rows[0] if rows else False:
                feature_columns.append(name)
    if not feature_columns:
        raise ValueError(f"no training features available for {doc_type}.{field}")
    trainable_reviews = {"auto_accepted", "human_confirmed", "human_corrected"}
    rows = [row for row in rows if row.get(f"label_review__{field}") in trainable_reviews]
    if not rows:
        raise ValueError(f"no trainable labels for {doc_type}.{field}; pending_review and unreadable labels are excluded")
    values = []
    for row in rows:
        parsed = []
        for name in feature_columns:
            spec = manifest[name]
            value = row.get(name, "")
            if value == "":
                parsed.append(np.nan)
            elif spec.get("dtype") == "cat":
                vocab = spec.get("vocab", [])
                parsed.append(vocab.index(value) if value in vocab else vocab.index("other") if "other" in vocab else -1)
            else:
                parsed.append(float(value))
        values.append(parsed)
    return TrainingDataset(np.asarray(values, dtype=float), np.asarray([int(row.get(label, "0") or 0) for row in rows], dtype=int), feature_columns, np.asarray([row.get("m_template_group", "UNKNOWN") for row in rows]), np.asarray([row.get("m_document_id", "") for row in rows]), rows, rows[0].get("m_cand_type", "") if rows else "")