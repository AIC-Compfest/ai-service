from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import FieldStatus, ReviewState
from .store import LabelStore


TRAINABLE_STATES = {ReviewState.AUTO, ReviewState.CONFIRMED, ReviewState.CORRECTED}


def export_labels(candidate_dir: str | Path, label_root: str | Path, *, include_missed_as_negative: bool = True, reground: bool = False) -> dict[str, Any]:
    candidate_dir, label_root = Path(candidate_dir), Path(label_root)
    store = LabelStore(label_root)
    labels = {(label.document_id, label.field): label for label in store.list()}
    report: dict[str, Any] = {"labels": len(labels), "files": [], "stale_pipeline": []}
    for source in sorted(candidate_dir.glob("candidates_*.csv")):
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            columns = handle.seek(0) or next(csv.reader(handle))
        if not rows:
            continue
        doc_type = rows[0]["m_doc_type"]
        candidate_type = rows[0]["m_cand_type"]
        versions = {row.get("m_pipeline_version", "") for row in rows if row.get("m_pipeline_version", "")}
        stale = [label for label in labels.values() if label.doc_type == doc_type and label.m_pipeline_version and versions and label.m_pipeline_version not in versions]
        if stale and not reground:
            raise ValueError(f"stale labels detected for {doc_type}; rerun ground with --reground before export")
        field_names = [name.removeprefix("label__") for name in columns if name.startswith("label__")]
        derived = []
        for row in rows:
            output = dict(row)
            for field in field_names:
                label = labels.get((row["m_document_id"], field))
                positive = label and row["m_candidate_id"] in set(filter(None, label.positive_candidate_ids.split("|")))
                trainable = label and label.review_state in TRAINABLE_STATES
                value = "" if not label or not trainable or (label.status == FieldStatus.MISSED and not include_missed_as_negative) else int(bool(positive))
                output[f"label__{field}"] = value
                output[f"label_state__{field}"] = label.status if label else ""
                output[f"label_review__{field}"] = label.review_state if label else ""
                output[f"label_source__{field}"] = label.label_source if label else ""
            derived.append(output)
        destination = candidate_dir / f"candidate_labels_{doc_type}_{candidate_type}.csv"
        _write(destination, derived)
        report["files"].append(str(destination))
    (candidate_dir / "label_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row}, key=lambda name: (0 if name.startswith("m_") else 2 if name.startswith("label") else 1, name))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)