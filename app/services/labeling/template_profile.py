from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .store import LabelStore


def build_template_profile(candidate_dir: str | Path, labels_dir: str | Path, document_id: str, output_dir: str | Path) -> dict[str, Any]:
    candidate_dir, output_dir = Path(candidate_dir), Path(output_dir)
    store = LabelStore(labels_dir)
    labels = [label for label in store.list(document_id=document_id) if label.positive_candidate_ids]
    candidates = _load_candidates(candidate_dir, document_id)
    profile = {"template_group": document_id, "document_id": document_id, "fields": [], "created_from": "audited_labels"}
    for label in labels:
        for candidate_id in label.positive_candidate_ids.split("|"):
            candidate = candidates.get(candidate_id)
            if not candidate:
                continue
            profile["fields"].append({"field": label.field, "cand_type": candidate.get("m_cand_type", ""), "page_pos": "first", "center": [(float(candidate["m_x1"]) + float(candidate["m_x2"])) / 2, (float(candidate["m_y1"]) + float(candidate["m_y2"])) / 2], "size": [float(candidate["m_x2"]) - float(candidate["m_x1"]), float(candidate["m_y2"]) - float(candidate["m_y1"])], "ctx": {key.removeprefix("ctx_"): candidate.get(key, "") for key in ("ctx_left", "ctx_above", "ctx_line")}, "value_norm": candidate.get("m_parsed_norm", ""), "anchor_text": label.anchor_text})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document_id}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def _load_candidates(candidate_dir: Path, document_id: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in candidate_dir.glob("candidates_*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("m_document_id") == document_id:
                    result[row["m_candidate_id"]] = row
    return result