from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import token_set_ratio

from .schema import FieldLabel, LabelSource, ReviewState
from .store import LabelStore


def transfer_template(candidate_dir: str | Path, labels_dir: str | Path, profile_path: str | Path, variant_id: str, template_group: str, *, label_version: str = "1.0.0", pipeline_version: str = "") -> int:
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    candidates = _load_candidates(Path(candidate_dir), variant_id)
    store = LabelStore(labels_dir)
    count = 0
    for item in profile.get("fields", []):
        pool = [candidate for candidate in candidates.values() if candidate.get("m_cand_type") == item["cand_type"]]
        scored = []
        for candidate in pool:
            center = ((float(candidate["m_x1"]) + float(candidate["m_x2"])) / 2, (float(candidate["m_y1"]) + float(candidate["m_y2"])) / 2)
            dx, dy = center[0] - item["center"][0], center[1] - item["center"][1]
            pos = math.exp(-((dx / .08) ** 2 + (dy / .08) ** 2))
            base_context = " ".join(item.get("ctx", {}).values())
            candidate_context = " ".join(candidate.get(key, "") for key in ("ctx_left", "ctx_above", "ctx_line"))
            context = token_set_ratio(base_context, candidate_context) / 100 if base_context and candidate_context else .5
            score = .55 * context + .45 * pos
            scored.append((score, candidate))
        scored.sort(key=lambda value: (-value[0], value[1].get("m_candidate_id", "")))
        if not scored:
            continue
        best_score, best = scored[0]
        state = ReviewState.AUTO if best_score >= .75 else ReviewState.PENDING
        label = FieldLabel(label_version, variant_id, "POD", template_group, 0, item["field"], "present" if best_score >= .5 else "absent", state, LabelSource.TRANSFER, best_score, best.get("m_raw_text", ""), best.get("m_parsed_norm", ""), int(best.get("m_page", 1)), best.get("m_candidate_id", ""), "", "transfer", best_score, 0, 0, item.get("anchor_text", ""), "", "", "", "", "", pipeline_version)
        store.upsert(label)
        count += 1
    return count


def _load_candidates(candidate_dir: Path, document_id: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in candidate_dir.glob("candidates_*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("m_document_id") == document_id:
                    result[row["m_candidate_id"]] = row
    return result