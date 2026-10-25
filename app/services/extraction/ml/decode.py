from __future__ import annotations

from typing import Any


def decode(scores: dict[str, list[tuple[dict[str, Any], float]]], field_specs: dict[str, dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    result = {}
    used = set()
    for field, values in scores.items():
        values = sorted(values, key=lambda item: (-item[1], int(item[0].get("m_page", 0)), int(item[0].get("m_line_idx", 0))))
        many = field_specs.get(field, {}).get("cardinality") == "many"
        selected = []
        for candidate, score in values:
            if candidate["m_candidate_id"] in used:
                continue
            selected.append((candidate, score))
            used.add(candidate["m_candidate_id"])
            if not many:
                break
        best_score = selected[0][1] if selected else 0.0
        result[field] = {"status": "accepted" if best_score >= thresholds.get(field, {}).get("t_auto", 0.9) else "review" if selected else "unknown", "candidates": selected, "score": best_score}
    return result
