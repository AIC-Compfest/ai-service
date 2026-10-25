"""Where field confidences come from (see docs/table_reliability/CONFIDENCE.md).

A value is only as trustworthy as its source. Table-derived fields use the structural score of the tables they were
read from (a measured 0..1 quantity); fields found by keyword heuristics keep a labelled constant. No constant here
is calibrated against labelled data: they are placeholders with an explicit ``confidence_source`` so they can be found.
"""
from __future__ import annotations

from typing import Any

HEURISTIC_CONSTANT = "heuristic_constant"      # uncalibrated literal chosen by a developer
LINE_SCAN_CONSTANT = 0.50                       # items rebuilt from loose text lines (no table found): weaker than a table
MAX_TABLE_DERIVED = 0.95                        # a table read can still be wrong; never report certainty
TABLE_SOURCE = "table_structural_score"
LINE_SCAN_SOURCE = "line_scan_heuristic"


def structural_score(table: dict[str, Any]) -> float | None:
    score = ((table.get("validation") or {}).get("structural") or {}).get("score")
    if score is None:
        score = table.get("confidence")
    return float(score) if score else None


def table_derived_confidence(tables: list[dict[str, Any]], has_value: bool) -> tuple[float, str]:
    """Confidence of a value read from ``tables``: the weakest structural score among them (capped), or the line-scan constant."""
    if not has_value:
        return 0.0, "no_value"
    scores = [s for s in (structural_score(t) for t in tables) if s is not None]
    if not tables or not scores:
        return LINE_SCAN_CONSTANT, LINE_SCAN_SOURCE
    return round(min(min(scores), MAX_TABLE_DERIVED), 4), TABLE_SOURCE
