"""Deterministic candidate-recall evaluation against literal ground truth."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


def evaluate(candidate_dir: Path, ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(candidate_dir.glob("candidates_*.csv")):
        rows.extend(csv.DictReader(path.open(encoding="utf-8", newline="")))
    by_doc = defaultdict(list)
    for row in rows: by_doc[row["m_document_id"]].append(row)
    hits, totals, misses = defaultdict(int), defaultdict(int), []
    for item in ground_truth:
        field, doc = item["field"], item["document_id"]
        totals[field] += 1
        value = str(item["value"])
        same = [r for r in by_doc[doc] if r["m_parsed_value"] == value and int(r["m_page"]) == int(item["page"])]
        target = item["bbox_norm"]
        if any(iou([float(row[k]) for k in ("m_x1", "m_y1", "m_x2", "m_y2")], target) >= .3 for row in same):
            hits[field] += 1
        else:
            reason = "no_candidate_with_value" if not same else "bbox_iou_below_0.3"
            misses.append({"document_id": doc, "field": field, "value": value, "reason": reason})
    return {"recall_by_field": {field: hits[field] / total for field, total in sorted(totals.items())}, "misses": misses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    result = evaluate(args.candidates, truth["fields"] if isinstance(truth, dict) else truth)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out: args.out.write_text(encoded, encoding="utf-8")
    else: print(encoded, end="")

if __name__ == "__main__": main()
