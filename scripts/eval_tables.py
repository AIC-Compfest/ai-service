"""Evaluate reviewed table gold against extracted table evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(tables):
    return {tuple(str(cell).strip().casefold() for cell in row) for table in tables for row in table.get("rows", [])}


def evaluate(gold: dict, prediction: dict) -> dict:
    """Return exact normalized row metrics only for explicitly reviewed gold."""
    if not gold.get("reviewed"):
        raise ValueError("gold must set reviewed=true before evaluation")
    expected, actual = _rows(gold.get("tables", [])), _rows(prediction.get("tables", []))
    matched = len(expected & actual)
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(expected) if expected else 0.0
    return {"reviewed": True, "expected_rows": len(expected), "predicted_rows": len(actual), "matched_rows": matched, "row_precision": precision, "row_recall": recall, "row_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.gold.read_text()), json.loads(args.prediction.read_text())), indent=2))


if __name__ == "__main__":
    main()
