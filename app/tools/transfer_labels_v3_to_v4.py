from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(row[f"m_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def _iou(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _match(source: dict[str, str], targets: list[dict[str, str]]) -> dict[str, str] | None:
    source_type = source.get("m_cand_type", "")
    pool = [row for row in targets if row.get("m_document_id") == source.get("m_document_id")
            and row.get("m_cand_type") == source_type and row.get("m_page") == source.get("m_page")]
    if not pool:
        return None
    exact = [row for row in pool if row.get("m_raw_text", "") == source.get("m_raw_text", "") and source.get("m_raw_text", "")]
    if exact:
        return max(exact, key=lambda row: _iou(_bbox(source), _bbox(row)))
    ranked = sorted(((_iou(_bbox(source), _bbox(row)), row) for row in pool), key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked and ranked[0][0] >= 0.5 else None


def transfer(v3_dir: Path, v4_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"source": str(v3_dir), "destination": str(v4_dir), "fields": {}, "unmatched_positive_rows": []}
    sources = sorted(v3_dir.glob("candidate_labels_*.csv"))
    if not sources:
        raise SystemExit(f"No v3 candidate label exports found under {v3_dir}")
    for source_path in sources:
        v3_header, v3_rows = _read(source_path)
        candidate_name = source_path.name.replace("candidate_labels_", "candidates_", 1)
        target_path = v4_dir / candidate_name
        if not target_path.exists():
            continue
        v4_header, v4_rows = _read(target_path)
        fields = [name.removeprefix("label__") for name in v3_header if name.startswith("label__")]
        field_stats = {field: {"v3_positive": 0, "matched_positive": 0, "unmatched_positive": 0, "matched_negative": 0} for field in fields}
        for field in fields:
            label = f"label__{field}"
            review = f"label_review__{field}"
            if label not in v4_header:
                v4_header.append(label)
            if review not in v4_header:
                v4_header.append(review)
            for source in v3_rows:
                value = source.get(label, "")
                if value not in {"0", "1"}:
                    continue
                match = _match(source, v4_rows)
                if value == "1":
                    field_stats[field]["v3_positive"] += 1
                if match is None:
                    if value == "1":
                        field_stats[field]["unmatched_positive"] += 1
                        report["unmatched_positive_rows"].append({"field": field, "doc_type": source.get("m_doc_type"), "document_id": source.get("m_document_id"), "cand_type": source.get("m_cand_type"), "page": source.get("m_page"), "raw_text": source.get("m_raw_text"), "bbox": _bbox(source)})
                    continue
                # Carry both explicit positives and negatives when candidate identity
                # transfers; unmatched/new candidates remain unlabeled.
                match[label] = value
                match[review] = "auto_accepted"
                if value == "1":
                    field_stats[field]["matched_positive"] += 1
                else:
                    field_stats[field]["matched_negative"] += 1
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=v4_header, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(v4_rows)
        for field, stats in field_stats.items():
            key = f"{source_path.stem.removeprefix('candidate_labels_')}.{field}"
            report["fields"][key] = stats
    report_path = v4_dir / "label_transfer_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Transferred labels into {v4_dir}")
    for field, stats in sorted(report["fields"].items()):
        print(f"{field}: positives {stats['matched_positive']}/{stats['v3_positive']} matched; {stats['unmatched_positive']} unmatched")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer v3 manual candidate labels to matching v4 candidates.")
    parser.add_argument("--v3", type=Path, required=True, help="candidate_v3 directory with candidate_labels_*.csv")
    parser.add_argument("--v4", type=Path, required=True, help="candidate_v4 directory with candidates_*.csv")
    args = parser.parse_args()
    transfer(args.v3, args.v4)


if __name__ == "__main__":
    main()
