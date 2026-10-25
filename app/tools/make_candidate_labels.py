#!/usr/bin/env python
"""Build the candidate_labels_<DOC>_<type>.csv files the trainer reads, from labels already
written into the candidate CSVs (label__<field> columns).

Use this when the candidate files were labeled by hand, so the label store and
`labeling export` are not needed. Each labeled cell is marked as trainable with the
review state you choose (default: auto_accepted), exactly as if it had been entered
through the labeling flow. Otherwise, use `labeling.py` to ground the labels into a label store and export them.

Example (PowerShell, from the repo root):

    conda run -n aic_compfest python -m app.tools.make_candidate_labels `
        --candidate-dir data\\candidates_v3

What it writes into --candidate-dir:
    candidate_labels_<DOC>_<type>.csv   one per candidates_<DOC>_<type>.csv (all original columns plus
                                        label_state__, label_review__, label_source__ per field)
    label_report.json                   counts per file, same shape as `labeling export` writes

The original candidates_*.csv files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REVIEW_STATES = ("auto_accepted", "human_confirmed", "human_corrected")
VALID_LABELS = {"0", "1", ""}
REPORT_NAME = "label_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create candidate_labels_*.csv from the manual label__ columns in candidates_*.csv.")
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="folder with candidates_*.csv; the candidate_labels_*.csv files are written here too")
    parser.add_argument("--review-state", choices=REVIEW_STATES, default="auto_accepted",
                        help="review state given to every labeled cell (default: auto_accepted)")
    parser.add_argument("--label-source", default="manual",
                        help="value written to label_source__<field> (default: manual)")
    parser.add_argument("--report", type=Path, help=f"path of the report (default: <candidate-dir>/{REPORT_NAME})")
    parser.add_argument("--overwrite", action="store_true", help="replace existing candidate_labels_*.csv files")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the table, write nothing")
    args = parser.parse_args(argv)
    if not args.candidate_dir.is_dir():
        parser.error(f"not a folder: {args.candidate_dir}")
    return args


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def build_file(path: Path, review_state: str, label_source: str) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Return (summary, output rows, output header) for one candidates_*.csv file."""
    header, rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path.name} has no rows")
    doc_types = {row.get("m_doc_type", "") for row in rows}
    cand_types = {row.get("m_cand_type", "") for row in rows}
    if len(doc_types) != 1 or len(cand_types) != 1 or "" in doc_types or "" in cand_types:
        raise ValueError(f"{path.name}: m_doc_type / m_cand_type must be one value per file, got {sorted(doc_types)} / {sorted(cand_types)}")
    doc_type, cand_type = doc_types.pop(), cand_types.pop()

    label_columns = [name for name in header if name.startswith("label__")]
    for name in label_columns:
        bad = sorted({row[name] for row in rows} - VALID_LABELS)
        if bad:
            raise ValueError(f"{path.name}: column {name} has values {bad[:5]}; only 0, 1 or blank are allowed")

    output_header = list(header)
    per_field: dict[str, Counter[str]] = {}
    for name in label_columns:
        field = name[len("label__"):]
        for column in (f"label_state__{field}", f"label_review__{field}", f"label_source__{field}"):
            if column not in output_header:
                output_header.append(column)
        counts: Counter[str] = Counter()
        counts["rows"] = len(rows)
        counts["labeled"] = sum(row[name] in ("0", "1") for row in rows)
        counts["positives"] = sum(row[name] == "1" for row in rows)
        counts["negatives"] = sum(row[name] == "0" for row in rows)
        counts["unlabeled"] = counts["rows"] - counts["labeled"]
        per_field[field] = counts

    output_rows = []
    for row in rows:
        out = dict(row)
        for name in label_columns:
            field = name[len("label__"):]
            value = row[name]
            labeled = value in ("0", "1")
            out[f"label_state__{field}"] = ("present" if value == "1" else "absent") if labeled else ""
            out[f"label_review__{field}"] = review_state if labeled else ""
            out[f"label_source__{field}"] = label_source if labeled else ""
        output_rows.append(out)

    summary = {
        "source": path.name,
        "file": f"candidate_labels_{doc_type}_{cand_type}.csv",
        "doc_type": doc_type,
        "cand_type": cand_type,
        "fields": per_field,
    }
    return summary, output_rows, output_header


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = sorted(args.candidate_dir.glob("candidates_*.csv"))
    if not sources:
        print(f"No candidates_*.csv files in {args.candidate_dir}", file=sys.stderr)
        return 1

    built = []
    errors = []
    for path in sources:
        try:
            built.append(build_file(path, args.review_state, args.label_source))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        print("Stopped: fix these before exporting:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1

    # The trainer picks the label file per document type, so one field must not be labeled in two
    # candidate files of the same document type.
    owners: dict[tuple[str, str], list[str]] = defaultdict(list)
    for summary, _, _ in built:
        for field in summary["fields"]:
            owners[(summary["doc_type"], field)].append(summary["source"])
    duplicates = {key: files for key, files in owners.items() if len(files) > 1}
    if duplicates:
        print("Stopped: the same label is in more than one candidates file of one document type:", file=sys.stderr)
        for (doc, field), files in sorted(duplicates.items()):
            print(f"  - {doc}.{field}: {files}", file=sys.stderr)
        return 1

    existing = [summary["file"] for summary, _, _ in built if (args.candidate_dir / summary["file"]).exists()]
    if existing and not args.overwrite and not args.dry_run:
        print(f"Stopped: {len(existing)} candidate_labels file(s) already exist, e.g. {existing[0]}. "
              "Use --overwrite to replace them.", file=sys.stderr)
        return 1

    print(f"{'file':44} {'field':24} {'rows':>6} {'labeled':>8} {'positive':>9} {'negative':>9} {'unlabeled':>10}")
    total_labeled = 0
    report_files = []
    for summary, output_rows, output_header in built:
        for field, counts in summary["fields"].items():
            total_labeled += counts["labeled"]
            print(f"{summary['file']:44} {field:24} {counts['rows']:6d} {counts['labeled']:8d} "
                  f"{counts['positives']:9d} {counts['negatives']:9d} {counts['unlabeled']:10d}")
        if not args.dry_run:
            destination = args.candidate_dir / summary["file"]
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=output_header, lineterminator="\n")
                writer.writeheader()
                writer.writerows(output_rows)
        report_files.append(str(args.candidate_dir / summary["file"]))
        print()

    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    report = {
        "labels": total_labeled,
        "files": report_files,
        "stale_pipeline": [],
        "review_state": args.review_state,
        "label_source": args.label_source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": "app.tools.make_candidate_labels",
    }
    report_path = args.report or args.candidate_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_files)} candidate_labels files and {report_path} ({total_labeled} labeled cells).")
    print("Next: python -m app.tools.train_all --candidate-dir " + str(args.candidate_dir) + " --out models\\v3 --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())