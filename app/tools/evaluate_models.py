#!/usr/bin/env python
"""Evaluate trained models from their out-of-fold (OOF) predictions.

Reads every <models-dir>/<DOC>.<field>.binary.v1/ bundle (bundle.json + oof_predictions.json)
and writes:
    evaluation_fields.csv      one row per field: document-level and candidate-level metrics + verdict
    evaluation_thresholds.csv  accepted / precision / coverage for thresholds 0.1 .. 0.9 per field
    evaluation_report.md       readable report grouped by document type

Nothing here retrains a model. All numbers come from the OOF scores that train_binary wrote,
so they are estimates on templates the model did not see during its fold.

Metrics per field
  Candidate-level (what the trainer's binary_metrics reports, plus threshold-free measures):
    prevalence      share of candidate rows that are positive (the chance a random candidate is right)
    ap              average precision (area under precision-recall; prevalence is the baseline)
    auc             ROC AUC (0.5 is random)
    precision_at_floor / coverage_at_floor   at bundle t_unknown (or --threshold), with 95% Wilson CI
  Document-level (the question extraction actually asks):
    top1_hit        share of documents whose highest-scored candidate is a positive
    random_top1     expected hit rate if one candidate were picked at random per document
    top1_lift       top1_hit / random_top1
    Documents with no positive candidate are excluded from top1 and counted separately.

Verdicts (thresholds are flags):
    no_positives    field has no positive rows at all
    strong          top1_hit >= --strong-hit and lift >= 2
    usable          top1_hit >= --usable-hit and lift >= --min-lift
    weak            lift > 1 but below the usable bar
    no_better_than_random   lift <= 1.2
    Any verdict gets a "low_sample" note when fewer than --min-docs documents have a positive candidate.

Example (PowerShell, from the repo root):
    conda run -n aic_compfest python -m app.tools.evaluate_models --models-dir models\\all_v3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.training.cv import make_group_folds
from app.services.training.metrics import wilson_interval

THRESHOLD_GRID = [round(0.1 * step, 1) for step in range(1, 10)]
FIELD_COLUMNS = [
    "doc_type", "field", "verdict", "notes", "rows", "documents", "docs_with_positive", "docs_without_positive",
    "template_groups", "folds", "positives", "prevalence", "top1_hit", "top1_ci_low", "top1_ci_high",
    "random_top1", "top1_lift", "ap", "auc", "threshold", "accepted", "precision_at_floor",
    "precision_ci_low", "precision_ci_high", "coverage_at_floor", "accepted_at_t_auto", "precision_at_t_auto",
]


# ----------------------------------------------------------------------------- metrics
def average_precision(labels: list[int], scores: list[float]) -> float | None:
    """Same definition as sklearn.average_precision_score: tied scores move together."""
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: -pair[0])
    true_pos = seen = 0
    previous_recall = 0.0
    total = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        block = ordered[i:j]
        true_pos += sum(label for _, label in block)
        seen += len(block)
        recall = true_pos / positives
        total += (recall - previous_recall) * (true_pos / seen)
        previous_recall = recall
        i = j
    return total


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):  # average ranks over ties
        j = i
        while j + 1 < len(ordered) and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = average
        i = j + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def document_top1(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    hits = 0
    random_expectation = 0.0
    with_positive = 0
    for candidates in by_document.values():
        positives = sum(row["label"] for row in candidates)
        if positives == 0:
            continue
        with_positive += 1
        best = max(candidates, key=lambda row: row["score"])  # first max wins on ties
        hits += best["label"]
        random_expectation += positives / len(candidates)
    return {
        "documents": len(by_document),
        "docs_with_positive": with_positive,
        "docs_without_positive": len(by_document) - with_positive,
        "top1_hit": hits / with_positive if with_positive else None,
        "random_top1": random_expectation / with_positive if with_positive else None,
    }


def at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    accepted = [label for label, score in zip(labels, scores) if score >= threshold]
    correct = sum(accepted)
    n = len(labels)
    low, high = wilson_interval(correct, len(accepted)) if accepted else (0.0, 0.0)
    return {
        "accepted": len(accepted),
        "precision": correct / len(accepted) if accepted else None,
        "precision_ci_low": low if accepted else None,
        "precision_ci_high": high if accepted else None,
        "coverage": len(accepted) / n if n else 0.0,
    }


# ----------------------------------------------------------------------------- per field
def evaluate_bundle(bundle: Path, threshold_override: float | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    rows = json.loads((bundle / "oof_predictions.json").read_text(encoding="utf-8"))
    doc_type, field = meta["doc_type"], meta["field"]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    groups = [row["template_group"] for row in rows]
    t_floor = threshold_override if threshold_override is not None else float(meta["thresholds"]["t_unknown"])
    t_auto = float(meta["thresholds"]["t_auto"])

    top1 = document_top1(rows)
    floor = at_threshold(labels, scores, t_floor)
    auto = at_threshold(labels, scores, t_auto)
    positives = sum(labels)
    result: dict[str, Any] = {
        "doc_type": doc_type, "field": field, "rows": len(rows), "documents": top1["documents"],
        "docs_with_positive": top1["docs_with_positive"], "docs_without_positive": top1["docs_without_positive"],
        "template_groups": len(set(groups)), "folds": len(make_group_folds(groups)),
        "positives": positives, "prevalence": positives / len(rows) if rows else 0.0,
        "top1_hit": top1["top1_hit"], "random_top1": top1["random_top1"],
        "top1_lift": (top1["top1_hit"] / top1["random_top1"]) if top1["random_top1"] else None,
        "ap": average_precision(labels, scores), "auc": roc_auc(labels, scores),
        "threshold": t_floor, "accepted": floor["accepted"], "precision_at_floor": floor["precision"],
        "precision_ci_low": floor["precision_ci_low"], "precision_ci_high": floor["precision_ci_high"],
        "coverage_at_floor": floor["coverage"], "accepted_at_t_auto": auto["accepted"],
        "precision_at_t_auto": auto["precision"],
    }
    sweep = []
    for threshold in THRESHOLD_GRID:
        point = at_threshold(labels, scores, threshold)
        sweep.append({"doc_type": doc_type, "field": field, "threshold": threshold, **point})
    return result, sweep


def verdict_for(result: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    notes = []
    if result["positives"] == 0:
        return "no_positives", "no positive rows; nothing to learn"
    if result["top1_hit"] is None:
        return "no_positives", "no document has a positive candidate"
    hit, lift = result["top1_hit"], result["top1_lift"] or 0.0
    if lift <= 1.2:
        verdict = "no_better_than_random"
    elif hit >= args.strong_hit and lift >= 2:
        verdict = "strong"
    elif hit >= args.usable_hit and lift >= args.min_lift:
        verdict = "usable"
    else:
        verdict = "weak"
    if result["docs_with_positive"] < args.min_docs:
        notes.append(f"low_sample ({result['docs_with_positive']} docs)")
    if result["coverage_at_floor"] >= 0.9:
        notes.append("accepts almost every candidate")
    if result["accepted"] == 0:
        notes.append("never reaches the floor")
    if result["docs_without_positive"] > 0:
        count = result["docs_without_positive"]
        notes.append(f"{count} doc{'s' if count != 1 else ''} with no true candidate")
    return verdict, "; ".join(notes)


# ----------------------------------------------------------------------------- output
def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, results: list[dict[str, Any]], skipped: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Model evaluation (out-of-fold)", "",
        f"Models: `{args.models_dir}`  |  floor threshold: from bundle (t_unknown) unless overridden",
        "",
        "**How to read this.** `top1_hit` is the share of documents where the best-scored candidate is correct; "
        "`random_top1` is what picking a candidate at random would give; `top1_lift` = hit / random. "
        "Precision at the floor is only meaningful next to `prevalence`, and a `-` means no candidate reached the floor.",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result["verdict"]] += 1
    lines.append("Verdicts: " + ", ".join(f"{verdict}={count}" for verdict, count in sorted(counts.items())))
    if skipped:
        lines.append("")
        lines.append("Not evaluated (no bundle): " + ", ".join(f"{s['doc_type']}.{s['field']} ({s['status']})" for s in skipped))
    for doc_type in sorted({r["doc_type"] for r in results}):
        lines += ["", f"## {doc_type}", "",
                  "| field | verdict | top1 hit (95% CI) | random | lift | prec@floor (CI) | accepted | coverage | prevalence | docs | notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for result in (r for r in results if r["doc_type"] == doc_type):
            ci = (f"{fmt(result['top1_ci_low'])}–{fmt(result['top1_ci_high'])}" if result["top1_hit"] is not None else "-")
            prec = (f"{fmt(result['precision_at_floor'])} ({fmt(result['precision_ci_low'])}–{fmt(result['precision_ci_high'])})"
                    if result["accepted"] else "-")
            lines.append(f"| {result['field']} | {result['verdict']} | {fmt(result['top1_hit'])} ({ci}) | {fmt(result['random_top1'])} | "
                         f"{fmt(result['top1_lift'], 2)} | {prec} | {result['accepted']} | {fmt(result['coverage_at_floor'])} | "
                         f"{fmt(result['prevalence'])} | {result['docs_with_positive']} | {result['notes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models from their out-of-fold predictions.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="folder holding the <DOC>.<field>.binary.v1 bundles")
    parser.add_argument("--out", type=Path, help="report folder (default: <models-dir>/evaluation)")
    parser.add_argument("--threshold", type=float, help="floor threshold to report (default: each bundle's t_unknown)")
    parser.add_argument("--doc-types", nargs="+", metavar="DOC", help="evaluate only these document types")
    parser.add_argument("--fields", nargs="+", metavar="FIELD", help="evaluate only these fields (DOC.field or bare name)")
    parser.add_argument("--strong-hit", type=float, default=0.8, help="top1 hit needed for 'strong' (default 0.8)")
    parser.add_argument("--usable-hit", type=float, default=0.6, help="top1 hit needed for 'usable' (default 0.6)")
    parser.add_argument("--min-lift", type=float, default=1.5, help="top1 lift over random needed for 'usable' (default 1.5)")
    parser.add_argument("--min-docs", type=int, default=20, help="documents with a positive below this get a low_sample note (default 20)")
    args = parser.parse_args(argv)
    if not args.models_dir.is_dir():
        parser.error(f"not a folder: {args.models_dir}")
    args.out = args.out or args.models_dir / "evaluation"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundles = sorted(path for path in args.models_dir.glob("*.binary.v1") if (path / "oof_predictions.json").exists())
    if not bundles:
        print(f"No bundles with oof_predictions.json in {args.models_dir}", file=sys.stderr)
        return 1

    skipped: list[dict[str, Any]] = []
    summary_path = args.models_dir / "training_summary.json"
    if summary_path.exists():
        for record in json.loads(summary_path.read_text(encoding="utf-8")).get("results", []):
            if record.get("status") in ("skipped", "failed", "not_run"):
                skipped.append({"doc_type": record["doc_type"], "field": record["field"], "status": record["status"]})

    results, sweep = [], []
    for bundle in bundles:
        meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        if args.doc_types and meta["doc_type"] not in args.doc_types:
            continue
        if args.fields and f"{meta['doc_type']}.{meta['field']}" not in args.fields and meta["field"] not in args.fields:
            continue
        result, points = evaluate_bundle(bundle, args.threshold)
        result["verdict"], result["notes"] = verdict_for(result, args)
        result["top1_ci_low"], result["top1_ci_high"] = (
            wilson_interval(round(result["top1_hit"] * result["docs_with_positive"]), result["docs_with_positive"])
            if result["top1_hit"] is not None else (None, None))
        results.append(result)
        sweep.extend(points)

    if not results:
        print("No models matched the filters.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "evaluation_fields.csv", FIELD_COLUMNS, results)
    write_csv(args.out / "evaluation_thresholds.csv",
              ["doc_type", "field", "threshold", "accepted", "precision", "precision_ci_low", "precision_ci_high", "coverage"], sweep)
    write_markdown(args.out / "evaluation_report.md", results, skipped, args)

    print(f"{'doc_type':14} {'field':24} {'verdict':22} {'top1':>6} {'random':>7} {'lift':>6} {'acc':>5} {'cov':>6} {'docs':>5}  notes")
    for r in results:
        print(f"{r['doc_type']:14} {r['field']:24} {r['verdict']:22} {fmt(r['top1_hit']):>6} {fmt(r['random_top1']):>7} "
              f"{fmt(r['top1_lift'], 2):>6} {r['accepted']:5d} {fmt(r['coverage_at_floor']):>6} {r['docs_with_positive']:5d}  {r['notes']}")
    print(f"\nWrote {args.out / 'evaluation_report.md'}, evaluation_fields.csv, evaluation_thresholds.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())