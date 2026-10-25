"""Evaluate table predictions against gold (GOLD_REVIEW schema) at several levels.

Single document:
    eval_tables.py --gold gold.json --prediction evidence.json [--allow-unreviewed]
Corpus:
    eval_tables.py --gold-dir gold/regression --pred-dir out/pred/router_none --manifest gold/manifest.json \\
        --splits dev,val --row-content-dir gold/row_content --allow-unreviewed --report out/reports/x.json --markdown out/reports/x.md

Predictions are pipeline evidence (``pages[].tables[]``); a flat ``{"tables": [...]}`` draft is also accepted.
Gold with ``reviewed=false`` is rejected unless ``--allow-unreviewed`` is given, and the report is then stamped
as NOT human gold. Metrics are documented in ``app/tools/table_eval/metrics.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from app.tools.table_eval.metrics import (  # noqa: E402
    FAILURE_CODES,
    evaluate_doc,
    evaluate_row_content,
    summarize,
    summarize_row_content,
)
from app.tools.table_eval.model import EvalDoc, EvalTable, gold_from_json, prediction_from_evidence, read_json, validate_review  # noqa: E402

STAMP = "GT sintetis dan structural gold synthetic_grounded/constructed_exact; bukan gold manusia."


def _load_gold(gold: dict, allow_unreviewed: bool) -> EvalDoc:
    if "pages" in gold:
        return gold_from_json(gold, allow_unreviewed=allow_unreviewed)
    # Flat draft produced by app.tools.prelabel_tables: {"tables": [{"page_number", "headers", "rows"}]}.
    validate_review(gold, allow_unreviewed=allow_unreviewed)
    doc = EvalDoc(document_id=str(gold.get("document_id", "")), reviewed=bool(gold.get("reviewed")))
    for index, table in enumerate(gold.get("tables", []), start=1):
        doc.tables.append(
            EvalTable(
                table_id=str(table.get("table_id", index)),
                page=int(table.get("page_number", 1)),
                bbox=tuple(table.get("bbox_norm", (0.0, 0.0, 1.0, 1.0))),  # type: ignore[arg-type]
                role="line_items",
                headers=[str(h) for h in table.get("headers", [])],
                rows=[[str(c) for c in row] for row in table.get("rows", [])],
            )
        )
    return doc


def evaluate(gold: dict, prediction: dict, *, allow_unreviewed: bool = False, view: str = "line_items") -> dict:
    """Evaluate one document. Raises ``GoldRejected`` for unreviewed gold unless explicitly allowed."""
    gold_doc = _load_gold(gold, allow_unreviewed)
    result = evaluate_doc(gold_doc, prediction_from_evidence(prediction, gold_doc.document_id), view=view)
    summary = summarize(result.counts, result.failures)
    summary["reviewed"] = gold_doc.reviewed
    summary["stamp"] = None if gold_doc.reviewed else STAMP
    summary["pages_detail"] = result.pages
    return summary


def _group_keys(entry: dict | None, is_constructed: bool) -> list[str]:
    if is_constructed:
        return ["all", "constructed"]
    if entry is None:
        return ["all"]
    return ["all", f"type={entry['doc_type']}", f"split={entry['split']}", f"type={entry['doc_type']}|split={entry['split']}"]


def evaluate_corpus(gold_dir: Path, pred_dir: Path, *, manifest: dict | None, splits: set[str] | None, allow_unreviewed: bool, view: str, row_content_dir: Path | None, iou: float) -> dict:
    entries = {entry["id"]: entry for entry in (manifest or {}).get("entries", [])}
    counts: dict[str, Counter] = {}
    failures: dict[str, Counter] = {}
    content: dict[str, Counter] = {}
    docs: dict[str, int] = {}
    missing, failed_pages = [], []
    for gold_path in sorted(gold_dir.rglob("*.json")):
        gold = read_json(gold_path)
        if "pages" not in gold:
            continue
        doc_id = gold["document_id"]
        is_constructed = doc_id.startswith("constructed/")
        entry = entries.get(doc_id)
        if splits and not is_constructed and (entry is None or entry["split"] not in splits):
            continue
        pred_path = pred_dir / f"{doc_id}.json"
        if not pred_path.exists():
            pred_path = pred_dir / f"{Path(doc_id).name}.json"
        if not pred_path.exists():
            missing.append(doc_id)
            continue
        gold_doc = gold_from_json(gold, allow_unreviewed=allow_unreviewed)
        pred_doc = prediction_from_evidence(read_json(pred_path), doc_id)
        result = evaluate_doc(gold_doc, pred_doc, view=view, iou_threshold=iou)
        failed_pages.extend(page for page in result.pages if page["failures"])
        row_counts = None
        if row_content_dir is not None:
            rc_path = row_content_dir / f"{doc_id}.json"
            if rc_path.exists():
                rc = read_json(rc_path)
                row_counts = evaluate_row_content([t["rows"] for t in rc["tables"]], pred_doc)
        for key in _group_keys(entry, is_constructed):
            counts.setdefault(key, Counter()).update(result.counts)
            failures.setdefault(key, Counter()).update(result.failures)
            docs[key] = docs.get(key, 0) + 1
            if row_counts is not None:
                content.setdefault(key, Counter()).update(row_counts)
    report = {"stamp": STAMP, "allow_unreviewed": allow_unreviewed, "view": view, "iou_threshold": iou, "splits": sorted(splits) if splits else "all", "missing_predictions": missing, "groups": {}}
    for key in sorted(counts):
        group = summarize(counts[key], failures[key])
        group["documents"] = docs[key]
        if key in content:
            group["row_content"] = summarize_row_content(content[key])
        report["groups"][key] = group
    report["failed_pages"] = failed_pages
    return report


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(report: dict, title: str = "") -> str:
    lines = [f"### {title}" if title else "### Table evaluation", "", f"_{report['stamp']}_ view=`{report['view']}` IoU>={report['iou_threshold']} splits={report['splits']}", ""]
    lines.append("| group | docs | pages | gold/pred tables | det P / R / F1 | header | cols | rows | cell F1 | amount@table | summary leak | multi-table pages | row-content P / R |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, g in report["groups"].items():
        def frac(item):
            return f"{item['num']}/{item['den']}"

        rc = g.get("row_content")
        lines.append(
            f"| {key} | {g['documents']} | {g['pages']} | {g['gold_tables']}/{g['pred_tables']} | {_fmt(g['detection']['precision'])} / {_fmt(g['detection']['recall'])} / {_fmt(g['detection']['f1'])} "
            f"| {frac(g['header_exact'])} | {frac(g['column_count_exact'])} | {frac(g['row_count_exact'])} | {_fmt(g['cell']['f1'])} | {frac(g['amount_recall_in_matched_table'])} "
            f"| {g['summary_leak']['leaked']}/{g['summary_leak']['summary_rows']} | {frac(g['multi_table_pages_fully_correct'])} | {'-' if rc is None else _fmt(rc['row_precision']) + ' / ' + _fmt(rc['row_recall'])} |"
        )
    lines += ["", "Failure taxonomy (pages with the failure), group `all`:", ""]
    fp = report["groups"].get("all", {}).get("failures_pages", {})
    lines.append(" ".join(f"{code}={fp.get(code, 0)}" for code in FAILURE_CODES))
    unscored = report["groups"].get("all", {}).get("unscored_pred_tables")
    if unscored:
        lines.append(f"\nPredicted tables on pages without gold (not scored, gold not exhaustive): {unscored}")
    if report["missing_predictions"]:
        lines.append(f"\nMissing predictions: {len(report['missing_predictions'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--gold-dir", type=Path)
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--splits", default="")
    parser.add_argument("--row-content-dir", type=Path)
    parser.add_argument("--view", choices=("line_items", "any"), default="line_items")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if args.gold and args.prediction:
        result = evaluate(read_json(args.gold), read_json(args.prediction), allow_unreviewed=args.allow_unreviewed, view=args.view)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if not (args.gold_dir and args.pred_dir):
        parser.error("use --gold with --prediction, or --gold-dir with --pred-dir")
    manifest = read_json(args.manifest) if args.manifest else None
    splits = {s for s in args.splits.split(",") if s} or None
    report = evaluate_corpus(args.gold_dir, args.pred_dir, manifest=manifest, splits=splits, allow_unreviewed=args.allow_unreviewed, view=args.view, row_content_dir=args.row_content_dir, iou=args.iou)
    markdown = render_markdown(report, args.title)
    print(markdown)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
