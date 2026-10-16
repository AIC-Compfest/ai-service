"""Fase 2 experiment: PyMuPDF strategy / rejection options scored against structural gold (dev, then val).

    .\\.venv\\Scripts\\python.exe scripts\\experiments\\pymupdf_variants.py [--splits dev]

Thresholds are chosen on dev and CONFIRMED on val. Never pass holdout here.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))

import pymupdf  # noqa: E402

from app.services.candidates.context import build_page  # noqa: E402
from app.services.candidates.pipeline import table_evidence  # noqa: E402
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider  # noqa: E402
from app.tools.table_eval.metrics import evaluate_doc, summarize  # noqa: E402
from app.tools.table_eval.model import gold_from_json, prediction_from_evidence, read_json  # noqa: E402
from app.tools.table_eval.synthetic import build_all  # noqa: E402

VARIANTS = {
    "V0 lines->text, no filter": {},
    "V1 lines only": {"strategies": ["lines"]},
    "V2 lines->text, page<=0.6": {"max_page_area_ratio": 0.6},
    "V3 lines->text, page<=0.5": {"max_page_area_ratio": 0.5},
    "V4 lines->text, page<=0.6, hdr<=80": {"max_page_area_ratio": 0.6, "max_header_chars": 80},
    "V5 merge, page<=0.6, hdr<=80": {"max_page_area_ratio": 0.6, "max_header_chars": 80, "merge_strategies": True},
    "V6 merge, page<=0.4, hdr<=80": {"max_page_area_ratio": 0.4, "max_header_chars": 80, "merge_strategies": True},
    "V7 lines->text, page<=0.6, hdr<=80, rows>=2": {"max_page_area_ratio": 0.6, "max_header_chars": 80, "min_data_rows": 2},
}


def real_docs(splits: set[str]):
    dataset = Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset"))
    manifest = {e["id"]: e for e in read_json(AI_ROOT / "gold" / "manifest.json")["entries"]}
    for path in sorted((AI_ROOT / "gold" / "regression").rglob("*.json")):
        gold = read_json(path)
        entry = manifest[gold["document_id"]]
        if entry["split"] in splits:
            yield gold, pymupdf.open(dataset / entry["path"])


def fixture_docs():
    for fixture in build_all():
        yield fixture.gold, pymupdf.open(stream=fixture.pdf, filetype="pdf")


def run(docs, options: dict) -> dict:
    provider = PyMuPDFTableProvider(options)
    counts, failures = Counter(), Counter()
    for gold, document in docs:
        pages = []
        for number, page in enumerate(document, start=1):
            candidate = build_page({"page_number": number, "lines": [], "_source_page": page})
            pages.append({"page_number": number, "tables": [table_evidence(t) for t in provider.tables(candidate, None, gold["document_id"])]})
        result = evaluate_doc(gold_from_json(gold, allow_unreviewed=True), prediction_from_evidence({"pages": pages}, gold["document_id"]))
        counts.update(result.counts)
        failures.update(result.failures)
    return summarize(counts, failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="dev")
    args = parser.parse_args()
    splits = set(args.splits.split(","))
    assert "holdout" not in splits, "holdout must not be used for design"
    sets = {f"real:{','.join(sorted(splits))}": list(real_docs(splits)), "constructed:vector": list(fixture_docs())}
    for label, docs in sets.items():
        print(f"\n## {label} ({len(docs)} docs)")
        print("| variant | gold/pred | det P/R/F1 | header | cols | rows | cell F1 | amount@table | leak | F6 F7 F8 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, options in VARIANTS.items():
            m = run(docs, options)
            d = m["detection"]
            f = lambda i: f"{i['num']}/{i['den']}"  # noqa: E731
            fp = m["failures_pages"]
            print(f"| {name} | {m['gold_tables']}/{m['pred_tables']} | {d['precision'] or 0:.2f}/{d['recall'] or 0:.2f}/{d['f1'] or 0:.2f} | {f(m['header_exact'])} | {f(m['column_count_exact'])} | {f(m['row_count_exact'])} | {m['cell']['f1'] or 0:.2f} | {f(m['amount_recall_in_matched_table'])} | {m['summary_leak']['leaked']}/{m['summary_leak']['summary_rows']} | {fp['F6']} {fp['F7']} {fp['F8']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
