"""Stress test: many varied, seeded PDF pages (1 to 4 tables each) with exact gold; PyMuPDF alone vs the full router.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\stress_tables.py [--pages 240] [--first-seed 0] [--mode vector|phrase]
                                                          [--configs pymupdf,token_grid_none,router_none] [--out out/stress] [--show-failures N]

Prints one table per layout feature (tag) with detection, header, column and row accuracy and the share of pages that are
completely correct (no failure code F1..F11). The pages are built by construction (app/tools/table_eval/stress.py): the numbers
show WHICH layouts break an extractor, not how common they are in production.  Gold is exact but no human reviewed it.
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

from app.tools.table_eval.metrics import evaluate_doc, summarize  # noqa: E402
from app.tools.table_eval.model import gold_from_json, prediction_from_evidence  # noqa: E402
from app.tools.table_eval.runner import predict_fixture  # noqa: E402
from app.tools.table_eval.stress import build_stress  # noqa: E402


def _pct(item: dict) -> str:
    return "n/a" if item["den"] == 0 else f"{item['num']}/{item['den']} ({item['num'] / item['den']:.0%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=int, default=240)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--mode", choices=("vector", "phrase"), default="vector")
    parser.add_argument("--configs", default="pymupdf,token_grid_none,router_none")
    parser.add_argument("--out", type=Path, default=AI_ROOT / "out" / "stress")
    parser.add_argument("--show-failures", type=int, default=0, help="print N failing page names per config for the worst tag")
    args = parser.parse_args()

    pages = build_stress(args.pages, args.first_seed)
    configs = args.configs.split(",")
    gold_dir = args.out / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (gold_dir / f"{page.fixture.name}.json").write_text(json.dumps(page.fixture.gold), encoding="utf-8")
        (gold_dir / f"{page.fixture.name}.pdf").write_bytes(page.fixture.pdf)

    results: dict[str, dict[str, tuple[Counter, Counter, int, int]]] = {}
    failing: dict[str, dict[str, list[str]]] = {}
    for config in configs:
        per_tag: dict[str, list] = {}
        for page in pages:
            gold = gold_from_json(page.fixture.gold, allow_unreviewed=True)
            prediction = predict_fixture(page.fixture.pdf, config, args.mode, f"constructed/{page.fixture.name}", 0)
            pred_dir = args.out / f"pred_{config}_{args.mode}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / f"{page.fixture.name}.json").write_text(json.dumps(prediction), encoding="utf-8")
            result = evaluate_doc(gold, prediction_from_evidence(prediction, gold.document_id), view="line_items")
            clean = not any(p["failures"] for p in result.pages)
            for tag in ["all"] + page.tags:
                entry = per_tag.setdefault(tag, [Counter(), Counter(), 0, 0])
                entry[0].update(result.counts)
                entry[1].update(result.failures)
                entry[2] += 1
                entry[3] += int(clean)
            if not clean:
                failing.setdefault(config, {}).setdefault("all", []).append(page.fixture.name)
        results[config] = {tag: tuple(v) for tag, v in per_tag.items()}

    tags = sorted(results[configs[0]], key=lambda t: (t != "all", t))
    print(f"# Stress: {len(pages)} pages, mode={args.mode}, seeds {args.first_seed}..{args.first_seed + args.pages - 1} (exact gold, not human reviewed)\n")
    for tag in tags:
        print(f"## tag: {tag}")
        print("| config | pages | gold tables | det P / R / F1 | header exact | cols exact | rows exact | cell F1 | pages fully correct |")
        print("|---|---|---|---|---|---|---|---|---|")
        for config in configs:
            counts, failures, n_pages, clean = results[config].get(tag, (Counter(), Counter(), 0, 0))
            if not n_pages:
                continue
            s = summarize(counts, failures)
            det = s["detection"]
            f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"| {config} | {n_pages} | {s['gold_tables']} | {f(det['precision'])} / {f(det['recall'])} / {f(det['f1'])} | {_pct(s['header_exact'])} | {_pct(s['column_count_exact'])} | {_pct(s['row_count_exact'])} | {f(s['cell']['f1'])} | {clean}/{n_pages} ({clean / n_pages:.0%}) |")
        print()
    print("## failure codes (pages with the code), tag all")
    print("| config | " + " | ".join(f"F{i}" for i in range(1, 12)) + " |")
    print("|---|" + "---|" * 11)
    for config in configs:
        _, failures, _, _ = results[config]["all"]
        s = summarize(results[config]["all"][0], failures)
        print(f"| {config} | " + " | ".join(str(s["failures_pages"][f"F{i}"]) for i in range(1, 12)) + " |")
    print("\nF1 header wrong, F2 column count wrong, F3 row count/order wrong, F4 one table spans several gold tables, F5 one gold table split, F6 extra table, F7 page-wide table, F8 gold table missed, F9 summary leaked into items, F10 item taken as summary")
    if args.show_failures:
        for config in configs:
            print(f"\n{config}: first failing pages -> {failing.get(config, {}).get('all', [])[: args.show_failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
