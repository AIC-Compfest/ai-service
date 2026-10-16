"""Compare evaluator reports side by side (one row per report) for a chosen group.

    compare_reports.py --group all --label token_grid=out/reports/baseline/token_grid_none.json --label router=...

Prints markdown with numerators/denominators and the F1..F11 page counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _frac(item: dict) -> str:
    return f"{item['num']}/{item['den']}" + ("" if item["rate"] is None else f" ({item['rate']:.2f})")


def _num(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="all")
    parser.add_argument("--label", action="append", required=True, help="name=path/to/report.json")
    args = parser.parse_args()
    header = "| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for entry in args.label:
        name, path = entry.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        g = report["groups"].get(args.group)
        if g is None:
            print(f"| {name} | (group {args.group} absent) |")
            continue
        det = g["detection"]
        codes = " ".join(str(g["failures_pages"][f"F{i}"]) for i in range(1, 12))
        print(
            f"| {name} | {g['gold_tables']}/{g['pred_tables']} | {_num(det['precision'])} / {_num(det['recall'])} / {_num(det['f1'])} | {_frac(g['header_exact'])} | {_frac(g['column_count_exact'])} "
            f"| {_frac(g['row_count_exact'])} | {_num(g['row_sequence']['f1'])} | {_num(g['cell']['f1'])} | {_frac(g['amount_recall_in_matched_table'])} | {_frac(g['amount_recall_on_page'])} "
            f"| {g['summary_leak']['leaked']}/{g['summary_leak']['summary_rows']} | {_frac(g['multi_table_pages_fully_correct'])} | {codes} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
