"""Print predicted vs gold tables for one constructed fixture (debug aid).

    .\\.venv\\Scripts\\python.exe scripts\\experiments\\show_fixture.py blank_column [--mode vector|phrase] [--config token_grid_none]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))

from app.tools.table_eval.model import evidence_from_gold  # noqa: E402
from app.tools.table_eval.runner import predict_fixture  # noqa: E402
from app.tools.table_eval.synthetic import build_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--mode", default="vector")
    parser.add_argument("--config", default="token_grid_none")
    args = parser.parse_args()
    fixture = build_fixture(args.name)
    pred = predict_fixture(fixture.pdf, args.config, args.mode, f"constructed/{args.name}")
    gold = evidence_from_gold(fixture.gold)
    for page_pred, page_gold in zip(pred["pages"], gold["pages"]):
        print(f"--- page {page_pred['page_number']}: gold {len(page_gold['tables'])} pred {len(page_pred['tables'])}")
        for label, tables in (("GOLD", page_gold["tables"]), ("PRED", page_pred["tables"])):
            for table in tables:
                bbox = [round(v, 3) for v in table["bbox_norm"]]
                print(f"{label} bbox={bbox} headers={table['headers']} rows={len(table['rows'])} summary={len(table.get('summary_rows', []))}")
                for row in table["rows"][:3]:
                    print("      ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
