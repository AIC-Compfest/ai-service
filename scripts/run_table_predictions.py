"""Produce prediction evidence for the evaluator.

    .\\.venv\\Scripts\\python.exe scripts\\run_table_predictions.py --config router_none --out out/pred/router_none
    .\\.venv\\Scripts\\python.exe scripts\\run_table_predictions.py --config api_flag_on --splits dev,val --out out/pred/api_on
    .\\.venv\\Scripts\\python.exe scripts\\run_table_predictions.py --config router_none --constructed vector --out out/pred_fx/router_vector

Real documents come from gold/manifest.json (GT-bearing PDFs only). ``--constructed vector|phrase`` runs the
deterministic constructed fixtures instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from app.tools.table_eval.model import read_json, write_json  # noqa: E402
from app.tools.table_eval.runner import CONFIGS, predict_fixture, predict_pdf  # noqa: E402
from app.tools.table_eval.synthetic import build_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=AI_ROOT / "gold" / "manifest.json")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset")))
    parser.add_argument("--splits", default="dev,val,holdout", help="Comma list; use 'holdout' only for the end-of-phase run")
    parser.add_argument("--constructed", choices=("vector", "phrase"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="stop after N documents (0 = all)")
    parser.add_argument("--max-pages", type=int, default=0, help="only documents with at most N pages (0 = any)")
    parser.add_argument("--types", default="", help="comma list of doc types, e.g. INVOICE,POD (empty = all)")
    args = parser.parse_args()

    started = time.time()
    count = 0
    if args.constructed:
        for fixture in build_all():
            write_json(args.out / f"{fixture.name}.json", predict_fixture(fixture.pdf, args.config, args.constructed, f"constructed/{fixture.name}", args.seed))
            count += 1
    else:
        wanted = set(args.splits.split(","))
        for entry in read_json(args.manifest)["entries"]:
            if not entry["has_gt"] or entry["source_kind"] != "pdf" or entry["split"] not in wanted:
                continue
            if (args.max_pages and entry["pages"] > args.max_pages) or (args.types and entry["doc_type"] not in args.types.split(",")):
                continue
            if args.limit and count >= args.limit:
                break
            write_json(args.out / f"{entry['id']}.json", predict_pdf(args.dataset_root / entry["path"], args.config, entry["doc_type"], entry["id"]))
            count += 1
    print(f"{args.config}: {count} document(s) -> {args.out} in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
