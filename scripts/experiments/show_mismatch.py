"""List matched tables whose header/columns/rows differ from gold, for one prediction directory (debug aid).

    .\\.venv\\Scripts\\python.exe scripts\\experiments\\show_mismatch.py out/pred/p3/token_grid_none --splits dev [--limit 12] [--what header]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))

from app.tools.table_eval.metrics import match_tables  # noqa: E402
from app.tools.table_eval.model import gold_from_json, prediction_from_evidence, read_json  # noqa: E402
from app.tools.table_eval.normalize import norm_cell  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--what", choices=("header", "cols", "rows", "unmatched", "extra"), default="header")
    args = parser.parse_args()
    assert "holdout" not in args.splits
    splits = set(args.splits.split(","))
    manifest = {e["id"]: e for e in read_json(AI_ROOT / "gold" / "manifest.json")["entries"]}
    shown = 0
    for path in sorted((AI_ROOT / "gold" / "regression").rglob("*.json")):
        gold_json = read_json(path)
        doc_id = gold_json["document_id"]
        if manifest[doc_id]["split"] not in splits:
            continue
        pred_path = args.pred_dir / f"{doc_id}.json"
        if not pred_path.exists():
            continue
        gold = gold_from_json(gold_json, allow_unreviewed=True)
        pred = prediction_from_evidence(read_json(pred_path), doc_id)
        for page in sorted(gold.pages()):
            g_tables, p_tables = gold.on_page(page), pred.on_page(page)
            matches = match_tables(g_tables, p_tables, 0.5)
            matched_g = {id(g) for g, _ in matches}
            matched_p = {id(p) for _, p in matches}
            if args.what == "unmatched":
                for g in g_tables:
                    if id(g) not in matched_g and shown < args.limit:
                        shown += 1
                        print(f"## {doc_id} p{page} UNMATCHED gold bbox={[round(v, 2) for v in g.bbox]} headers={g.headers}")
                        for p in p_tables:
                            print(f"   pred bbox={[round(v, 2) for v in p.bbox]} headers={p.headers} rows={len(p.rows)}")
            elif args.what == "extra":
                for p in p_tables:
                    if id(p) not in matched_p and shown < args.limit:
                        shown += 1
                        print(f"## {doc_id} p{page} EXTRA pred bbox={[round(v, 2) for v in p.bbox]} headers={p.headers} rows={len(p.rows)} first={p.rows[0] if p.rows else None}")
            else:
                for g, p in matches:
                    bad = (
                        [norm_cell(h) for h in g.headers] != [norm_cell(h) for h in p.headers] if args.what == "header" else
                        g.n_cols != p.n_cols if args.what == "cols" else len(g.rows) != len(p.rows)
                    )
                    if bad and shown < args.limit:
                        shown += 1
                        print(f"## {doc_id} p{page}\n  gold hdr {g.headers} rows={len(g.rows)} first={g.rows[0] if g.rows else None}\n  pred hdr {p.headers} rows={len(p.rows)} first={p.rows[0] if p.rows else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
