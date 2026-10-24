"""Write the v3/v4 out-of-fold evaluation comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["doc_type"], row["field"]): row for row in rows}


def value(row: dict[str, str] | None, key: str) -> str:
    return "" if row is None else row.get(key, "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=ROOT / "models/all_v3")
    parser.add_argument("--v4", type=Path, default=ROOT / "models/all_v4")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "reports/out/all_types/candidates_v4",
    )
    args = parser.parse_args()

    v3_eval = index(read_csv(args.v3 / "evaluation/evaluation_fields.csv"))
    v4_eval = index(read_csv(args.v4 / "evaluation/evaluation_fields.csv"))
    # This field was explicitly removed from the v4 registry by the owner.
    v3_eval.pop(("RATE_AGREEMENT", "destination"), None)
    v4_eval.pop(("RATE_AGREEMENT", "destination"), None)
    v3_decisions = index(read_csv(args.v3.parent.parent / "reports/out/all_types/candidates_v3/decisions/summary.csv"))
    v4_decisions = index(read_csv(args.out / "decisions/summary.csv"))

    keys = sorted(set(v3_eval) | set(v4_eval))
    columns = [
        "doc_type", "field",
        "v3_top1_hit", "v4_top1_hit", "v3_ap", "v4_ap", "v3_auc", "v4_auc",
        "v3_accepted_precision", "v4_accepted_precision",
        "v3_coverage", "v4_coverage", "v3_decision", "v4_decision",
        "v3_docs_without_positive", "v4_docs_without_positive",
        "v3_answerable_docs", "v4_answerable_docs",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for key in keys:
        e3, e4 = v3_eval.get(key), v4_eval.get(key)
        d3, d4 = v3_decisions.get(key), v4_decisions.get(key)
        rows.append({
            "doc_type": key[0], "field": key[1],
            "v3_top1_hit": value(e3, "top1_hit"), "v4_top1_hit": value(e4, "top1_hit"),
            "v3_ap": value(e3, "ap"), "v4_ap": value(e4, "ap"),
            "v3_auc": value(e3, "auc"), "v4_auc": value(e4, "auc"),
            "v3_accepted_precision": value(d3, "accepted_precision"),
            "v4_accepted_precision": value(d4, "accepted_precision"),
            "v3_coverage": value(d3, "coverage"), "v4_coverage": value(d4, "coverage"),
            "v3_decision": value(d3, "mode"), "v4_decision": value(d4, "mode"),
            "v3_docs_without_positive": value(e3, "docs_without_positive"),
            "v4_docs_without_positive": value(e4, "docs_without_positive"),
            "v3_answerable_docs": value(e3, "docs_with_positive"),
            "v4_answerable_docs": value(e4, "docs_with_positive"),
        })
    with (args.out / "comparison_v3_v4.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(row: dict[str, str], k: str) -> str:
        raw = row.get(k, "")
        try:
            return f"{float(raw):.3f}"
        except (ValueError, TypeError):
            return raw or "—"

    lines = [
        "# Candidate model comparison: v3 vs v4", "",
        "Metrics use grouped out-of-fold predictions. `top1_hit`, AP and AUC are from the model evaluation; accepted precision and coverage are from each run's recorded decision summary. Counts are answerable documents. The small, template-grouped dataset makes per-field estimates noisy.", "",
        "| Field | n v3/v4 | top1 v3 → v4 | AP v3 → v4 | AUC v3 → v4 | accepted precision v3 → v4 | coverage v3 → v4 | decision v3 → v4 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        field = f"{row['doc_type']}.{row['field']}"
        n = f"{row['v3_answerable_docs'] or '—'}/{row['v4_answerable_docs'] or '—'}"
        lines.append(
            f"| {field} | {n} | {fmt(row, 'v3_top1_hit')} → {fmt(row, 'v4_top1_hit')} | "
            f"{fmt(row, 'v3_ap')} → {fmt(row, 'v4_ap')} | {fmt(row, 'v3_auc')} → {fmt(row, 'v4_auc')} | "
            f"{fmt(row, 'v3_accepted_precision')} → {fmt(row, 'v4_accepted_precision')} | "
            f"{fmt(row, 'v3_coverage')} → {fmt(row, 'v4_coverage')} | "
            f"{row['v3_decision'] or '—'} → {row['v4_decision'] or '—'} |"
        )
    lines.extend([
        "", "`RATE_AGREEMENT.destination` is intentionally excluded from v4 and is not trained, evaluated or decided. The two POD jabatan fields remain excluded by design.",
        "",
    ])
    (args.out / "comparison_v3_v4.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out / 'comparison_v3_v4.csv'}")
    print(f"Wrote {args.out / 'comparison_v3_v4.md'}")


if __name__ == "__main__":
    main()
