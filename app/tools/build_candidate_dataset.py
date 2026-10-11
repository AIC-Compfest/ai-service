from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from app.services.candidates.pipeline import CandidatePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic candidate CSV datasets.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="config/candidates", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    for manifest in rows[:args.limit]:
        evidence = json.loads(Path(manifest["evidence_json_path"]).read_text(encoding="utf-8"))
        evidence = restrict_pages(evidence, manifest.get("page_start"), manifest.get("page_end"))
        generated = CandidatePipeline(args.config).generate(manifest["document_id"], manifest["doc_type"], evidence, args.out.name)
        for row in generated:
            grouped[f"{manifest['doc_type']}_{row['m_cand_type']}"].append(row)
    for name, records in grouped.items():
        records.sort(key=lambda row: (row["m_document_id"], row["m_page"], row["m_y1"], row["m_x1"], row["m_cand_type"], row["m_candidate_id"]))
        columns = sorted({key for record in records for key in record}, key=_column_order)
        with (args.out / f"candidates_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    (args.out / "run_report.json").write_text(json.dumps({"documents": len(rows[:args.limit]), "candidate_counts": {key: len(value) for key, value in grouped.items()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def restrict_pages(evidence: dict, page_start: str | None, page_end: str | None) -> dict:
    """Return a shallow evidence copy containing only the requested logical segment."""
    if not page_start and not page_end:
        return evidence
    start = int(page_start) if page_start else min(int(page.get("page_number", 1)) for page in evidence.get("pages", []))
    end = int(page_end) if page_end else max(int(page.get("page_number", 1)) for page in evidence.get("pages", []))
    if start > end:
        raise ValueError(f"page_start ({start}) cannot be greater than page_end ({end})")
    filtered = dict(evidence)
    filtered["pages"] = [page for page in evidence.get("pages", []) if start <= int(page.get("page_number", 1)) <= end]
    return filtered


def _column_order(name: str) -> tuple[int, str]:
    if name.startswith("m_"):
        return 0, name
    if name.startswith("ctx_"):
        return 2, name
    if name.startswith("label__") or name.startswith("label_"):
        return 3, name
    return 1, name


if __name__ == "__main__":
    main()