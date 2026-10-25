"""Evaluate the Docling pipeline on DATASET/Dataset: per-file evidence JSON + accuracy against the ground-truth JSONs.

Outputs (default ``out/eval_docling``):
  json/<category>/<file>.json   full pipeline evidence per file (what the API would return)
  results.json                  per-file metrics + aggregates
  report.md                     human-readable summary

Accuracy is *extraction* accuracy of Docling (text lines + tables), not of the downstream field extractors:
  value_recall_strict   share of ground-truth leaf values found as one contiguous string in the extracted text
                        (case / spacing / punctuation insensitive, so "38.850.000" == 38850000)
  value_recall_lenient  share of values whose every word is present somewhere in the extracted text (tolerates
                        wrapped lines and reading-order differences)
  table_recall          values that belong to ground-truth line-item arrays found inside Docling table cells
  classification        document type of the page/document vs the dataset folder
Images (PNG/JPG) have no ground truth; they are reported with extraction statistics and classification only.

Resumable: files whose JSON already exists are not reprocessed (use --force).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATASET = Path(os.environ.get("VERIFLOW_DATASET_ROOT", Path(__file__).resolve().parents[2] / "DATASET" / "Dataset"))
CATEGORIES = {"invoice": "INVOICE", "PoD": "POD", "rate card": "RATE_AGREEMENT", "surat jalan": "SURAT_JALAN"}
META_KEYS = {"file", "file_name", "variant_name", "template", "template_source", "template_id", "variation", "document_type", "document_name", "id", "language"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def words(text: str) -> set[str]:
    return {w for w in re.split(r"[^0-9a-z]+", text.casefold()) if w}


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    elif value is not None and not isinstance(value, bool):
        yield path, value


def leaf_text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_ground_truth(dataset: Path) -> dict[str, dict]:
    """basename -> ground-truth payload, across every ground_truth*.json (schemas differ per document type)."""
    found: dict[str, dict] = {}

    def visit(node):
        if isinstance(node, dict):
            name = next((node[k] for k in ("file", "file_name") if isinstance(node.get(k), str) and node[k].lower().endswith(".pdf")), None)
            if name:
                payload = node.get("ground_truth") or node.get("fields") or {k: v for k, v in node.items() if k not in META_KEYS}
                found.setdefault(name, payload)
                return
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    for path in sorted(dataset.rglob("ground_truth*.json")):
        visit(json.loads(path.read_text(encoding="utf-8")))
    return found


def evidence_text(evidence: dict) -> tuple[str, str]:
    """(all extracted text, table cell text)"""
    lines, cells = [], []
    for page in evidence["pages"]:
        lines.extend(line["text"] for line in page.get("lines", []))
        for table in page.get("tables", []):
            cells.extend(table.get("headers", []))
            for row in list(table.get("rows", [])) + list(table.get("summary_rows", [])):
                cells.extend(str(cell) for cell in row)
    return "\n".join(lines), "\n".join(cells)


def score_document(evidence: dict, truth: dict) -> dict:
    text, table_text = evidence_text(evidence)
    flat, vocab = norm(text), words(text)
    table_flat, table_vocab = norm(table_text), words(table_text)
    total = strict = lenient = skipped = 0
    t_total = t_found = 0
    misses = []
    per_field: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, value in leaves(truth):
        raw = leaf_text(value)
        key = norm(raw)
        if len(key) < 3 or (isinstance(value, str) and ISO_DATE.match(raw)):
            skipped += 1  # too short to be meaningful, or an ISO date whose printed form differs
            continue
        total += 1
        hit_strict = key in flat
        hit_lenient = hit_strict or (bool(words(raw)) and words(raw) <= vocab)
        strict += hit_strict
        lenient += hit_lenient
        top = re.sub(r"\[\d+\]", "", path).split(".")[0]
        per_field[top][0] += 1
        per_field[top][1] += hit_strict
        if not hit_strict:
            misses.append({"path": path, "value": raw, "lenient": hit_lenient})
        if "[" in path:  # inside a list of objects = line items / goods rows
            t_total += 1
            t_found += key in table_flat or (bool(words(raw)) and words(raw) <= table_vocab)
    return {
        "gt_values": total, "gt_skipped": skipped,
        "found_strict": strict, "found_lenient": lenient,
        "value_recall_strict": round(strict / total, 4) if total else None,
        "value_recall_lenient": round(lenient / total, 4) if total else None,
        "table_values": t_total, "table_found": t_found,
        "table_recall": round(t_found / t_total, 4) if t_total else None,
        "per_field": {k: {"values": v[0], "found": v[1]} for k, v in per_field.items()},
        "misses": misses[:40],
    }


def discover(dataset: Path, category: str | None):
    for folder, doc_type in CATEGORIES.items():
        if category and category.casefold() not in (folder.casefold(), doc_type.casefold()):
            continue
        for path in sorted((dataset / folder).rglob("*")):
            if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"} and path.is_file():
                yield folder, doc_type, path


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def aggregate(rows: list[dict], key: str):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=Path("out/eval_docling"))
    parser.add_argument("--category", help="invoice | PoD | rate card | surat jalan (or doc type)")
    parser.add_argument("--kind", choices=["pdf", "image", "all"], default="all")
    parser.add_argument("--limit", type=int, help="max files per category")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from app.services.pipeline import Phase5DocumentPipeline

    truth = load_ground_truth(args.dataset_dir)
    pipeline = Phase5DocumentPipeline()
    results: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for folder, doc_type, path in discover(args.dataset_dir, args.category):
        is_pdf = path.suffix.lower() == ".pdf"
        if (args.kind == "pdf" and not is_pdf) or (args.kind == "image" and is_pdf):
            continue
        if args.limit and counts[folder] >= args.limit:
            continue
        counts[folder] += 1
        rel = path.relative_to(args.dataset_dir / folder).as_posix()
        subset = Path(rel).parent.as_posix() if "/" in rel else ""
        target = args.out / "json" / folder / (rel.replace("/", "__") + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {"category": doc_type, "folder": folder, "file": path.name, "subset": subset, "kind": "pdf" if is_pdf else "image"}
        try:
            if target.exists() and not args.force:
                evidence, elapsed = json.loads(target.read_text(encoding="utf-8")), None
            else:
                start = time.perf_counter()
                evidence = pipeline.process_document(path, filename=path.name, document_id=f"EVAL-{folder.replace(' ', '')}-{path.stem}"[:80])
                elapsed = round(time.perf_counter() - start, 1)
                evidence.pop("raw_bytes", None)
                for page in evidence["pages"]:
                    page.pop("page_image", None)
                target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            pages = evidence["pages"]
            page_types = [p["classification"]["document_type"] for p in pages]
            row.update({
                "status": "ok", "seconds": elapsed, "pages": len(pages),
                "engines": sorted({p["engine_used"] for p in pages}),
                "lines": sum(len(p["lines"]) for p in pages),
                "avg_confidence": round(sum(p["average_confidence"] for p in pages) / max(len(pages), 1), 3),
                "tables": sum(len(p["tables"]) for p in pages),
                "doc_type_predicted": evidence["classification"]["document_type"],
                "doc_type_correct": evidence["classification"]["document_type"] == doc_type,
                "page_type_accuracy": round(sum(t == doc_type for t in page_types) / max(len(page_types), 1), 3),
                "json": target.as_posix(),
            })
            if is_pdf and path.name in truth:
                row["accuracy"] = score_document(evidence, truth[path.name])
                for key in ("value_recall_strict", "value_recall_lenient", "table_recall"):
                    row[key] = row["accuracy"][key]
        except Exception as exc:  # one bad file must not stop the run
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-600:]})
        results.append(row)
        print(f"[{len(results):3d}] {folder:12s} {rel:60s} {row['status']} {row.get('seconds')}s lines={row.get('lines')} tables={row.get('tables')} strict={row.get('value_recall_strict')}", flush=True)

    write_reports(args.out, results)


def write_reports(out: Path, results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    scored = [r for r in ok if "accuracy" in r]
    groups: dict[str, dict] = {}
    for label, rows in [("ALL", ok)] + [(c, [r for r in ok if r["category"] == c]) for c in CATEGORIES.values()]:
        sub = [r for r in rows if "accuracy" in r]
        gt = sum(r["accuracy"]["gt_values"] for r in sub)
        groups[label] = {
            "files": len(rows), "errors": len([r for r in results if r["status"] == "error" and (label == "ALL" or r["category"] == label)]),
            "files_scored": len(sub),
            "doc_type_accuracy": round(sum(r["doc_type_correct"] for r in rows) / len(rows), 4) if rows else None,
            "page_type_accuracy": aggregate(rows, "page_type_accuracy"),
            "gt_values": gt,
            "value_recall_strict_micro": round(sum(r["accuracy"]["found_strict"] for r in sub) / gt, 4) if gt else None,
            "value_recall_lenient_micro": round(sum(r["accuracy"]["found_lenient"] for r in sub) / gt, 4) if gt else None,
            "value_recall_strict_macro": aggregate(sub, "value_recall_strict"),
            "table_recall_micro": (round(sum(r["accuracy"]["table_found"] for r in sub) / sum(r["accuracy"]["table_values"] for r in sub), 4) if sum(r["accuracy"]["table_values"] for r in sub) else None),
            "avg_confidence": aggregate(rows, "avg_confidence"),
            "avg_tables_per_file": round(sum(r["tables"] for r in rows) / len(rows), 2) if rows else None,
            "avg_seconds": aggregate(rows, "seconds"),
        }
    by_kind = {k: groups_for(ok, k) for k in ("pdf", "image")}
    (out / "results.json").write_text(json.dumps({"aggregate": groups, "by_kind": by_kind, "files": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Docling evaluation on DATASET/Dataset", "", f"Files: {len(results)} ({len(ok)} ok, {len(results) - len(ok)} error), scored against ground truth: {len(scored)}", "",
             "| Category | files | scored | doc-type acc | page-type acc | value recall (strict) | value recall (lenient) | table recall | avg conf | tables/file |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, g in groups.items():
        lines.append(f"| {label} | {g['files']} | {g['files_scored']} | {pct(g['doc_type_accuracy'])} | {pct(g['page_type_accuracy'])} | {pct(g['value_recall_strict_micro'])} | {pct(g['value_recall_lenient_micro'])} | {pct(g['table_recall_micro'])} | {g['avg_confidence']} | {g['avg_tables_per_file']} |")
    lines += ["", "## By input kind (PDF text layer vs image/RapidOCR)", "", "| Kind | files | doc-type acc | avg conf | lines/file | tables/file | strict recall (PDF only) |", "|---|---|---|---|---|---|---|"]
    for kind, g in by_kind.items():
        lines.append(f"| {kind} | {g['files']} | {pct(g['doc_type_accuracy'])} | {g['avg_confidence']} | {g['avg_lines']} | {g['avg_tables']} | {pct(g['strict'])} |")
    lines += ["", "## Per PDF subset (clean / ori / kotor ...)", "", "| Category | subset | files | strict recall | lenient recall |", "|---|---|---|---|---|"]
    for (cat, sub), rows in sorted({(r["category"], r["subset"]): [x for x in scored if (x["category"], x["subset"]) == (r["category"], r["subset"])] for r in scored}.items()):
        total = sum(r["accuracy"]["gt_values"] for r in rows)
        lines.append(f"| {cat} | {sub or '-'} | {len(rows)} | {pct(sum(r['accuracy']['found_strict'] for r in rows) / total if total else None)} | {pct(sum(r['accuracy']['found_lenient'] for r in rows) / total if total else None)} |")
    worst = sorted(scored, key=lambda r: r["value_recall_strict"])[:10]
    lines += ["", "## Lowest 10 files (strict recall)", "", "| file | subset | recall | missed examples |", "|---|---|---|---|"]
    for r in worst:
        examples = "; ".join(f"{m['path']}={m['value']!r}" for m in r["accuracy"]["misses"][:3]).replace("|", "/")
        lines.append(f"| {r['file']} | {r['subset'] or '-'} | {pct(r['value_recall_strict'])} | {examples} |")
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {r['folder']}/{r['file']}: {r['error']}" for r in errors]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:14]))


def groups_for(rows: list[dict], kind: str) -> dict:
    sub = [r for r in rows if r["kind"] == kind]
    scored = [r for r in sub if "accuracy" in r]
    gt = sum(r["accuracy"]["gt_values"] for r in scored)
    return {
        "files": len(sub),
        "doc_type_accuracy": round(sum(r["doc_type_correct"] for r in sub) / len(sub), 4) if sub else None,
        "avg_confidence": aggregate(sub, "avg_confidence"),
        "avg_lines": round(sum(r["lines"] for r in sub) / len(sub), 1) if sub else None,
        "avg_tables": round(sum(r["tables"] for r in sub) / len(sub), 2) if sub else None,
        "strict": round(sum(r["accuracy"]["found_strict"] for r in scored) / gt, 4) if gt else None,
    }


if __name__ == "__main__":
    main()
