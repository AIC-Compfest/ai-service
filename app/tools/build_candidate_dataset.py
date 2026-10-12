from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from app.services.candidates.pipeline import CandidatePipeline
from app.services.candidates.line_items import LineItemBuilder
from app.services.candidates.tables import TokenGridTableProvider
from app.services.candidates.training_schema import classify_column, load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic candidate CSV datasets.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="config/candidates", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    pipeline = CandidatePipeline(args.config)
    report = {"empty_pages": [], "cap_overflows": [], "timings_ms": {}}
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    for manifest in rows[:args.limit]:
        evidence = json.loads(Path(manifest["evidence_json_path"]).read_text(encoding="utf-8"))
        evidence = restrict_pages(evidence, manifest.get("page_start"), manifest.get("page_end"))
        generated = pipeline.generate(manifest["document_id"], manifest["doc_type"], evidence, args.out.name, manifest.get("vendor_group", "UNKNOWN"), manifest.get("template_group", "UNKNOWN"))
        for key in ("empty_pages", "cap_overflows"):
            report[key].extend(pipeline.last_report.get(key, []))
        report["timings_ms"].update(pipeline.last_report.get("timings_ms", {}))
        for row in generated:
            grouped[f"{manifest['doc_type']}_{row['m_cand_type']}"].append(row)
    registry = CandidatePipeline(args.config).registry
    for doc_type in sorted({row["doc_type"] for row in rows}):
        for cand_type in sorted({field.cand_type for field in registry.for_doc(doc_type) if field.cand_type != "table"}):
            grouped.setdefault(f"{doc_type}_{cand_type}", [])
    for name, records in grouped.items():
        records.sort(key=lambda row: (row["m_document_id"], row["m_page"], row["m_y1"], row["m_x1"], row["m_cand_type"], row["m_candidate_id"]))
        doc_type, cand_type = _split_dataset_name(name, registry)
        columns = sorted({key for record in records for key in record}, key=_column_order) or _empty_candidate_columns(doc_type, cand_type, registry)
        _write_records(args.out / f"candidates_{name}.csv", records, columns)
    _write_line_items(args, rows[:args.limit])
    _write_status_template(args.out, rows[:args.limit], registry)
    _write_feature_manifest(args.out, args.config, grouped, registry)
    report.update({"documents": len(rows[:args.limit]), "candidate_counts": {key: len(value) for key, value in grouped.items()}})
    (args.out / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_line_items(args, manifests):
    by_type = {}
    for manifest in manifests:
        evidence = json.loads(Path(manifest["evidence_json_path"]).read_text(encoding="utf-8"))
        evidence = restrict_pages(evidence, manifest.get("page_start"), manifest.get("page_end"))
        provider = TokenGridTableProvider(args.config / "line_item_aliases.yaml", args.config / "abbreviations.yaml")
        tables = []
        from app.services.candidates.context import build_page
        for page_data in evidence.get("pages", []):
            tables.extend(provider.tables(build_page(page_data), manifest["doc_type"], manifest["document_id"]))
        result = LineItemBuilder(args.config).build(manifest["document_id"], manifest["doc_type"], tables)
        by_type.setdefault(manifest["doc_type"], {"columns": [], "cells": [], "rows": [], "previews": []})
        for key in by_type[manifest["doc_type"]]:
            by_type[manifest["doc_type"]][key].extend(result[key])
    preview_path = args.out / "li_preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as preview_handle:
        for doc_type in sorted({manifest["doc_type"] for manifest in manifests}):
            data = by_type.get(doc_type, {"columns": [], "cells": [], "rows": [], "previews": []})
            for key, records in (("columns", data["columns"]), ("cells", data["cells"]), ("rows", data["rows"])):
                _write_records(args.out / f"li_{key}_{doc_type}.csv", records)
            for preview in data["previews"]:
                preview_handle.write(json.dumps(preview, ensure_ascii=False, sort_keys=True) + "\n")


def _write_status_template(output, manifests, registry):
    records = []
    for manifest in manifests:
        for field in registry.for_doc(manifest["doc_type"]):
            records.append({"m_document_id": manifest["document_id"], "m_doc_type": manifest["doc_type"], "field": field.field, "status": "", "truth_text": ""})
    _write_records(output / "doc_field_status_template.csv", records)


def _write_feature_manifest(output, config_dir, grouped, registry):
    policy = load_policy(config_dir)
    manifest = {}
    for records in grouped.values():
        for row in records:
            for name, value in row.items():
                role, train_policy = classify_column(name, policy)
                manifest.setdefault(name, {"name": name, "role": role, "train_policy": train_policy, "dtype": "bool" if isinstance(value, bool) else "float" if isinstance(value, float) else "int" if isinstance(value, int) else "cat" if isinstance(value, str) else "str", "nullable": value == "", "range": None, "applies_to": ["*"], "description": "Generated candidate column"})
    (output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training = {}
    doc_types = sorted({field.doc_type for field in registry.fields})
    for doc_type in doc_types:
        for field in registry.for_doc(doc_type):
            keep = [name for name, spec in manifest.items() if spec["train_policy"] in {"keep", "ablate"} and spec["dtype"] in {"bool", "int", "float", "cat"}]
            ablate = [name for name, spec in manifest.items() if spec["train_policy"] == "ablate"]
            dropped = [name for name, spec in manifest.items() if spec["train_policy"] not in {"keep", "ablate"}]
            training[f"{doc_type}.{field.field}"] = {"source_csv": f"candidates_{doc_type}_{field.cand_type}.csv", "label_column": f"label__{field.field}", "feature_columns": keep, "ablate_columns": ablate, "dropped_columns": dropped, "group_column": "m_template_group"}
    (output / "training_schema.json").write_text(json.dumps(training, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "training_policy.yaml").write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")


def _write_records(path, records, columns=None):
    columns = columns or sorted({key for record in records for key in record}, key=_column_order)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["m_document_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _empty_candidate_columns(doc_type, cand_type, registry):
    columns = ["m_schema_version", "m_run_id", "m_candidate_id", "m_doc_type", "m_document_id", "m_vendor_group", "m_template_group", "m_page", "m_cand_type", "m_raw_text", "m_parsed_value", "m_parsed_norm", "m_x1", "m_y1", "m_x2", "m_y2", "m_line_idx", "m_tok_start", "m_tok_end", "m_overlap_types", "ctx_left", "ctx_right", "ctx_above", "ctx_below", "ctx_line", "ctx_sent", "label_status", "label_notes"]
    columns.extend(f"label__{field.field}" for field in registry.field_specs_for_type(doc_type, cand_type))
    return sorted(columns, key=_column_order)


def _split_dataset_name(name, registry):
    for doc_type in sorted({field.doc_type for field in registry.fields}, key=len, reverse=True):
        prefix = f"{doc_type}_"
        if name.startswith(prefix):
            return doc_type, name[len(prefix):]
    return name, "text"


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
