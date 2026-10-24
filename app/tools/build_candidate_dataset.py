from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from app.services.candidates.pipeline import CandidatePipeline
from app.services.evidence.docling_tables import normalize_docling_evidence
from app.services.candidates.fuzzy_fields import FuzzyFieldMatcher
from app.services.candidates.training_schema import classify_column, load_policy
from app.services.candidates.version import candidate_pipeline_version


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
    report = {"m_pipeline_version": pipeline.pipeline_version, "empty_pages": [], "cap_overflows": [], "timings_ms": {}, "tables_per_doc": {}, "table_rank": {}, "evidence_role_agreement": 0.0, "table_recall_misses": [], "tables_cut_by_segment": 0}
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    _validate_manifest(rows)
    for manifest in rows[:args.limit]:
        evidence = json.loads(Path(manifest["evidence_json_path"]).read_text(encoding="utf-8"))
        evidence = restrict_pages(evidence, manifest.get("page_start"), manifest.get("page_end"))
        generated = pipeline.generate(manifest["document_id"], manifest["doc_type"], evidence, args.out.name, manifest.get("vendor_group", "UNKNOWN"), manifest.get("template_group", "UNKNOWN"))
        table_rows = [row for row in generated if row["m_cand_type"] == "table"]
        report["tables_per_doc"][manifest["document_id"]] = len(table_rows)
        report["table_rank"][manifest["document_id"]] = [{"table_id": row.get("m_table_id", ""), "rank": row.get("tb_heuristic_rank"), "score": row.get("tb_heuristic_score")} for row in table_rows]
        if any(row.get("m_evidence_role") == "line_items" and row.get("tb_heuristic_rank") == 1 for row in table_rows):
            report["evidence_role_agreement"] += 1
        if not table_rows and any(field.field == "line_items" for field in pipeline.registry.for_doc(manifest["doc_type"])):
            report["table_recall_misses"].append(manifest["document_id"])
        report["tables_cut_by_segment"] += sum(int(row.get("tb_cut_by_segment", 0) or 0) for row in table_rows)
        for item in generated:
            item["m_is_base"] = manifest.get("is_base", "1") or "1"
            item["m_base_document_id"] = manifest.get("base_document_id", "")
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
    _write_line_items(args, rows[:args.limit], pipeline)
    _write_status_template(args.out, rows[:args.limit], registry)
    _write_feature_manifest(args.out, args.config, grouped, registry)
    report.update({"documents": len(rows[:args.limit]), "candidate_counts": {key: len(value) for key, value in grouped.items()}})
    report["evidence_role_agreement"] = report["evidence_role_agreement"] / max(report["documents"], 1)
    (args.out / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_line_items(args, manifests, pipeline):
    preview_path = args.out / "li_preview.jsonl"
    for old in args.out.glob("li_columns_*.csv"):
        old.unlink()
    for old in args.out.glob("li_cells_*.csv"):
        old.unlink()
    for old in args.out.glob("li_rows_*.csv"):
        old.unlink()
    previews = []
    for manifest in manifests:
        evidence = json.loads(Path(manifest["evidence_json_path"]).read_text(encoding="utf-8"))
        evidence = restrict_pages(evidence, manifest.get("page_start"), manifest.get("page_end"))
        normalized, _ = normalize_docling_evidence(evidence)
        generated = pipeline.generate(manifest["document_id"], manifest["doc_type"], normalized, args.out.name, manifest.get("vendor_group", "UNKNOWN"), manifest.get("template_group", "UNKNOWN"))
        table_rows = {row.get("m_table_id"): row for row in generated if row["m_cand_type"] == "table"}
        matcher = FuzzyFieldMatcher(manifest["doc_type"], args.config / "line_item_aliases.yaml", args.config / "abbreviations.yaml")
        for page in normalized.get("pages", []):
            for table in page.get("tables", []):
                row = table_rows.get(table.get("table_id"), {})
                previews.append({"document_id": manifest["document_id"], "table_id": table.get("table_id"), "page": page.get("page_number"), "table_summary": {key: value for key, value in row.items() if key.startswith("tb_")}, "column_map": {str(header): (matcher.match(str(header)).top[0][0] if matcher.match(str(header)).top else "") for header in table.get("headers", [])}, "evidence_role": table.get("role")})
    with preview_path.open("w", encoding="utf-8") as preview_handle:
        for preview in previews:
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
                applies_to = [f"{field.doc_type}.{field.field}" for field in registry.for_doc(row.get("m_doc_type", "")) if field.cand_type == row.get("m_cand_type", "")]
                manifest.setdefault(name, {"name": name, "role": role, "train_policy": train_policy, "dtype": "bool" if isinstance(value, bool) else "float" if isinstance(value, float) else "int" if isinstance(value, int) else "cat" if isinstance(value, str) else "str", "nullable": value == "", "range": None, "vocab": [], "applies_to": [], "description": "Generated candidate column"})
                spec = manifest[name]
                spec["applies_to"] = sorted(set(spec.get("applies_to", [])) | set(applies_to))
                if spec["dtype"] == "cat" and value != "":
                    spec["vocab"] = sorted(set(spec.get("vocab", [])) | {str(value), "other"})
                if spec["dtype"] in {"int", "float"} and value != "":
                    numeric = float(value)
                    current = spec.get("range")
                    spec["range"] = [numeric, numeric] if current is None else [min(current[0], numeric), max(current[1], numeric)]
    for name in ("m_schema_version", "m_pipeline_version", "m_run_id", "m_candidate_id", "m_doc_type", "m_document_id", "m_vendor_group", "m_template_group", "m_is_base", "m_base_document_id", "m_page", "m_page_end", "m_table_id", "m_cand_type", "m_raw_text", "m_parsed_value", "m_parsed_norm", "m_x1", "m_y1", "m_x2", "m_y2", "m_line_idx", "m_tok_start", "m_tok_end", "m_overlap_types", "m_evidence_role", "m_evidence_role_reason"):
        manifest.setdefault(name, {"name": name, "role": "meta_id", "train_policy": "always_drop", "dtype": "str", "nullable": True, "range": None, "vocab": [], "applies_to": ["*"], "description": "Candidate metadata"})
    (output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training = {}
    doc_types = sorted({field.doc_type for field in registry.fields})
    for doc_type in doc_types:
        for field in registry.for_doc(doc_type):
            applicable = lambda spec: f"{doc_type}.{field.field}" in spec.get("applies_to", []) or "*" in spec.get("applies_to", [])
            keep = [name for name, spec in manifest.items() if spec["train_policy"] in {"keep", "ablate"} and spec["dtype"] in {"bool", "int", "float", "cat"} and applicable(spec)]
            ablate = [name for name, spec in manifest.items() if spec["train_policy"] == "ablate" and applicable(spec)]
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
    columns = ["m_schema_version", "m_pipeline_version", "m_run_id", "m_candidate_id", "m_doc_type", "m_document_id", "m_vendor_group", "m_template_group", "m_is_base", "m_base_document_id", "m_page", "m_cand_type", "m_raw_text", "m_parsed_value", "m_parsed_norm", "m_x1", "m_y1", "m_x2", "m_y2", "m_line_idx", "m_tok_start", "m_tok_end", "m_overlap_types", "ctx_left", "ctx_right", "ctx_above", "ctx_below", "ctx_line", "ctx_sent", "label_status", "label_notes"]
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


def _validate_manifest(rows):
    ids = {row["document_id"] for row in rows}
    for row in rows:
        if row.get("is_base", "1") not in {"0", "1", "", 0, 1}:
            raise ValueError(f"invalid is_base for {row['document_id']}")
        if row.get("is_base", "1") in {"0", 0} and row.get("base_document_id") not in ids:
            raise ValueError(f"variant {row['document_id']} has no resolving base_document_id")
        if row.get("is_base", "1") in {"1", 1} and row.get("base_document_id", ""):
            raise ValueError(f"base {row['document_id']} must not have base_document_id")
        if row.get("is_base", "1") in {"0", 0} and row.get("template_group") != row.get("base_document_id"):
            raise ValueError(f"variant {row['document_id']} template_group must equal base_document_id")


if __name__ == "__main__":
    main()
