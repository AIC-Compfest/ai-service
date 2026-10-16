from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from app.services.candidates.pipeline import CandidatePipeline
from app.services.candidates.registry import FieldRegistry
from app.services.candidates.version import candidate_pipeline_version
from app.services.labeling.export import export_labels
from app.services.labeling.grounding import ground_truth_field
from app.services.labeling.schema import FieldStatus, FieldLabel, LabelSource, ReviewState, TruthField, TruthRecord
from app.services.labeling.store import LabelStore
from app.services.labeling.template_profile import build_template_profile
from app.services.labeling.transfer import transfer_template
from app.core.config import candidates_config_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 labeling workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, default=Path("reports/POD_Training"))
    prepare.add_argument("--ground-truth", type=Path)
    prepare.add_argument("--out", type=Path, default=Path("data/phase2"))
    prepare.add_argument("--config", type=Path, default=candidates_config_dir())
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--skip-ocr", action="store_true")
    base = sub.add_parser("llm-base")
    base.add_argument("--ground-truth", type=Path, required=True)
    base.add_argument("--out", type=Path, default=Path("data/labels"))
    base.add_argument("--dry-run", action="store_true")
    ground = sub.add_parser("ground")
    ground.add_argument("--truth-dir", type=Path, default=Path("data/labels/truth_llm"))
    ground.add_argument("--candidate-dir", type=Path, required=True)
    ground.add_argument("--labels", type=Path, default=Path("data/labels"))
    ground.add_argument("--config", type=Path, default=candidates_config_dir())
    export = sub.add_parser("export")
    export.add_argument("--candidate-dir", type=Path, required=True)
    export.add_argument("--labels", type=Path, default=Path("data/labels"))
    export.add_argument("--reground", action="store_true")
    accept = sub.add_parser("accept-local", help="Promote fake/local truth for smoke training only")
    accept.add_argument("--labels", type=Path, default=Path("data/labels"))
    templates = sub.add_parser("build-templates")
    templates.add_argument("--candidate-dir", type=Path, required=True)
    templates.add_argument("--labels", type=Path, default=Path("data/labels"))
    templates.add_argument("--out", type=Path, default=Path("data/labels/templates"))
    transfer = sub.add_parser("transfer")
    transfer.add_argument("--candidate-dir", type=Path, required=True)
    transfer.add_argument("--labels", type=Path, default=Path("data/labels"))
    transfer.add_argument("--templates", type=Path, default=Path("data/labels/templates"))
    transfer.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": _prepare, "llm-base": _llm_base, "ground": _ground, "export": _export, "accept-local": _accept_local, "build-templates": _templates, "transfer": _transfer}[args.command](args)


def _prepare(args) -> None:
    truth_path = args.ground_truth or args.source_dir / "ground_truth_all_17_documents.json"
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    out = args.out
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    records = []
    pipeline_version = candidate_pipeline_version(args.config)
    for document in data["documents"][:args.limit]:
        variations = document["variations"]
        base_id = f"pod-{document['id']}-v1"
        for index, variation in enumerate(variations):
            document_id = f"pod-{document['id']}-v{index + 1}"
            evidence_path = evidence_dir / f"{document_id}.json"
            source = args.source_dir / variation["file"]
            if not args.skip_ocr:
                from app.services.pipeline import Phase5DocumentPipeline
                evidence = Phase5DocumentPipeline().process_document(source, document_id=document_id)
                evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({"document_id": document_id, "evidence_json_path": str(evidence_path), "doc_type": "POD", "vendor_group": "UNKNOWN", "template_group": base_id, "page_start": "1", "page_end": "1", "source_file": str(source), "is_base": "1" if index == 0 else "0", "base_document_id": "" if index == 0 else base_id, "ground_truth_json": str(truth_path), "m_pipeline_version": pipeline_version})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        columns = list(records[0]) if records else ["document_id"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {manifest_path} ({len(records)} logical documents)")


def _llm_base(args) -> None:
    data = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    target = args.out / "truth_llm"
    target.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry-run: would create {sum(len(item['variations'][:1]) for item in data['documents'])} base truth records; no network request sent")
        return
    for document in data["documents"]:
        variation = document["variations"][0]
        fields = _pod_truth_fields(variation["fields"])
        truth = TruthRecord(f"pod-{document['id']}-v1", "POD", "extract_truth_v1", "fake-local", fields)
        (target / f"{truth.document_id}.json").write_text(json.dumps(truth.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(list(target.glob('*.json')))} local base truth records")


def _ground(args) -> None:
    store = LabelStore(args.labels)
    registry = FieldRegistry.load(args.config / "field_registry.yaml")
    rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for source in args.candidate_dir.glob("candidates_*.csv"):
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_doc.setdefault(row["m_document_id"], []).append(row)
    for truth_path in sorted(args.truth_dir.glob("*.json")):
        truth = TruthRecord.from_dict(json.loads(truth_path.read_text(encoding="utf-8")))
        document_candidates = rows_by_doc.get(truth.document_id, [])
        candidate_version = document_candidates[0].get("m_pipeline_version", "") if document_candidates else ""
        for field in registry.for_doc(truth.doc_type):
            candidates = [row for row in rows_by_doc.get(truth.document_id, []) if row["m_cand_type"] == field.cand_type]
            label = ground_truth_field(truth, field.field, field.cand_type, candidates, template_group=truth.document_id, pipeline_version=candidate_version)
            store.upsert(label)
    print(f"Stored {len(store.list())} field labels")


def _export(args) -> None:
    print(json.dumps(export_labels(args.candidate_dir, args.labels, reground=args.reground), indent=2, sort_keys=True))


def _accept_local(args) -> None:
    store = LabelStore(args.labels)
    labels = store.list()
    for label in labels:
        if label.label_source == LabelSource.LLM_BASE:
            label.review_state = ReviewState.AUTO
            store.upsert(label, reviewer="local-smoke", reason="explicit fake/local truth smoke test")
    print(f"Promoted {sum(label.label_source == LabelSource.LLM_BASE for label in labels)} local labels")


def _templates(args) -> None:
    base_ids = sorted({label.document_id for label in LabelStore(args.labels).list() if label.is_base})
    for document_id in base_ids:
        build_template_profile(args.candidate_dir, args.labels, document_id, args.out)
    print(f"Wrote {len(base_ids)} template profiles")


def _transfer(args) -> None:
    count = 0
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_base") == "0":
                profile = args.templates / f"{row['base_document_id']}.json"
                if profile.exists():
                    count += transfer_template(args.candidate_dir, args.labels, profile, row["document_id"], row.get("template_group", row["base_document_id"]))
    print(f"Transferred {count} variant field labels")


def _pod_truth_fields(fields: dict[str, Any]) -> dict[str, TruthField]:
    result = {}
    result["nomor_berita_acara"] = TruthField(FieldStatus.PRESENT, fields.get("nomor_berita_acara"))
    result["hari"] = TruthField(FieldStatus.PRESENT, fields.get("hari"))
    result["tanggal"] = TruthField(FieldStatus.PRESENT, fields.get("tanggal"))
    for side in ("pihak_pertama", "pihak_kedua"):
        for sub in ("nama", "jabatan", "alamat"):
            result[f"{side}_{sub}"] = TruthField(FieldStatus.PRESENT, fields.get(side, {}).get(sub))
    return result


if __name__ == "__main__":
    main()