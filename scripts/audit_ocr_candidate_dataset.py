"""Audit every supported DATASET document through raw input -> OCR -> candidates.

The report intentionally does not claim model accuracy: a candidate matching
model cannot be evaluated before labelled candidates and a trained model exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct execution with ``python scripts/audit_ocr_candidate_dataset.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.candidates.pipeline import CandidatePipeline
from app.services.ocr.engine import OCREngine, to_evidence_json


DOC_TYPES = {
    "invoice": "INVOICE",
    "pod": "POD",
    "rate card": "RATE_AGREEMENT",
    "surat jalan": "SURAT_JALAN",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REQUIRED_CANDIDATE_COLUMNS = {
    "m_candidate_id", "m_doc_type", "m_document_id", "m_page", "m_cand_type",
    "m_raw_text", "m_parsed_value", "m_x1", "m_y1", "m_x2", "m_y2",
    "q_conf_mean", "q_box_estimated", "ctx_left", "ctx_right", "ctx_line",
}


def document_type_for(path: Path) -> str | None:
    for parent in path.parents:
        doc_type = DOC_TYPES.get(parent.name.casefold())
        if doc_type:
            return doc_type
    return None


def validate_evidence(evidence: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for page in evidence.get("pages", []):
        for line_index, line in enumerate(page.get("lines", [])):
            if not isinstance(line.get("tokens"), list):
                errors.append(f"line {line_index}: tokens is not a list")
            for token_index, token in enumerate(line.get("tokens", [])):
                bbox = token.get("bbox_norm")
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(0.0 <= float(value) <= 1.0 for value in bbox):
                    errors.append(f"line {line_index} token {token_index}: invalid bbox_norm")
                if "_estimated" not in token:
                    errors.append(f"line {line_index} token {token_index}: missing _estimated")
    return errors


def validate_candidates(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row_index, row in enumerate(rows):
        missing = REQUIRED_CANDIDATE_COLUMNS - set(row)
        if missing:
            errors.append(f"candidate {row_index}: missing {sorted(missing)}")
        for key in ("m_x1", "m_y1", "m_x2", "m_y2"):
            if key in row and not 0.0 <= float(row[key]) <= 1.0:
                errors.append(f"candidate {row_index}: {key} out of range")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive OCR and candidate audit for DATASET images.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\AIC\DATASET"))
    parser.add_argument("--out", type=Path, required=True, help="Output directory for report and optional artifacts.")
    parser.add_argument("--write-artifacts", action="store_true", help="Write evidence and candidate rows per source image.")
    parser.add_argument("--limit", type=int, help="Limit documents for a fast tuning iteration.")
    args = parser.parse_args()

    sources = [path for path in sorted(args.dataset_dir.rglob("*")) if path.suffix.casefold() in IMAGE_EXTENSIONS and document_type_for(path)]
    if args.limit:
        sources = sources[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.write_artifacts:
        (args.out / "evidence").mkdir(exist_ok=True)
        (args.out / "candidates").mkdir(exist_ok=True)

    engine = OCREngine()
    pipeline = CandidatePipeline("config/candidates")
    documents: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = defaultdict(int)

    for source in sources:
        doc_type = document_type_for(source)
        assert doc_type is not None
        document_id = source.relative_to(args.dataset_dir).with_suffix("").as_posix().replace("/", "--")
        raw = source.read_bytes()
        page = engine.process_image(raw, page_number=1, quality={})
        evidence = to_evidence_json([page])
        rows = pipeline.generate(document_id, doc_type, evidence, run_id="dataset_audit")
        evidence_errors = validate_evidence(evidence)
        candidate_errors = validate_candidates(rows)
        enabled_types = {field.cand_type for field in pipeline.registry.for_doc(doc_type)}
        emitted_types = {row["m_cand_type"] for row in rows}
        record = {
            "source": str(source.relative_to(args.dataset_dir)),
            "document_id": document_id,
            "document_type": doc_type,
            "raw": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "ocr": {
                "engine_used": page.engine_used,
                "width": page.width,
                "height": page.height,
                "lines": len(page.lines),
                "tokens": sum(len(line.tokens) for line in page.lines),
                "average_confidence": round(page.average_confidence(), 4),
                "preprocessing": page.quality.get("ocr_preprocessing", "none"),
            },
            "candidate_generation": {
                "rows": len(rows),
                "types": dict(sorted(Counter(row["m_cand_type"] for row in rows).items())),
                "enabled_types_without_candidates": sorted(enabled_types - emitted_types),
            },
            "contract": {"evidence_errors": evidence_errors, "candidate_errors": candidate_errors},
            "model_matching": {"status": "not_run", "reason": "No labelled candidate set or trained matching model is present."},
        }
        documents.append(record)
        aggregate["documents"] += 1
        aggregate["ocr_lines"] += record["ocr"]["lines"]
        aggregate["ocr_tokens"] += record["ocr"]["tokens"]
        aggregate["candidate_rows"] += len(rows)
        aggregate["contract_failures"] += int(bool(evidence_errors or candidate_errors))
        if args.write_artifacts:
            (args.out / "evidence" / f"{document_id}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            (args.out / "candidates" / f"{document_id}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{aggregate['documents']}/{len(sources)}] {record['source']}: {record['ocr']['average_confidence']:.3f}, {len(rows)} candidates")

    report = {"summary": dict(aggregate), "documents": documents}
    (args.out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
