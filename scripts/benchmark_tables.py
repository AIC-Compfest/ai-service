"""Capability and timing benchmark for the table pipeline (not an accuracy claim by itself).

Per document and aggregate: time per page (model load measured separately), OCR confidence, share of
estimated tokens inside table regions, documents with tables/rows, arithmetic pass rate when the validator
ran, and - when gold exists for the document - the structural metrics of ``app.tools.table_eval``.

Document selection comes from gold/manifest.json (decision D-027 supersedes the four-file allow-list D-009).

Usage (from ai-service):
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_tables.py --label baseline --splits dev,val --tables on
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_tables.py --label raster --kinds image --limit 3
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_tables.py --label fixtures --constructed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def discover_documents(manifest: dict, dataset_root: Path, *, splits: set[str] | None, kinds: set[str]) -> list[tuple[Path, str, str, bool, str]]:
    """Select manifest entries: ``(path, doc_type, entry_id, is_pdf, split)``. Missing files are reported, never substituted."""
    selected = []
    for entry in manifest["entries"]:
        if entry["source_kind"] not in kinds:
            continue
        if splits and entry["split"] not in splits:
            continue
        path = dataset_root / entry["path"]
        if not path.is_file():
            print(f"[WARN] manifest entry missing on disk: {entry['path']}")
            continue
        selected.append((path, entry["doc_type"], entry["id"], entry["source_kind"] == "pdf", entry["split"]))
    return selected


def region_table_tokens(page_evidence: dict[str, Any]) -> dict[str, Any]:
    """Token statistics inside the heuristic table region (kept for continuity with older benchmark files)."""
    table_bbox = next((r.get("bbox_norm") for r in page_evidence.get("regions", []) if r.get("region_type") == "table"), None)
    tokens = []
    if table_bbox:
        x1, y1, x2, y2 = table_bbox
        for line in page_evidence.get("lines", []):
            for tok in line.get("tokens", []):
                tb = tok.get("bbox_norm", [0, 0, 0, 0])
                if x1 <= (tb[0] + tb[2]) / 2 <= x2 and y1 <= (tb[1] + tb[3]) / 2 <= y2:
                    tokens.append(tok)
    estimated = sum(1 for t in tokens if t.get("_estimated", False))
    confs = [t.get("confidence", 0.0) for t in tokens if "confidence" in t]
    return {
        "tokens_in_table_region": len(tokens),
        "tokens_estimated": estimated,
        "pct_estimated": round(estimated / len(tokens) * 100, 1) if tokens else 0.0,
        "avg_confidence_table_region": round(sum(confs) / len(confs), 4) if confs else None,
        "has_table_region": table_bbox is not None,
    }


def page_metrics(evidence: Any, page_number: int, elapsed_ms: float) -> dict[str, Any]:
    data = evidence.to_dict()
    tables = data.get("tables", [])
    arith = [t.get("validation", {}).get("arithmetic") for t in tables]
    evaluated = sum(a.get("evaluated", 0) for a in arith if a)
    passed = sum(a.get("passed", 0) for a in arith if a)
    return {
        "page": page_number,
        "engine_used": evidence.engine_used,
        "fallback_used": evidence.fallback_used,
        "lines": len(evidence.lines),
        "tokens": sum(len(line.tokens) for line in evidence.lines),
        "avg_confidence": round(evidence.average_confidence(), 4),
        "tables_detected": len(tables),
        "table_row_count": sum(t.get("row_count", 0) for t in tables),
        "ocr_time_ms": round(elapsed_ms, 2),
        "arith_rows_evaluated": evaluated,
        "arith_rows_passed": passed,
        "_tables": tables,
        **region_table_tokens(data),
    }


def benchmark_document(path: Path, doc_type: str, doc_id: str, is_pdf: bool, engine: Any, pymupdf_module: Any) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    if is_pdf:
        with pymupdf_module.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                started = time.perf_counter()
                evidence = engine.process_pdf_page(page, number, {})
                pages.append(page_metrics(evidence, number, (time.perf_counter() - started) * 1000))
    else:
        started = time.perf_counter()
        evidence = engine.process_image(path.read_bytes(), page_number=1, quality={})
        pages.append(page_metrics(evidence, 1, (time.perf_counter() - started) * 1000))
    n = len(pages) or 1
    evaluated = sum(p["arith_rows_evaluated"] for p in pages)
    return {
        "document_id": doc_id,
        "doc_type": doc_type,
        "source": path.name,
        "page_count": len(pages),
        "avg_ocr_time_ms_per_page": round(sum(p["ocr_time_ms"] for p in pages) / n, 2),
        "avg_confidence": round(sum(p["avg_confidence"] for p in pages) / n, 4),
        "has_table": any(p["tables_detected"] > 0 for p in pages),
        "has_line_items": any(p["table_row_count"] > 0 for p in pages),
        "arith_row_pass_rate": round(sum(p["arith_rows_passed"] for p in pages) / evaluated, 4) if evaluated else None,
        "pages": pages,
    }


def aggregate_metrics(docs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(docs)
    if not total:
        return {"total_documents": 0}
    confs = [d["avg_confidence"] for d in docs]
    pct_est = [p["pct_estimated"] for d in docs for p in d["pages"] if p.get("has_table_region")]
    evaluated = sum(p["arith_rows_evaluated"] for d in docs for p in d["pages"])
    passed = sum(p["arith_rows_passed"] for d in docs for p in d["pages"])
    return {
        "total_documents": total,
        "documents_with_table": sum(d["has_table"] for d in docs),
        "documents_with_line_items": sum(d["has_line_items"] for d in docs),
        "avg_ocr_confidence": round(sum(confs) / total, 4),
        "low_confidence_docs_below_0_6": sum(c < 0.6 for c in confs),
        "avg_ms_per_page": round(sum(d["avg_ocr_time_ms_per_page"] for d in docs) / total, 2),
        "avg_pct_estimated_in_table_region": round(sum(pct_est) / len(pct_est), 1) if pct_est else None,
        "arith_rows_evaluated": evaluated,
        "arith_row_pass_rate": round(passed / evaluated, 4) if evaluated else None,
    }


def structure_metrics(docs: list[dict[str, Any]], gold_dir: Path | None, allow_unreviewed: bool) -> dict[str, Any] | None:
    """Structural metrics for documents that have gold; presence is NOT accuracy, this is."""
    if gold_dir is None:
        return None
    from app.tools.table_eval.metrics import evaluate_doc, summarize
    from app.tools.table_eval.model import gold_from_json, prediction_from_evidence, read_json

    counts: Counter = Counter()
    failures: Counter = Counter()
    used = 0
    for doc in docs:
        gold_path = gold_dir / f"{doc['document_id']}.json"
        if not gold_path.exists():
            continue
        gold = gold_from_json(read_json(gold_path), allow_unreviewed=allow_unreviewed)
        prediction = prediction_from_evidence({"pages": [{"page_number": p["page"], "tables": p["_tables"]} for p in doc["pages"]]}, doc["document_id"])
        result = evaluate_doc(gold, prediction)
        counts.update(result.counts)
        failures.update(result.failures)
        used += 1
    report = summarize(counts, failures)
    report["documents_with_gold"] = used
    report["stamp"] = "GT sintetis/structural gold synthetic_grounded; bukan gold manusia."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=AI_ROOT / "gold" / "manifest.json")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset")))
    parser.add_argument("--gold-dir", type=Path, default=AI_ROOT / "gold" / "regression")
    parser.add_argument("--out", type=Path, default=AI_ROOT / "out" / "benchmarks")
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--splits", default="dev,val", help="Comma list; 'holdout' only at the end of a phase")
    parser.add_argument("--kinds", default="pdf", help="pdf, image or pdf,image")
    parser.add_argument("--tables", choices=("on", "off"), default="on", help="Sets ENABLE_TABLE_EXTRACTOR for this run")
    parser.add_argument("--preprocess", action="store_true", help="Sets ENABLE_OCR_PREPROCESS")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--constructed", action="store_true", help="Benchmark the constructed fixtures instead of the dataset")
    parser.add_argument("--allow-unreviewed", action="store_true", default=True)
    args = parser.parse_args()

    from app.core.config import settings

    settings.enable_table_extractor = args.tables == "on"
    settings.enable_ocr_preprocess = args.preprocess
    import pymupdf

    from app.services.ocr.engine import OCREngine

    if args.constructed:
        from app.tools.table_eval.synthetic import build_all

        fixture_dir = args.out / "constructed_fixtures"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        sources = []
        for fixture in build_all():
            path = fixture_dir / f"{fixture.name}.pdf"
            path.write_bytes(fixture.pdf)
            sources.append((path, "CONSTRUCTED", f"constructed/{fixture.name}", True, "constructed"))
        gold_dir = AI_ROOT / "gold" / "constructed"
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        splits = {s for s in args.splits.split(",") if s} or None
        sources = discover_documents(manifest, args.dataset_root, splits=splits, kinds=set(args.kinds.split(",")))
        gold_dir = args.gold_dir
    if not sources:
        raise SystemExit("No documents selected; check --manifest/--splits/--kinds (nothing is substituted).")
    if args.limit:
        sources = sources[: args.limit]

    engine = OCREngine()
    model_load_ms = None
    if any(not is_pdf for _, _, _, is_pdf, _ in sources):
        # Real warm-up of the raster engine. Failure is fatal: a silent skip would put model load into page timings.
        import numpy as np

        started = time.perf_counter()
        _ = engine.paddle_ocr
        engine.process_image(_blank_png(), page_number=1, quality={})
        model_load_ms = round((time.perf_counter() - started) * 1000, 1)
        print(f"[INFO] raster engine loaded in {model_load_ms} ms (excluded from page timings)")

    documents, errors = [], []
    by_type: dict[str, list] = defaultdict(list)
    for index, (path, doc_type, doc_id, is_pdf, split) in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {doc_id}")
        try:
            result = benchmark_document(path, doc_type, doc_id, is_pdf, engine, pymupdf)
        except Exception as exc:  # reported in the file, not hidden
            errors.append({"document_id": doc_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [ERROR] {exc}")
            continue
        result["split"] = split
        documents.append(result)
        by_type[doc_type].append(result)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "benchmark_label": args.label,
        "timestamp_utc": timestamp,
        "python_version": sys.version,
        "settings": {"enable_table_extractor": settings.enable_table_extractor, "enable_ocr_preprocess": settings.enable_ocr_preprocess, "table_provider_order": list(settings.table_provider_order)},
        "model_load_ms": model_load_ms,
        "total_documents": len(documents),
        "errors": errors,
        "aggregate": aggregate_metrics(documents),
        "by_doc_type": {t: aggregate_metrics(d) for t, d in by_type.items()},
        "structure_metrics": structure_metrics(documents, gold_dir, args.allow_unreviewed),
        "documents": [{k: v for k, v in d.items() if k != "pages"} | {"pages": [{k: v for k, v in p.items() if k != "_tables"} for p in d["pages"]]} for d in documents],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{timestamp}_{args.label}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] {out_path}")
    print(json.dumps(report["aggregate"], indent=2))
    return 1 if errors and not documents else 0


def _blank_png() -> bytes:
    import cv2
    import numpy as np

    ok, buffer = cv2.imencode(".png", np.full((120, 240, 3), 255, dtype=np.uint8))
    return buffer.tobytes()


if __name__ == "__main__":
    raise SystemExit(main())
