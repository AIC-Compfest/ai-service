"""
benchmark_tables.py

Harness benchmark untuk evaluasi pipeline tabel per dokumen.
Menjalankan pipeline pada folder dokumen dan menulis JSON ke
out/benchmarks/<timestamp>_<label>.json.

Metrik per dokumen dan agregat:
- Waktu per halaman (warm, tidak menghitung waktu muat model)
- Rata-rata confidence OCR dan persentase token _estimated di region tabel
- Jumlah dokumen dengan >= 1 tabel dan >= 1 baris item
- Tingkat lolos validasi aritmetika (setelah Langkah 3 tersedia)
- Bila gold tersedia: row precision/recall/F1 dan cell accuracy

Penggunaan:
    python scripts/benchmark_tables.py --doc-dir D:/AIC/DATASET --label baseline
    python scripts/benchmark_tables.py --doc-dir D:/AIC/DATASET --label after_step1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Izinkan run langsung: python scripts/benchmark_tables.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}

DOC_TYPES = {
    "invoice": "INVOICE",
    "pod": "POD",
    "rate card": "RATE_AGREEMENT",
    "surat jalan": "SURAT_JALAN",
}

# The only approved real-data source for this work.  These names deliberately
# refer to the four top-level document-type collections in DATASET.  Do not
# recurse through DATASET wholesale: it also contains exports, archives, and
# duplicate/transformed fixtures which must not be used for benchmark claims.
CANONICAL_DATASET_DIRECTORIES = {
    "invoice": "INVOICE",
    "PoD": "POD",
    "rate card": "RATE_AGREEMENT",
    "surat jalan": "SURAT_JALAN",
}


def document_type_for(path: Path) -> str | None:
    """Infer doc type from parent directory name."""
    for parent in path.parents:
        doc_type = DOC_TYPES.get(parent.name.casefold())
        if doc_type:
            return doc_type
    return None


def discover_dataset_documents(dataset_root: Path) -> list[tuple[Path, str, bool]]:
    """Return only canonical documents from the four approved DATASET folders.

    A canonical source is a PDF or image immediately below an ``image`` or
    ``pdf`` directory of one of the four document-type folders.  Nested
    directories often contain transformed copies or third-party samples and
    are intentionally excluded from all benchmark metrics.
    """
    sources: list[tuple[Path, str, bool]] = []
    for directory_name, doc_type in CANONICAL_DATASET_DIRECTORIES.items():
        type_root = dataset_root / directory_name
        for media_directory in (type_root / "pdf", type_root / "image"):
            if not media_directory.is_dir():
                continue
            for path in sorted(media_directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS | PDF_EXTENSIONS:
                    continue
                sources.append((path, doc_type, path.suffix.lower() in PDF_EXTENSIONS))
    return sources


def _region_table_tokens(page_evidence: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics about tokens that fall in table regions."""
    regions = page_evidence.get("regions", [])
    table_bbox = None
    for region in regions:
        if region.get("region_type") == "table":
            table_bbox = region.get("bbox_norm")
            break

    tokens_in_region: list[dict] = []
    if table_bbox:
        x1, y1, x2, y2 = table_bbox
        for line in page_evidence.get("lines", []):
            for tok in line.get("tokens", []):
                tb = tok.get("bbox_norm", [0, 0, 0, 0])
                # Token center in region
                cx = (tb[0] + tb[2]) / 2
                cy = (tb[1] + tb[3]) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    tokens_in_region.append(tok)

    estimated = sum(1 for t in tokens_in_region if t.get("_estimated", False))
    total = len(tokens_in_region)
    conf_vals = [t.get("confidence", 0.0) for t in tokens_in_region if "confidence" in t]
    return {
        "tokens_in_table_region": total,
        "tokens_estimated": estimated,
        "pct_estimated": round(estimated / total * 100, 1) if total else 0.0,
        "avg_confidence_table_region": round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None,
        "has_table_region": table_bbox is not None,
    }


def benchmark_document(
    path: Path,
    doc_type: str,
    engine: Any,
    is_pdf: bool,
    fitz_module: Any = None,
) -> dict[str, Any]:
    """Run pipeline on one document and return per-document metrics."""
    raw = path.read_bytes()
    doc_id = path.stem

    pages_metrics: list[dict[str, Any]] = []
    total_ocr_time_ms = 0.0

    if is_pdf and fitz_module is not None:
        pdf = fitz_module.open(stream=raw, filetype="pdf")
        for page_num, page in enumerate(pdf, start=1):
            quality = {}
            t0 = time.perf_counter()
            evidence = engine.process_pdf_page(page, page_num, quality)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ocr_time_ms += elapsed_ms

            evdict = evidence.to_dict()
            tok_metrics = _region_table_tokens(evdict)
            tables = evdict.get("tables", [])

            pages_metrics.append({
                "page": page_num,
                "engine_used": evidence.engine_used,
                "fallback_used": evidence.fallback_used,
                "lines": len(evidence.lines),
                "tokens": sum(len(l.tokens) for l in evidence.lines),
                "avg_confidence": round(evidence.average_confidence(), 4),
                "tables_detected": len(tables),
                "table_row_count": sum(t.get("row_count", 0) for t in tables),
                "table_confidence": tables[0].get("confidence") if tables else None,
                "ocr_time_ms": round(elapsed_ms, 2),
                **tok_metrics,
            })
        pdf.close()
    else:
        t0 = time.perf_counter()
        evidence = engine.process_image(raw, page_number=1, quality={})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ocr_time_ms += elapsed_ms

        evdict = evidence.to_dict()
        tok_metrics = _region_table_tokens(evdict)
        tables = evdict.get("tables", [])

        pages_metrics.append({
            "page": 1,
            "engine_used": evidence.engine_used,
            "fallback_used": evidence.fallback_used,
            "lines": len(evidence.lines),
            "tokens": sum(len(l.tokens) for l in evidence.lines),
            "avg_confidence": round(evidence.average_confidence(), 4),
            "tables_detected": len(tables),
            "table_row_count": sum(t.get("row_count", 0) for t in tables),
            "table_confidence": tables[0].get("confidence") if tables else None,
            "ocr_time_ms": round(elapsed_ms, 2),
            **tok_metrics,
        })

    n_pages = len(pages_metrics)
    avg_ms_per_page = total_ocr_time_ms / n_pages if n_pages else 0
    avg_conf = (
        sum(p["avg_confidence"] for p in pages_metrics) / n_pages
        if n_pages else 0.0
    )
    has_table = any(p["tables_detected"] > 0 for p in pages_metrics)
    has_line_items = any(p["table_row_count"] > 0 for p in pages_metrics)

    return {
        "document_id": doc_id,
        "doc_type": doc_type,
        "source": str(path),
        "page_count": n_pages,
        "avg_ocr_time_ms_per_page": round(avg_ms_per_page, 2),
        "avg_confidence": round(avg_conf, 4),
        "has_table": has_table,
        "has_line_items": has_line_items,
        # Arith validation not available yet (Langkah 3); marked explicitly
        "arith_validation_available": False,
        "arith_row_pass_rate": None,
        "arith_doc_pass": None,
        # Gold not yet available
        "gold_available": False,
        "row_precision": None,
        "row_recall": None,
        "row_f1": None,
        "cell_accuracy": None,
        "pages": pages_metrics,
    }


def aggregate_metrics(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across all documents."""
    total = len(docs)
    if not total:
        return {"total_documents": 0}

    with_table = sum(1 for d in docs if d["has_table"])
    with_items = sum(1 for d in docs if d["has_line_items"])
    all_confs = [d["avg_confidence"] for d in docs]
    all_times = [d["avg_ocr_time_ms_per_page"] for d in docs]
    all_pct_est = [
        p["pct_estimated"]
        for d in docs
        for p in d["pages"]
        if p.get("has_table_region")
    ]
    low_conf_docs = sum(1 for c in all_confs if c < 0.6)

    return {
        "total_documents": total,
        "documents_with_table": with_table,
        "pct_with_table": round(with_table / total * 100, 1),
        "documents_with_line_items": with_items,
        "pct_with_line_items": round(with_items / total * 100, 1),
        "avg_ocr_confidence": round(sum(all_confs) / total, 4),
        "low_confidence_docs_below_0_6": low_conf_docs,
        "avg_ms_per_page": round(sum(all_times) / total, 2),
        "avg_pct_estimated_in_table_region": (
            round(sum(all_pct_est) / len(all_pct_est), 1) if all_pct_est else None
        ),
        # Arith & gold not yet available
        "arith_validation_available": False,
        "gold_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark table extraction pipeline on a directory of documents."
    )
    parser.add_argument(
        "--doc-dir", type=Path, default=Path(r"D:\AIC\DATASET"),
        help="Directory containing document images/PDFs (subdirs name = doc type)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("out/benchmarks"),
        help="Output directory for benchmark JSON"
    )
    parser.add_argument(
        "--label", default="benchmark",
        help="Label for this run (e.g. 'baseline', 'after_step1')"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Max documents to process (useful for quick runs)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic PDF fixtures instead of real dataset"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Import lazily so benchmark script doesn't crash if dependencies missing
    from app.services.ocr.engine import OCREngine
    try:
        import fitz as _fitz
    except ImportError:
        _fitz = None
        print("[WARN] PyMuPDF not available — PDF processing disabled")

    engine = OCREngine()
    # Warm-up: instantiate reader once so timing is accurate
    # (EasyOCR loads model on first call; we trigger it here)
    print("[INFO] Warming up OCR engine...")
    try:
        _dummy = engine.reader  # triggers lazy init
        print("[INFO] OCR engine ready")
    except Exception as ex:
        print(f"[WARN] OCR engine warm-up issue: {ex}")

    if args.synthetic:
        sources = _make_synthetic_fixtures(args.out)
        print(f"[INFO] Using {len(sources)} synthetic fixtures")
    else:
        if not args.doc_dir.is_dir():
            raise SystemExit(
                f"Approved dataset directory not found: {args.doc_dir}. "
                "Use --synthetic explicitly for synthetic fixtures."
            )
        sources = discover_dataset_documents(args.doc_dir)
        if not sources:
            raise SystemExit(
                f"No canonical documents found in {args.doc_dir}. "
                "Expected immediate files in the four approved type folders."
            )
        print(
            f"[INFO] Found {len(sources)} canonical documents in the four approved DATASET folders"
        )

    if args.limit:
        sources = sources[: args.limit]

    results: list[dict[str, Any]] = []
    by_type: dict[str, list] = defaultdict(list)

    for idx, (path, doc_type, is_pdf) in enumerate(sources, 1):
        print(f"[{idx}/{len(sources)}] {path.name} ({doc_type})")
        try:
            doc_metrics = benchmark_document(
                path, doc_type, engine,
                is_pdf=is_pdf,
                fitz_module=_fitz,
            )
            results.append(doc_metrics)
            by_type[doc_type].append(doc_metrics)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results.append({
                "document_id": path.stem,
                "doc_type": doc_type,
                "source": str(path),
                "error": str(exc),
            })

    # Build report
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "benchmark_label": args.label,
        "timestamp_utc": timestamp,
        "python_version": sys.version,
        "total_documents": len(results),
        "aggregate": aggregate_metrics([r for r in results if "error" not in r]),
        "by_doc_type": {
            dt: aggregate_metrics(docs) for dt, docs in by_type.items()
        },
        "documents": results,
    }

    out_path = args.out / f"{timestamp}_{args.label}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[DONE] Report saved to: {out_path}")
    print(json.dumps(report["aggregate"], indent=2))


def _make_synthetic_fixtures(out_dir: Path) -> list[tuple[Path, str, bool]]:
    """
    Generate synthetic PDF fixtures for benchmark testing when no real dataset
    is available. Creates one PDF per doc type with a simple table.
    Returns list of (path, doc_type, is_pdf) tuples.
    """
    try:
        import fitz
    except ImportError:
        print("[ERROR] PyMuPDF required for synthetic fixture generation")
        return []

    fixture_dir = out_dir / "synthetic_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixtures: list[tuple[Path, str, bool]] = []

    for doc_type_label, doc_type in [
        ("invoice", "INVOICE"),
        ("pod", "POD"),
        ("surat_jalan", "SURAT_JALAN"),
    ]:
        out_path = fixture_dir / f"synthetic_{doc_type_label}.pdf"
        if not out_path.exists():
            _create_synthetic_pdf(out_path, doc_type)
        fixtures.append((out_path, doc_type, True))

    return fixtures


def _create_synthetic_pdf(path: Path, doc_type: str) -> None:
    """Create a minimal synthetic PDF with a simple table for benchmark testing."""
    try:
        import fitz
    except ImportError:
        return

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Header
    page.insert_text((50, 50), f"SYNTHETIC {doc_type} DOCUMENT", fontsize=14)
    page.insert_text((50, 80), "PT Example Vendor | Invoice No: INV-2026-001", fontsize=10)
    page.insert_text((50, 100), "Date: 01 Oktober 2026", fontsize=10)

    # Table header
    y = 200
    cols = [50, 200, 320, 420, 500]
    headers = ["No", "Description", "Qty", "Rate (IDR)", "Total (IDR)"]
    for i, h in enumerate(headers):
        page.insert_text((cols[i], y), h, fontsize=9)

    # Table rows
    rows = [
        ["1", "Layanan Pengiriman JKT-SBY", "125", "200.000", "25.000.000"],
        ["2", "Fuel Surcharge 5%", "1", "1.250.000", "1.250.000"],
        ["3", "Handling Fee", "1", "500.000", "500.000"],
    ]
    for ri, row in enumerate(rows):
        y += 20
        for ci, cell in enumerate(row):
            page.insert_text((cols[ci], y), cell, fontsize=9)

    # Totals
    y += 30
    page.insert_text((380, y), "Subtotal:", fontsize=9)
    page.insert_text((500, y), "26.750.000", fontsize=9)
    y += 15
    page.insert_text((380, y), "PPN 11%:", fontsize=9)
    page.insert_text((500, y), "2.942.500", fontsize=9)
    y += 15
    page.insert_text((380, y), "Grand Total:", fontsize=10)
    page.insert_text((500, y), "29.692.500", fontsize=10)

    doc.save(str(path))
    doc.close()
    print(f"  [SYNTH] Created: {path}")


if __name__ == "__main__":
    main()
