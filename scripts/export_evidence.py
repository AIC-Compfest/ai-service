"""Export OCR evidence JSON (same shape as out/ocr_candidate_audit_tuned/evidence/) for a dataset folder.

Usage (from d:\\AIC\\ai-service):
    .\\.venv\\Scripts\\python.exe scripts\\export_evidence.py ..\\DATASET\\Dataset
    .\\.venv\\Scripts\\python.exe scripts\\export_evidence.py scripts\\dataset_A\n    .\\.venv\\Scripts\\python.exe scripts\\export_evidence.py ..\\DATASET\\Dataset\\invoice\\pdf\\invoice_18_variasi_1.pdf   (single file)
    .\\.venv\\Scripts\\python.exe scripts\\export_evidence.py ..\\DATASET\\Dataset\\invoice --out out\\invoice_run

Every PDF / PNG / JPG under the dataset folder (recursively) becomes one file:
    <out>/evidence/<relative/path/without/extension, '/' replaced by '--'>.json
Default <out> is out/<dataset_folder_name>_evidence, created if missing.
A summary of the run is written to <out>/export_report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="dataset folder (searched recursively) or a single pdf/png/jpg file")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: out/<dataset_name>_evidence)")
    p.add_argument("--formats", default="pdf,png,jpg", help="comma list of pdf,png,jpg (default: all three)")
    p.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all)")
    p.add_argument("--contains", default="", help="only files whose relative path contains this text")
    p.add_argument("--skip-existing", action="store_true", help="do not re-process files that already have an output JSON")
    return p.parse_args()


def doc_id_for(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().replace("/", "--")


def main() -> int:
    args = parse_args()
    target_in = args.dataset.resolve()
    single_file = target_in.is_file()
    if not single_file and not target_in.is_dir():
        print(f"Path not found: {target_in}", file=sys.stderr)
        return 2
    if single_file and target_in.suffix.lower() not in EXTENSIONS:
        print(f"Unsupported file type: {target_in.suffix} (use pdf/png/jpg)", file=sys.stderr)
        return 2
    root = target_in.parent if single_file else target_in

    wanted = {"." + f.strip().lower().lstrip(".") for f in args.formats.split(",")} & (EXTENSIONS | {".jpeg"})
    if ".jpg" in wanted:
        wanted.add(".jpeg")
    if single_file:
        sources = [target_in]
    else:
        sources = [
            p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in wanted and args.contains.lower() in p.relative_to(root).as_posix().lower()
        ]
    if args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        print("No PDF/PNG/JPG files found.", file=sys.stderr)
        return 2

    out = (args.out or ROOT / "out" / f"{(target_in.stem if single_file else root.name)}_evidence").resolve()
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {root}  ({len(sources)} files)\nOutput : {evidence_dir}", flush=True)

    # Heavy import after argument parsing so --help stays instant.
    from app.services.pipeline import Phase5DocumentPipeline

    pipeline = Phase5DocumentPipeline()
    records: list[dict] = []
    started = time.perf_counter()

    for i, src in enumerate(sources, 1):
        doc_id = doc_id_for(src, root)
        target = evidence_dir / f"{doc_id}.json"
        rec = {"source": src.relative_to(root).as_posix(), "output": target.name}
        if args.skip_existing and target.exists():
            rec["status"] = "skipped_existing"
            records.append(rec)
            print(f"[{i:>3}/{len(sources)}] SKIP {rec['source']}", flush=True)
            continue

        t0 = time.perf_counter()
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")  # silence library logging
        try:
            evidence = pipeline.process_document(src, filename=src.name, document_id=doc_id)
            target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            pages = evidence.get("pages", [])
            rec.update(
                status="ok",
                pages=len(pages),
                lines=sum(len(p.get("lines", [])) for p in pages),
                engines=sorted({p.get("engine_used") for p in pages if p.get("engine_used")}),
            )
        except Exception as exc:  # one bad file must not stop the batch
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            sys.stdout.close()
            sys.stdout, sys.stderr = real_out, real_err
        rec["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
        records.append(rec)
        detail = f"pages={rec.get('pages')} lines={rec.get('lines')}" if rec["status"] == "ok" else rec["error"][:90]
        print(f"[{i:>3}/{len(sources)}] {rec['status'].upper():<5} {rec['elapsed_seconds']:>6.1f}s  {rec['source']}  {detail}", flush=True)

    summary = {
        "files": len(records),
        "ok": sum(r["status"] == "ok" for r in records),
        "skipped_existing": sum(r["status"] == "skipped_existing" for r in records),
        "error": sum(r["status"] == "error" for r in records),
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(root),
        "summary": summary,
        "records": records,
    }
    (out / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {summary['ok']} ok, {summary['skipped_existing']} skipped, {summary['error']} error in {summary['total_seconds']}s")
    print(f"Evidence: {evidence_dir}\nReport  : {out / 'export_report.json'}")
    return 0 if summary["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
