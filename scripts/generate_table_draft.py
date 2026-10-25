"""
generate_table_draft.py

Skrip praktis untuk menjalankan ekstraksi tabel lokal pada suatu dokumen (PDF / Gambar)
dan menghasilkan dua file JSON:
  1. Full Evidence JSON: hasil ekstraksi mentah OCR, koordinat, dan deteksi tabel.
  2. Gold Draft JSON: draf siap-review (reviewed: false) yang bisa diedit oleh manusia
     sebelum dikunci menjadi kunci jawaban resmi (Gold Data).

Penggunaan:
    python scripts/generate_table_draft.py "<path-ke-dataset>/invoice/pdf/invoice_1_variasi_1.pdf"
    python scripts/generate_table_draft.py path/to/doc.pdf --out-dir out/my_drafts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Pastikan import app berjalan normal saat dijalankan langsung dari terminal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.services.pipeline import Phase5DocumentPipeline
from app.tools.table_eval.model import gold_pages_from_evidence


def generate_draft(
    input_path: Path,
    output_dir: Path,
    document_id: str | None = None,
) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"File dokumen tidak ditemukan: {input_path}")

    # Flags are set here (not at import time) so importing this module has no side effects.
    settings.enable_table_extractor = True
    settings.enable_ocr_preprocess = True

    doc_id = document_id or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Menjalankan ekstraksi pipeline untuk: {input_path.name} (id={doc_id})...")
    pipeline = Phase5DocumentPipeline()
    evidence = pipeline.process_document(
        input_path,
        filename=input_path.name,
        document_id=doc_id,
    )

    # 1. Simpan full evidence JSON
    evidence_path = output_dir / f"{doc_id}_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[2/3] Full evidence tersimpan di: {evidence_path}")

    # 2. Buat Draf Gold Tabel (GOLD_REVIEW: pages[].tables[].rows[].cells[])
    # Deteksi doc_type dari klasifikasi pipeline
    detected_type = (
        evidence.get("classification", {}).get("document_type")
        or "UNKNOWN"
    )

    # Susun skema gold resmi (kompatibel dengan eval_tables.py dan GOLD_REVIEW_AND_PADDLE_DIAGNOSTICS.md)
    gold_draft: dict[str, Any] = {
        "document_id": doc_id,
        "source": {
            "relative_path": input_path.name,  # file name only; keep the path relative to the dataset root when reviewing
            "sha256": evidence.get("source_sha256", ""),
            "source_kind": input_path.suffix.lstrip(".").lower(),
        },
        "document_type": detected_type,
        "reviewed": False,
        "review": {
            "schema_version": "1",
            "prepared_by": "",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "approved_by": "",
            "approved_at_utc": "",
            "notes": "Draf otomatis dibuat dari ekstraksi pipeline. Periksa kesesuaian baris dan sel terhadap dokumen asli sebelum mengubah reviewed menjadi true.",
        },
        "source_label": "pipeline_draft",
        "pages": gold_pages_from_evidence(evidence),
    }

    draft_path = output_dir / f"{doc_id}_draft.json"
    draft_path.write_text(
        json.dumps(gold_draft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[3/3] Draf Gold siap-review tersimpan di: {draft_path}")

    return evidence_path, draft_path


def main():
    parser = argparse.ArgumentParser(description="Generate table evidence & review-ready gold draft.")
    parser.add_argument("input", type=Path, help="Path ke file PDF atau citra")
    parser.add_argument("--out-dir", type=Path, default=Path("out/drafts"), help="Direktori output")
    parser.add_argument("--doc-id", type=str, default=None, help="Custom document ID")

    args = parser.parse_args()

    try:
        ev_file, draft_file = generate_draft(args.input, args.out_dir, args.doc_id)
        print("\n" + "=" * 60)
        print("SUKSES GENERATE DRAF TABEL!")
        print(f"-> Full Evidence : {ev_file}")
        print(f"-> Gold Draft    : {draft_file}")
        print("\nLangkah selanjutnya:")
        print(f"1. Buka file PDF asli dan file draf: {draft_file}")
        print("2. Cocokkan header & baris (rows) dengan dokumen asli.")
        print('3. Jika sudah sesuai 100%, ubah `"reviewed": false` menjadi `"reviewed": true`.')
        print("4. Jalankan evaluasi dengan:")
        print(f"   python scripts/eval_tables.py --gold {draft_file} --prediction {ev_file}   (evidence pages[].tables is read directly)")
        print("=" * 60)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
