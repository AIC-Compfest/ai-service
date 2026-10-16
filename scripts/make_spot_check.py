"""Prepare the owner spot-check of the bootstrap structural gold (non-blocking).

Selects 20 stratified pages (document type x header found), excludes the holdout split, draws the gold on the
rendered page and writes out/review/gold_check/CHECKLIST.md for the owner to fill in.

    .\\.venv\\Scripts\\python.exe scripts\\make_spot_check.py
"""
from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(AI_ROOT / "scripts"))

import pymupdf  # noqa: E402
from PIL import Image  # noqa: E402
from review_overlay import draw_tables, gold_page_to_overlay  # noqa: E402

QUOTA = {"INVOICE": 6, "POD": 4, "RATE_AGREEMENT": 5, "SURAT_JALAN": 5}


def main() -> int:
    dataset_root = Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset"))
    manifest = json.loads((AI_ROOT / "gold" / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in manifest["entries"]}
    candidates: dict[str, list[tuple[str, Path]]] = {k: [] for k in QUOTA}
    for path in sorted((AI_ROOT / "gold" / "regression").rglob("*.json")):
        gold = json.loads(path.read_text(encoding="utf-8"))
        entry = entries[gold["document_id"]]
        if entry["split"] == "holdout":
            continue
        candidates[gold["document_type"]].append((gold["document_id"], path))
    out = AI_ROOT / "out" / "review" / "gold_check"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_type, quota in QUOTA.items():
        items = candidates[doc_type]
        stride = max(len(items) // quota, 1)
        for doc_id, gold_path in items[::stride][:quota]:
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            page = gold["pages"][0]
            pdf = dataset_root / entries[doc_id]["path"]
            pix = pymupdf.open(pdf)[page["page_number"] - 1].get_pixmap(dpi=100)
            image = Image.open(BytesIO(pix.tobytes("png")))
            name = doc_id.replace("/", "__") + f"_p{page['page_number']}.png"
            draw_tables(image, gold_page_to_overlay(page["tables"]), title=f"GOLD {doc_id} p{page['page_number']}").save(out / name)
            rows.append((name, doc_id, page["page_number"], len(page["tables"]), page["tables"][0]["quality"]["header_found"]))
    lines = [
        "# Spot-check structural gold (synthetic_grounded)",
        "",
        "Buka tiap PNG. Merah = tabel, biru = batas kolom, hijau = sel, magenta = baris ringkasan.",
        "Isi Y atau N. Gold bukan `reviewed` sebelum dua reviewer berbeda menyetujui (aturan GOLD_REVIEW).",
        "",
        "| file | dokumen | hal | tabel | header ditemukan | header benar | kolom benar | baris benar | ringkasan terpisah | catatan |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, doc_id, page_number, n_tables, header_found in rows:
        lines.append(f"| {name} | {doc_id} | {page_number} | {n_tables} | {'ya' if header_found else 'tidak'} |  |  |  |  |  |")
    (out / "CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} pages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
