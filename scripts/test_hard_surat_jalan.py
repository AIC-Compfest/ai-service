import os
import sys
import json
from pathlib import Path

# Add ai-service to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.ocr.engine import OCREngine, to_evidence_json
from app.services.candidates.pipeline import CandidatePipeline

def test_hard_surat_jalan():
    img_path = Path(r"d:\AIC\DATASET\surat jalan\contoh-surat-jalan-10.png")
    if not img_path.exists():
        print(f"File tidak ditemukan: {img_path}")
        return

    print("=" * 70)
    print("DETAILED HARD CASE SCAN TESTING: contoh-surat-jalan-10.png")
    print("=" * 70)

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    engine = OCREngine()
    evidence = engine.process_image(img_bytes, page_number=1, quality={})

    print(f"Engine Digunakan     : {evidence.engine_used}")
    print(f"Dimensi Dokumen     : {evidence.width} x {evidence.height} px")
    print(f"Rata-rata Akurasi   : {evidence.average_confidence():.4f}")
    print(f"Total Baris Teks    : {len(evidence.lines)}")

    print("\n" + "-" * 70)
    print("DAFTAR SELURUH TEKS HASIL EKSTRAKSI OCR (Line & Tokens & Confidence):")
    print("-" * 70)
    for i, line in enumerate(evidence.lines, 1):
        toks = " | ".join(f"'{t.text}' ({t.confidence:.2f})" for t in line.tokens)
        print(f"[{i:02d}] (Conf: {line.confidence:.2f}) [bbox: {[round(c, 1) for c in line.bbox_norm]}]: {line.text}")
        print(f"     Tokens: {toks}")

    # Build evidence and feed to Candidate Pipeline for SURAT_JALAN
    evidence_json = to_evidence_json([evidence])
    pipeline = CandidatePipeline(config_dir=Path("config/candidates"))

    candidates = pipeline.generate(
        document_id="contoh-surat-jalan-10",
        doc_type="SURAT_JALAN",
        ocr_result=evidence_json,
        run_id="hard_scan_audit"
    )

    print("\n" + "-" * 70)
    print("HASIL EKSTRAKSI FIELD & VALUE OLEH CANDIDATE GENERATION:")
    print("-" * 70)
    print(f"Total Kandidat yang berhasil digenerate: {len(candidates)}")

    by_type = {}
    for c in candidates:
        ct = c["m_cand_type"]
        by_type.setdefault(ct, []).append(c)

    for cand_type, items in by_type.items():
        print(f"\n--- [Tipe Field: {cand_type.upper()}] ({len(items)} kandidat) ---")
        for idx, item in enumerate(items, 1):
            val = item.get("m_parsed_value") or item.get("m_raw_text")
            raw = item.get("m_raw_text")
            conf = item.get("q_conf_mean", 0.0)
            bbox = [round(item.get("m_x1", 0), 3), round(item.get("m_y1", 0), 3), round(item.get("m_x2", 0), 3), round(item.get("m_y2", 0), 3)]
            ctx_line = item.get("ctx_line", "")
            print(f"  {idx:02d}. Value: '{val}' (Raw: '{raw}') | Conf: {conf:.2f} | BBox: {bbox}")
            if ctx_line:
                print(f"      Context Baris: \"{ctx_line}\"")

if __name__ == "__main__":
    test_hard_surat_jalan()
