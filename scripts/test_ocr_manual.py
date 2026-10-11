import os
import sys

# Tambahkan direktori root ai-service ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from app.services.ocr.engine import OCREngine

def main():
    img_path = r"d:\AIC\DATASET\invoice\Contoh-Template-Invoice-2.jpg"
    if not os.path.exists(img_path):
        print(f"File tidak ditemukan: {img_path}")
        return

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    print(f"Loading image ({len(img_bytes)} bytes)...")
    engine = OCREngine()
    evidence = engine.process_image(img_bytes, page_number=1, quality={})

    print("\n" + "="*50)
    print("HASIL TESTING OCR ENGINE (MANUAL)")
    print("="*50)
    print(f"Engine Used       : {evidence.engine_used}")
    print(f"Fallback Used     : {evidence.fallback_used}")
    print(f"Dimensi Dokumen   : {evidence.width} x {evidence.height} px")
    print(f"Rata-rata Conf.   : {evidence.average_confidence():.4f}")
    print(f"Total Baris Teks  : {len(evidence.lines)}")
    print(f"Total Region      : {len(evidence.regions)}")
    print(f"Total Tabel       : {len(evidence.tables)}")

    print("\n" + "-"*50)
    print("FULL EXTRACTED TEXT:")
    print("-"*50)
    print(evidence.full_text())

    print("\n" + "-"*50)
    print("DETAIL PER BARIS TEKS (OCRLine):")
    print("-"*50)
    for i, line in enumerate(evidence.lines, 1):
        print(f"{i:2d}. [{line.confidence:.2f}] (bbox: {[round(c, 1) for c in line.bbox_norm]}): {line.text}")
        if line.tokens:
            tok_str = ", ".join(f"'{t.text}' ({t.confidence:.2f})" for t in line.tokens)
            print(f"    Tokens: {tok_str}")

    print("\n" + "-"*50)
    print("LAYOUT REGIONS:")
    print("-"*50)
    for r in evidence.regions:
        print(f"- Type: {r.region_type:<12} | Conf: {r.confidence:.2f} | Bbox: {[round(c, 1) for c in r.bbox_norm]} | Snippet: {r.text[:60]}")

    print("\n" + "-"*50)
    print("TABLE STRUCTURES:")
    print("-"*50)
    if not evidence.tables:
        print("Tidak ada tabel terdeteksi struktur eksplisit.")
    else:
        for idx, t in enumerate(evidence.tables, 1):
            print(f"Tabel #{idx}:")
            print(f"  Headers: {t.headers}")
            print(f"  Jumlah baris data: {len(t.rows)}")
            for r_idx, row in enumerate(t.rows, 1):
                print(f"    Row {r_idx}: {row}")

if __name__ == "__main__":
    main()
