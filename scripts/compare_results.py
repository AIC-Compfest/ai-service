import json

with open("D:/AIC/ai-service/out/test_evidence_output.json", "r", encoding="utf-8") as f:
    pdf_pymupdf = json.load(f)

with open("D:/AIC/ai-service/out/test_evidence_docling_png_output.json", "r", encoding="utf-8") as f:
    png_docling = json.load(f)

print("="*70)
print("KOMPARASI UMUM: PYMUPDF (PDF) VS DOCLING + PADDLEOCR (PNG)")
print("="*70)

p_pdf = pdf_pymupdf["pages"][0]
p_png = png_docling["pages"][0]

print(f"{'Metrik':<25} | {'PyMuPDF (PDF)':<22} | {'Docling + Paddle (PNG)':<22}")
print("-" * 75)
print(f"{'OCR Engine Halaman':<25} | {p_pdf.get('engine_used'):<22} | {p_png.get('engine_used'):<22}")
print(f"{'Total Baris Teks':<25} | {len(p_pdf.get('lines', [])):<22} | {len(p_png.get('lines', [])):<22}")
print(f"{'Klasifikasi Dokumen':<25} | {pdf_pymupdf['classification']['document_type']} ({pdf_pymupdf['classification']['confidence']}) | {png_docling['classification']['document_type']} ({png_docling['classification']['confidence']})")
print(f"{'Tanda Tangan Terdeteksi':<25} | {str(pdf_pymupdf['signature_detection']['signature_present'])} ({pdf_pymupdf['signature_detection']['confidence']}) | {str(png_docling['signature_detection']['signature_present'])} ({png_docling['signature_detection']['confidence']})")
print(f"{'Jumlah Tabel':<25} | {len(p_pdf.get('tables', [])):<22} | {len(p_png.get('tables', [])):<22}")

print("\n" + "="*70)
print("DETAIL TABEL YANG DIHASILKAN")
print("="*70)

print("\n>>> TABEL DARI PYMUPDF (test_evidence_output.json):")
for i, t in enumerate(p_pdf.get("tables", [])):
    print(f"[{i+1}] ID: {t.get('table_id')} | Provider: {t.get('provider')} | Size: {t.get('row_count')} baris x {t.get('col_count')} kolom")
    print(f"    Headers: {t.get('headers')}")
    print(f"    Sample Row: {t.get('rows', [])[:1]}")

print("\n>>> TABEL DARI DOCLING (test_evidence_docling_png_output.json):")
for i, t in enumerate(p_png.get("tables", [])):
    print(f"[{i+1}] ID: {t.get('table_id')} | Provider: {t.get('provider')} | Reason: {t.get('reason')} | Size: {t.get('row_count')} baris x {t.get('col_count')} kolom")
    print(f"    Headers: {t.get('headers')}")
    print(f"    Sample Row: {t.get('rows', [])[:1]}")

print("\n" + "="*70)
print("KOMPARASI LINE ITEMS UTAMA (BARANG / JASA)")
print("="*70)
# Tabel ke-2 di PyMuPDF adalah item line utama
tbl_pdf_items = p_pdf["tables"][1] if len(p_pdf["tables"]) > 1 else None
# Cari tabel line item di Docling
tbl_docling_items = None
for t in p_png.get("tables", []):
    if t.get("role") == "line_items" or t.get("col_count", 0) >= 4:
        tbl_docling_items = t
        break

if tbl_pdf_items:
    print("\n--- PyMuPDF Line Items Table ---")
    print("Headers:", tbl_pdf_items.get("headers"))
    for idx, r in enumerate(tbl_pdf_items.get("rows", [])):
        print(f"  Baris {idx+1}: {r}")

if tbl_docling_items:
    print("\n--- Docling Line Items Table ---")
    print("Headers:", tbl_docling_items.get("headers"))
    for idx, r in enumerate(tbl_docling_items.get("rows", [])):
        print(f"  Baris {idx+1}: {r}")
