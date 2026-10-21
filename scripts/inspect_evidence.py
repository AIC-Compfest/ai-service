import json

with open("D:/AIC/ai-service/out/test_evidence_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("="*60)
print("1. DOCUMENT METADATA")
print("="*60)
print("Document ID :", data.get("document_id"))
print("Source File :", data.get("source_file"))
print("SHA256      :", data.get("source_sha256"))
print("Total Pages :", data.get("page_count"))

print("\n" + "="*60)
print("2. DOCUMENT CLASSIFICATION")
print("="*60)
cls = data.get("classification", {})
print("Predicted Type :", cls.get("document_type"))
print("Confidence     :", cls.get("confidence"))
print("Method         :", cls.get("method"))
print("Probabilities  :", cls.get("probabilities"))

print("\n" + "="*60)
print("3. SEGMENTS (BOUNDARY DETECTION)")
print("="*60)
for seg in data.get("segments", []):
    print(f"Segment #{seg.get('segment_number')}: {seg.get('document_type')} (Page {seg.get('start_page')} to {seg.get('end_page')}) | Conf: {seg.get('classification_confidence')}")

print("\n" + "="*60)
print("4. SIGNATURE & STAMP DETECTION")
print("="*60)
sig = data.get("signature_detection", {})
print("Signature Present :", sig.get("signature_present"), f"(conf: {sig.get('confidence')})")
print("BBox              :", sig.get("bbox"))

p0 = data["pages"][0]
print("\n" + "="*60)
print(f"5. PAGE 1 - PREPROCESSING & QUALITY")
print("="*60)
print("Dimensions :", f"{p0.get('width')} x {p0.get('height')}")
print("Engine Used:", p0.get('engine_used'))
print("Quality    :", p0.get('quality'))

print("\n" + "="*60)
print(f"6. OCR EXTRACTED TEXT (TOTAL {len(p0.get('lines', []))} LINES)")
print("="*60)
for idx, line in enumerate(p0.get("lines", [])):
    print(f"[{idx+1:02d}] {line.get('text')}")

print("\n" + "="*60)
print(f"7. TABLES DISCOVERED (TOTAL {len(p0.get('tables', []))} TABLES)")
print("="*60)
for idx, t in enumerate(p0.get("tables", [])):
    print(f"\n--- TABLE {idx+1}: {t.get('table_id')} ---")
    print(f"Provider: {t.get('provider')} | Dim: {t.get('row_count')} rows x {t.get('col_count')} cols | Role: {t.get('role')}")
    print(f"Headers : {t.get('headers')}")
    print("Rows Grid:")
    for r in t.get("rows", []):
        print("  ", r)
