import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Set environment
os.environ["ENABLE_DOCLING"] = "true"
os.environ["ENABLE_TABLE_EXTRACTOR"] = "true"

from app.services.pipeline import Phase5DocumentPipeline
from app.services.ocr.docling_engine import DoclingOCREngine
from app.services.preprocessing.document_ingestion import DocumentIngestionService
from app.services.preprocessing.image_preprocessor import ImagePreprocessor

def test_single_pdf_engine():
    print("=== TEST 1: Direct DoclingOCREngine on PDF ===")
    pdf_path = Path("D:/AIC/DATASET/Dataset/invoice/pdf/invoice_1_variasi_2.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return
    
    ingestion = DocumentIngestionService()
    ingested = ingestion.ingest_file(pdf_path)
    page_asset = ingested.pages[0]
    
    engine = DoclingOCREngine()
    preprocessor = ImagePreprocessor()
    cv_img = ImagePreprocessor.bytes_to_cv2(page_asset.image_bytes)
    quality = preprocessor.estimate_quality(cv_img)
    
    t0 = time.perf_counter()
    res = engine.process(
        page_asset.pdf_bytes,
        is_pdf=True,
        page_number=page_asset.page_number,
        quality=quality,
        page_width=page_asset.width,
        page_height=page_asset.height,
        fallback_image_bytes=page_asset.image_bytes,
    )
    t1 = time.perf_counter()
    print(f"Engine process time: {round((t1-t0)*1000, 2)} ms")
    print(f"engine_used: {res.get('engine_used')}")
    print(f"fallback_used: {res.get('fallback_used')}")
    print(f"lines count: {len(res.get('lines', []))}")
    print(f"regions count: {len(res.get('regions', []))}")
    print(f"average_confidence: {res.get('average_confidence')}")
    if res.get('lines'):
        print("Sample 3 lines:")
        for line in res['lines'][:3]:
            print(f"  - text: '{line['text']}', conf: {line['confidence']}, box: {line['bbox_norm']}, tokens: {len(line['tokens'])}")
    return res

def test_single_image_engine():
    print("\n=== TEST 2: Direct DoclingOCREngine on PNG Image ===")
    img_path = Path("D:/AIC/ai-service/out/contoh_invoice_1_variasi_2.png")
    if not img_path.exists():
        print(f"Image not found at {img_path}")
        return
    
    img_bytes = img_path.read_bytes()
    engine = DoclingOCREngine()
    
    try:
        t0 = time.perf_counter()
        res = engine.process(
            img_bytes,
            is_pdf=False,
            page_number=1,
            quality={"sharpness": 100, "dpi": 200},
            page_width=1654,
            page_height=2339,
            fallback_image_bytes=img_bytes,
        )
        t1 = time.perf_counter()
        print(f"Image engine process time: {round((t1-t0)*1000, 2)} ms")
        print(f"engine_used: {res.get('engine_used')}")
        print(f"lines count: {len(res.get('lines', []))}")
        print(f"regions count: {len(res.get('regions', []))}")
        if res.get('lines'):
            for line in res['lines'][:3]:
                print(f"  - text: '{line['text']}', tokens: {len(line['tokens'])}")
    except Exception as e:
        print(f"Image OCR failed with error: {e}")

def test_full_pipeline():
    print("\n=== TEST 3: Full Phase5DocumentPipeline ===")
    pdf_path = Path("D:/AIC/DATASET/Dataset/invoice/pdf/invoice_1_variasi_2.pdf")
    pipeline = Phase5DocumentPipeline()
    
    t0 = time.perf_counter()
    evidence = pipeline.process_document(pdf_path, filename=pdf_path.name)
    t1 = time.perf_counter()
    print(f"Full pipeline time: {round((t1-t0)*1000, 2)} ms")
    
    out_path = Path("D:/AIC/ai-service/out/test_evidence_docling_full_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    
    print(f"Saved full evidence to: {out_path} ({out_path.stat().st_size} bytes)")
    
    # Audit summary
    print("\n--- Evidence Audit Summary ---")
    print(f"document_id: {evidence.get('document_id')}")
    print(f"page_count: {evidence.get('page_count')}")
    pages = evidence.get("pages", [])
    print(f"pages in evidence: {len(pages)}")
    for p in pages:
        print(f"Page {p.get('page_number')}:")
        print(f"  engine_used: {p.get('engine_used')}")
        print(f"  lines: {len(p.get('lines', []))}")
        print(f"  tables: {len(p.get('tables', []))}")
        print(f"  regions: {len(p.get('regions', []))}")
        print(f"  classification: {p.get('classification', {}).get('document_type')} (conf: {p.get('classification', {}).get('confidence')})")
        print(f"  signature_present: {p.get('signature_detection', {}).get('signature_present')}")
        if p.get('tables'):
            tbl = p['tables'][0]
            print(f"  Table 0: num_rows={tbl.get('num_rows')}, num_cols={tbl.get('num_cols')}, headers={tbl.get('headers')}")

if __name__ == "__main__":
    test_single_pdf_engine()
    test_single_image_engine()
    test_full_pipeline()
