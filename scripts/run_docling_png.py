import json
import os
import sys
from pathlib import Path

os.environ["ENABLE_TABLE_EXTRACTOR"] = "true"

from app.core.config import settings
settings.enable_table_extractor = True

from app.services.pipeline import Phase5DocumentPipeline

img_path = Path(r"D:\AIC\ai-service\out\contoh_invoice_1_variasi_2.png")
if not img_path.exists():
    print(f"File {img_path} not found!")
    sys.exit(1)

print(f"Processing image through Phase 5 pipeline through Docling + RapidOCR: {img_path}")
pipeline = Phase5DocumentPipeline()
evidence = pipeline.process_document(img_path, filename=img_path.name, document_id="DOC-DOCLING-TEST")

out_file = Path(r"D:\AIC\ai-service\out\test_evidence_docling_png_output.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2, ensure_ascii=False)

print("SUCCESS: Result saved to", out_file)
