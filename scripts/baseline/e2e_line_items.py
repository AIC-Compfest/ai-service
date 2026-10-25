"""End-to-end: Phase5DocumentPipeline + InvoiceExtractor with ENABLE_TABLE_EXTRACTOR on/off.

Run from ai-service:  .\\.venv\\Scripts\\python.exe scripts\\baseline\\e2e_line_items.py on|off
Reports documents with tables, extracted vs GT line items, and documents with an exact row-count match.
Ingestion writes PNGs to out/pages/<document_id> (relative to CWD); document_ids are unique per mode.
"""
import json
import sys

from _common import AI_ROOT, INVOICE_PDF_DIR, load_invoice_gt, pdf_key

from app.core.config import settings
from app.services.extraction.invoice import InvoiceExtractor
from app.services.pipeline import Phase5DocumentPipeline

mode = sys.argv[1]
assert mode in ("on", "off"), "usage: e2e_line_items.py on|off"
settings.enable_table_extractor = mode == "on"

gt = load_invoice_gt()
pipeline, extractor = Phase5DocumentPipeline(), InvoiceExtractor()
rows = []
for pdf in sorted(INVOICE_PDF_DIR.glob("*.pdf")):
    key = pdf_key(pdf.name)
    if key not in gt:
        continue
    evidence = pipeline.process_document(str(pdf), filename=pdf.name, document_id=f"{mode}_{key}")
    n_tables = sum(len(p.get("tables", [])) for p in evidence["pages"])
    fields = extractor.extract("x", evidence)
    items = [f for f in fields if f.field == "line_items"]
    n_items = len(items[0].value) if items and items[0].value else 0
    rows.append({"file": key, "tables": n_tables, "extracted": n_items,
                 "gt": len(gt[key].get("line_items", [])), "engine": evidence["pages"][0].get("engine_used")})

print("flag", mode, "docs", len(rows))
print("docs with tables:", sum(1 for r in rows if r["tables"] > 0),
      "| extracted line_items:", sum(r["extracted"] for r in rows),
      "| GT line items:", sum(r["gt"] for r in rows),
      "| exact count match:", sum(1 for r in rows if r["extracted"] == r["gt"]))
out_dir = AI_ROOT / "out" / "baseline"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"e2e_{mode}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
