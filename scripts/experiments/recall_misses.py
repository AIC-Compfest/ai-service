"""List, per invoice, which GT amounts the router's table cells miss, and the page line that holds each miss.

Usage (from an ai-service checkout): python scripts\\experiments\\recall_misses.py <out.json>
Run it in two checkouts (e.g. a git worktree of the baseline tag) and diff the files to see what a change lost.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
from _common import CONFIG_DIR, INVOICE_PDF_DIR, digits, gt_amounts, load_invoice_gt, pdf_key  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.candidates.context import build_page  # noqa: E402
from app.services.candidates.tables import TableExtractor  # noqa: E402
from app.services.ocr.engine import OCREngine  # noqa: E402
from app.services.preprocessing.image_preprocessor import ImagePreprocessor  # noqa: E402

gt = load_invoice_gt()
engine, prep = OCREngine(), ImagePreprocessor()
router = TableExtractor.from_config(CONFIG_DIR, settings.table_provider_order)
out = {}
for pdf in sorted(INVOICE_PDF_DIR.glob("*.pdf")):
    key = pdf_key(pdf.name)
    if key not in gt:
        continue
    cells, lines, tables = set(), [], []
    for number, page_obj in enumerate(fitz.open(pdf), 1):
        png = page_obj.get_pixmap(dpi=100).tobytes("png")
        quality = prep.estimate_quality(cv2.imdecode(np.frombuffer(png, np.uint8), 1))
        evidence = engine.process_pdf_page(page_obj, number, quality, canonical_image_bytes=png).to_dict()
        lines.extend(line["text"] for line in evidence["lines"])
        page = build_page({"page_number": number, "lines": evidence["lines"], "_source_page": page_obj})
        for table in router.extract(page, "INVOICE", "d"):
            tables.append({"bbox": [round(v, 2) for v in table.bbox], "rows": len(table.rows), "role": getattr(table, "role", ""), "provider": table.source})
            for row in table.rows:
                cells.update(digits(c.text) for c in row.cells)
    missing = []
    for amount in gt_amounts(gt[key]):
        if digits(amount) not in cells:
            holder = next((text for text in lines if digits(amount) and digits(amount) in digits(text)), None)
            missing.append({"amount": str(amount), "line": holder})
    out[key] = {"missing": missing, "tables": tables}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("docs", len(out), "docs with a miss", sum(1 for v in out.values() if v["missing"]), "misses", sum(len(v["missing"]) for v in out.values()))
