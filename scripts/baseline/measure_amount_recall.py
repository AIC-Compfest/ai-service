"""Rough amount recall per provider (token_grid / pymupdf / router) on digital invoice PDFs.

Run from ai-service:  .\\.venv\\Scripts\\python.exe scripts\\baseline\\measure_amount_recall.py
GT is SYNTHETIC (not human gold). Metric is ROUGH: the digits of each GT amount must equal the digits of
some produced table cell. This is not a cell/row metric.
"""
import cv2
import fitz
import numpy as np
from _common import CONFIG_DIR, INVOICE_PDF_DIR, digits, gt_amounts, load_invoice_gt, pdf_key

from app.core.config import settings
from app.services.candidates.context import build_page
from app.services.candidates.tables import TableExtractor
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider
from app.services.candidates.tables.token_grid import TokenGridTableProvider
from app.services.ocr.engine import OCREngine
from app.services.preprocessing.image_preprocessor import ImagePreprocessor

gt = load_invoice_gt()
eng, prep = OCREngine(), ImagePreprocessor()
providers = {
    "token_grid": TokenGridTableProvider(CONFIG_DIR / "line_item_aliases.yaml", CONFIG_DIR / "abbreviations.yaml"),
    "pymupdf": PyMuPDFTableProvider(),
}
router = TableExtractor.from_config(CONFIG_DIR, settings.table_provider_order)

res = {k: [0, 0] for k in ("token_grid", "pymupdf", "router")}
docs = pages = pagefull = pages_tab = 0
missing_gt = []

for pdf in sorted(INVOICE_PDF_DIR.glob("*.pdf")):
    key = pdf_key(pdf.name)
    if key not in gt:
        missing_gt.append(pdf.name)
        continue
    docs += 1
    amounts = gt_amounts(gt[key])
    cells = {k: set() for k in res}
    for pn, pg in enumerate(fitz.open(pdf), 1):
        png = pg.get_pixmap(dpi=100).tobytes("png")
        quality = prep.estimate_quality(cv2.imdecode(np.frombuffer(png, np.uint8), 1))
        ev = eng.process_pdf_page(pg, pn, quality, canonical_image_bytes=png).to_dict()
        page = build_page({"page_number": pn, "lines": ev["lines"], "_source_page": pg})
        pages += 1
        for name, provider in providers.items():
            for t in provider.tables(page, "INVOICE", "d"):
                if name == "pymupdf":
                    pages_tab += 1
                    if (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]) > 0.6:
                        pagefull += 1
                for r in t.rows:
                    for c in r.cells:
                        cells[name].add(digits(c.text))
        for t in router.extract(page, "INVOICE", "d"):
            for r in t.rows:
                for c in r.cells:
                    cells["router"].add(digits(c.text))
    for name in res:
        for a in amounts:
            res[name][1] += 1
            if digits(a) in cells[name]:
                res[name][0] += 1

print("docs", docs, "pages", pages, "| PDFs without GT:", len(missing_gt))
for k, (hit, n) in res.items():
    print(k, f"{hit}/{n} = {hit / max(n, 1):.1%}")
print("pymupdf tables covering >60% page:", pagefull, "of", pages_tab, "tables")
