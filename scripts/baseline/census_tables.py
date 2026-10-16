"""Per-page table census over the canonical DATASET (digital PDF pages only).

For every PDF page report: PyMuPDF find_tables (strategy lines / text, count and page-wide count),
token_grid tables with doc_type=None (the API path), and the router result.
Writes out/baseline/census_tables.csv and prints aggregate counts.

Run from ai-service:  .\\.venv\\Scripts\\python.exe scripts\\baseline\\census_tables.py
"""
import csv

import cv2
import numpy as np
import pymupdf
from _common import AI_ROOT, CONFIG_DIR, DATASET_ROOT

from app.core.config import settings
from app.services.candidates.context import build_page
from app.services.candidates.tables import TableExtractor
from app.services.candidates.tables.token_grid import TokenGridTableProvider
from app.services.ocr.engine import OCREngine
from app.services.preprocessing.image_preprocessor import ImagePreprocessor

DOC_TYPES = {"invoice": "INVOICE", "PoD": "POD", "rate card": "RATE_AGREEMENT", "surat jalan": "SURAT_JALAN"}
engine, prep = OCREngine(), ImagePreprocessor()
grid = TokenGridTableProvider(CONFIG_DIR / "line_item_aliases.yaml", CONFIG_DIR / "abbreviations.yaml")
router = TableExtractor.from_config(CONFIG_DIR, settings.table_provider_order)


def pymupdf_tables(pg, strategy):
    try:
        return pg.find_tables(strategy=strategy).tables
    except Exception:
        return []


def wide(tables, W, H):
    return sum(1 for t in tables if (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]) / (W * H) > 0.6)


rows = []
for folder, doc_type in DOC_TYPES.items():
    for pdf in sorted((DATASET_ROOT / folder / "pdf").glob("*.pdf")):
        for pn, pg in enumerate(pymupdf.open(pdf), 1):
            W, H = pg.rect.width, pg.rect.height
            words = pg.get_text("words")
            rec = {"type": doc_type, "file": pdf.name, "page": pn, "words": len(words)}
            if len(words) < 5:
                rec.update(kind="raster_needed")
                rows.append(rec)
                continue
            png = pg.get_pixmap(dpi=100).tobytes("png")
            q = prep.estimate_quality(cv2.imdecode(np.frombuffer(png, np.uint8), 1))
            ev = engine.process_pdf_page(pg, pn, q, canonical_image_bytes=png).to_dict()
            page = build_page({"page_number": pn, "lines": ev["lines"], "_source_page": pg})
            lines_t, text_t = pymupdf_tables(pg, "lines"), pymupdf_tables(pg, "text")
            rec.update(
                kind="digital",
                mu_lines=len(lines_t), mu_lines_wide=wide(lines_t, W, H),
                mu_text=len(text_t), mu_text_wide=wide(text_t, W, H),
                grid_none=len(grid.tables(page, None, "c")),
                grid_typed=len(grid.tables(page, doc_type, "c")),
                router_none=len(router.extract(page, None, "c")),
            )
            rows.append(rec)

out = AI_ROOT / "out" / "baseline"
out.mkdir(parents=True, exist_ok=True)
fields = ["type", "file", "page", "words", "kind", "mu_lines", "mu_lines_wide", "mu_text", "mu_text_wide", "grid_none", "grid_typed", "router_none"]
with (out / "census_tables.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

digital = [r for r in rows if r["kind"] == "digital"]
print("pages:", len(rows), "digital:", len(digital), "raster_needed:", len(rows) - len(digital))
for doc_type in DOC_TYPES.values():
    sub = [r for r in digital if r["type"] == doc_type]
    if not sub:
        continue
    print(
        doc_type, "pages", len(sub),
        "| pymupdf(lines) pages>=1:", sum(r["mu_lines"] >= 1 for r in sub), ">=2:", sum(r["mu_lines"] >= 2 for r in sub),
        "| pymupdf(text) >=1:", sum(r["mu_text"] >= 1 for r in sub), ">=2:", sum(r["mu_text"] >= 2 for r in sub),
        "| grid(None) tables:", sum(r["grid_none"] for r in sub), "pages>=2:", sum(r["grid_none"] >= 2 for r in sub),
        "| router(None) tables:", sum(r["router_none"] for r in sub),
    )
