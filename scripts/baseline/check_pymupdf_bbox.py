"""Compare PyMuPDFTableProvider cell bboxes with PyMuPDF's own table.rows[r].cells[c].

Run from ai-service:  .\\.venv\\Scripts\\python.exe scripts\\baseline\\check_pymupdf_bbox.py
Reference tables are matched by table bbox (tolerance 0.002), not by order.
"""
import os

import pymupdf
from _common import INVOICE_PDF_DIR

from app.services.candidates.context import build_page
from app.services.candidates.tables.pymupdf_provider import PyMuPDFTableProvider

tables = compared = bad = skipped = 0
examples = []

for pdf in sorted(INVOICE_PDF_DIR.glob("*.pdf")):
    for pn, pg in enumerate(pymupdf.open(pdf), start=1):
        W, H = pg.rect.width, pg.rect.height
        page = build_page({"page_number": pn, "lines": [], "_source_page": pg})
        ours = PyMuPDFTableProvider().tables(page, "INVOICE", "d")
        ref_tables = []
        for strat in ("lines", "text"):  # same strategies as the provider
            try:
                ref_tables = pg.find_tables(strategy=strat).tables
            except Exception:
                continue
            if ref_tables:
                break
        for t in ours:
            ref = next(
                (f for f in ref_tables
                 if all(abs(a - b) < 0.002
                        for a, b in zip((f.bbox[0] / W, f.bbox[1] / H, f.bbox[2] / W, f.bbox[3] / H), t.bbox))),
                None,
            )
            if ref is None:
                skipped += 1
                continue
            tables += 1
            ext = bool(getattr(ref.header, "external", False))
            for r, row in enumerate(t.rows):
                rr = r + (0 if ext else 1)  # first data row is rows[1] when the header is internal
                if rr >= len(ref.rows):
                    continue
                for c, cell in enumerate(row.cells):
                    rc = ref.rows[rr].cells[c] if c < len(ref.rows[rr].cells) else None
                    if rc is None:
                        continue
                    compared += 1
                    nb = (rc[0] / W, rc[1] / H, rc[2] / W, rc[3] / H)
                    if any(abs(a - b) > 0.005 for a, b in zip(nb, cell.bbox)):
                        bad += 1
                        if len(examples) < 3:
                            examples.append((os.path.basename(str(pdf)), pn, r, c, cell.text))

print(f"tables matched={tables} skipped={skipped} cells compared={compared} wrong bbox={bad} ({bad / max(compared, 1):.1%})")
for e in examples:
    print("example:", e)
