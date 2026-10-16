"""Render a page with table evidence drawn on top (for human spot-checks and debugging).

Legend: red box = table bbox, blue lines = column boundaries (x1/x2 of each column),
green boxes = cells, orange text = header texts, magenta box = summary rows,
numbers at the left edge = row index.

Usage (from ai-service):
    .\\.venv\\Scripts\\python.exe scripts\\review_overlay.py <pdf> --page 1 --out out/review/x.png [--tables on]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

RED, BLUE, GREEN, ORANGE, MAGENTA = (220, 30, 30), (30, 90, 220), (30, 160, 60), (230, 130, 0), (190, 30, 190)


def _px(box, w, h):
    return (box[0] * w, box[1] * h, box[2] * w, box[3] * h)


def draw_tables(image: Image.Image, tables: list[dict], *, title: str = "") -> Image.Image:
    """Draw evidence tables (dicts as produced by ``table_evidence``) on a copy of ``image``."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    if title:
        draw.text((6, 4), title, fill=(0, 0, 0))
    for t_index, table in enumerate(tables):
        box = _px(table["bbox_norm"], w, h)
        draw.rectangle(box, outline=RED, width=3)
        role = table.get("role", "")
        draw.text((box[0] + 3, max(box[1] - 12, 0)), f"T{t_index + 1} {table.get('provider', '')} {role}", fill=RED)
        for column in table.get("columns", []):
            x1, x2 = column["bbox_norm"][0] * w, column["bbox_norm"][2] * w
            for x in (x1, x2):
                draw.line([(x, box[1]), (x, box[3])], fill=BLUE, width=1)
            draw.text((x1 + 2, box[1] + 2), str(column.get("header", ""))[:18], fill=ORANGE)
        for cell in table.get("cells", []):
            draw.rectangle(_px(cell["bbox_norm"], w, h), outline=GREEN, width=1)
        seen_rows: set[int] = set()
        for cell in table.get("cells", []):
            if cell["row"] not in seen_rows:
                seen_rows.add(cell["row"])
                draw.text((max(box[0] - 14, 0), cell["bbox_norm"][1] * h), str(cell["row"]), fill=BLUE)
        for summary in table.get("summary_cells", []) or []:
            draw.rectangle(_px(summary["bbox_norm"], w, h), outline=MAGENTA, width=2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--doc-type", default=None, help="Optional document type for alias-aware discovery")
    args = parser.parse_args()

    import pymupdf

    from app.core.config import settings
    from app.services.candidates.context import build_page
    from app.services.candidates.pipeline import table_evidence
    from app.services.candidates.tables import TableExtractor
    from app.services.ocr.engine import OCREngine

    settings.enable_table_extractor = True
    doc = pymupdf.open(args.pdf)
    pg = doc[args.page - 1]
    pix = pg.get_pixmap(dpi=args.dpi)
    png = pix.tobytes("png")
    from io import BytesIO

    image = Image.open(BytesIO(png))
    import cv2
    import numpy as np

    from app.services.preprocessing.image_preprocessor import ImagePreprocessor

    quality = ImagePreprocessor.estimate_quality(cv2.imdecode(np.frombuffer(png, np.uint8), 1))
    ev = OCREngine().process_pdf_page(pg, args.page, quality, canonical_image_bytes=png).to_dict()
    page = build_page({"page_number": args.page, "lines": ev["lines"], "_source_page": pg})
    extractor = TableExtractor.from_config(AI_ROOT / "config" / "candidates", settings.table_provider_order)
    tables = [table_evidence(t) for t in extractor.extract(page, args.doc_type, "overlay")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw_tables(image, tables, title=f"{args.pdf.name} p{args.page} tables={len(tables)}").save(args.out)
    print(f"{args.pdf.name} p{args.page}: {len(tables)} table(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
