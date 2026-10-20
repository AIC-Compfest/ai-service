"""Adapter: Docling ``TableItem`` -> internal ``Table`` (step 4 of the Docling flow).

Pure functions over duck-typed Docling objects (``.prov``, ``.data.table_cells`` ...), so they are unit-testable
without Docling installed. Everything leaving this module uses the internal contract: normalized [0, 1] boxes with a
top-left origin, ``Cell`` / ``Column`` / ``TableRow`` / ``Table``.

Docling mixes coordinate origins (table provenance boxes are BOTTOMLEFT while cell boxes are TOPLEFT in the same
result), so every box is converted with its own ``coord_origin``.
"""
from __future__ import annotations

from statistics import median
from typing import Any

from app.services.candidates.schema import Cell, Column, Table, TableRow

Box = tuple[float, float, float, float]


def normalize_box(bbox: Any, width: float, height: float) -> Box | None:
    """Docling ``BoundingBox`` -> normalized (x1, y1, x2, y2), top-left origin, clipped to the page; None if unusable."""
    if bbox is None or width <= 0 or height <= 0:
        return None
    try:
        left, right = float(bbox.l), float(bbox.r)
        top, bottom = float(bbox.t), float(bbox.b)
    except (AttributeError, TypeError, ValueError):
        return None
    origin = getattr(getattr(bbox, "coord_origin", None), "name", str(getattr(bbox, "coord_origin", "")))
    if "BOTTOMLEFT" in origin.upper():
        top, bottom = height - top, height - bottom
    x1, x2 = sorted((max(0.0, min(width, left)), max(0.0, min(width, right))))
    y1, y2 = sorted((max(0.0, min(height, top)), max(0.0, min(height, bottom))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def _clean(value: Any) -> str:
    return " ".join(str(value).split()) if value else ""


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def docling_table_to_internal(
    item: Any,
    *,
    page_width: float,
    page_height: float,
    page_number: int,
    document_id: str,
    ordinal: int,
    reason: str = "docling",
    min_cols: int = 2,
    min_data_rows: int = 1,
) -> Table | None:
    """Convert one Docling table; None when it is not a usable grid (too few columns/rows, no geometry)."""
    data = getattr(item, "data", None)
    cells = list(getattr(data, "table_cells", None) or [])
    n_rows, n_cols = int(getattr(data, "num_rows", 0) or 0), int(getattr(data, "num_cols", 0) or 0)
    if not cells or n_cols < min_cols or n_rows < 1:
        return None

    texts = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    header_flag = [[False] * n_cols for _ in range(n_rows)]
    boxes: dict[tuple[int, int], Box] = {}
    for cell in cells:
        r, c = int(cell.start_row_offset_idx), int(cell.start_col_offset_idx)
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue
        text = _clean(cell.text)
        texts[r][c] = f"{texts[r][c]} {text}".strip() if texts[r][c] else text
        # A spanning header cell flags every grid position it covers.
        for rr in range(r, min(int(cell.end_row_offset_idx), n_rows)):
            for cc in range(c, min(int(cell.end_col_offset_idx), n_cols)):
                header_flag[rr][cc] = header_flag[rr][cc] or bool(getattr(cell, "column_header", False))
        box = normalize_box(getattr(cell, "bbox", None), page_width, page_height)
        if box is not None and (r, c) not in boxes:
            boxes[(r, c)] = box

    prov = (getattr(item, "prov", None) or [None])[0]
    table_box = normalize_box(getattr(prov, "bbox", None), page_width, page_height)
    if table_box is None:
        if not boxes:
            return None
        table_box = _union(list(boxes.values()))

    # Leading rows whose non-empty cells are all flagged as column headers; at least one body row must remain.
    header_rows = 0
    while header_rows < n_rows - 1 and any(texts[header_rows]) and all(header_flag[header_rows][c] for c in range(n_cols) if texts[header_rows][c]):
        header_rows += 1
    if n_rows - header_rows < min_data_rows:
        return None

    # Geometry: column x-extents and row y-extents from the real cell boxes, table box as the last resort.
    columns_x: list[tuple[float, float]] = []
    for c in range(n_cols):
        xs = [(b[0], b[2]) for (r, cc), b in boxes.items() if cc == c]
        if xs:
            columns_x.append((median(x[0] for x in xs), median(x[1] for x in xs)))
        else:
            span = (table_box[2] - table_box[0]) / n_cols
            columns_x.append((table_box[0] + c * span, table_box[0] + (c + 1) * span))
    rows_y: list[tuple[float, float]] = []
    for r in range(n_rows):
        ys = [(b[1], b[3]) for (rr, c), b in boxes.items() if rr == r]
        if ys:
            rows_y.append((min(y[0] for y in ys), max(y[1] for y in ys)))
        else:
            span = (table_box[3] - table_box[1]) / n_rows
            rows_y.append((table_box[1] + r * span, table_box[1] + (r + 1) * span))

    header_lines_per_col = [[texts[r][c] for r in range(header_rows) if texts[r][c]] for c in range(n_cols)]
    columns = [
        Column(c, columns_x[c][0], columns_x[c][1], " ".join(header_lines_per_col[c]), header_lines_per_col[c] or [""])
        for c in range(n_cols)
    ]

    rows: list[TableRow] = []
    for r in range(header_rows, n_rows):
        row_cells: list[Cell] = []
        for c in range(n_cols):
            bbox = boxes.get((r, c)) or (columns_x[c][0], rows_y[r][0], columns_x[c][1], rows_y[r][1])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bbox = (columns_x[c][0], rows_y[r][0], max(columns_x[c][1], columns_x[c][0] + 1e-6), max(rows_y[r][1], rows_y[r][0] + 1e-6))
            row_cells.append(Cell(texts[r][c], bbox, row_idx=len(rows), col_idx=c))
        if any(cell.text for cell in row_cells):
            rows.append(TableRow(len(rows), row_cells, _union([cell.bbox for cell in row_cells])))
    if len(rows) < min_data_rows:
        return None

    return Table(
        f"{document_id}-p{page_number}-t{ordinal}", page_number, page_number, "grid", "docling",
        columns, rows, table_box, conf=0.0, stitched_pages=[page_number], reason=reason,
    )
