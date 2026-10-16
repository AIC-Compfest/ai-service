"""PyMuPDF table provider for selectable-text PDF pages.

Geometry comes from ``table.rows[r].cells[c]`` (PyMuPDF's own grid). The flat ``table.cells`` list is NOT a
dense row-major grid and must never be indexed as one (that was the 96.7 percent wrong-bbox bug, T-11).
"""

from __future__ import annotations

from statistics import median

from app.services.candidates.schema import Cell, Column, Page, Table, TableRow


# Defaults are overridden by the ``pymupdf`` section of config/candidates/table_resolution.yaml.
DEFAULT_OPTIONS: dict = {
    "strategies": ["lines", "text"],
    "merge_strategies": False,  # False: use the first strategy that yields an accepted table
    "max_page_area_ratio": 1.0,  # reject tables covering more than this share of the page (1.0 = off)
    "min_data_rows": 1,
    "min_cols": 2,
    "max_header_chars": 0,  # reject tables whose header cell holds a paragraph (0 = off)
    "merge_overlap": 0.3,  # a later-strategy table overlapping an earlier one by more than this is dropped
}


class PyMuPDFTableProvider:
    """Normalize ``Page.find_tables`` results into the internal Table model."""

    name = "pymupdf_find_tables"

    def __init__(self, options: dict | None = None):
        self.options = {**DEFAULT_OPTIONS, **(options or {})}

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        source = page.source_page
        if source is None or not hasattr(source, "find_tables"):
            return []
        accepted: list[Table] = []
        for strategy in self.options["strategies"]:
            try:
                found = source.find_tables(strategy=strategy).tables
            except Exception:
                continue
            tables = _normalize(found, page, document_id, self.options, start=len(accepted) + 1)
            if not tables:
                continue
            for table in tables:
                table.reason = f"pymupdf_find_tables:{strategy}"
            if not self.options["merge_strategies"]:
                return tables
            for table in tables:
                if not any(_overlap(table.bbox, other.bbox) > self.options["merge_overlap"] for other in accepted):
                    accepted.append(table)
        for index, table in enumerate(accepted, start=1):
            table.table_id = f"{document_id}-p{page.page_number}-t{index}"
        return accepted


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a, b) -> float:
    """Intersection area over the SMALLER table, so a small table inside a big one counts as overlapping."""
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller else 0.0


def _clean(value) -> str:
    """Collapse internal line breaks so wrapped cells read as one line (same as the other providers)."""
    return " ".join(str(value).split()) if value else ""


def _normalize(found, page: Page, document_id: str, options: dict | None = None, start: int = 1) -> list[Table]:
    options = {**DEFAULT_OPTIONS, **(options or {})}
    width, height = float(page.source_page.rect.width), float(page.source_page.rect.height)
    output, seen = [], set()
    for found_table in found:
        x1, y1, x2, y2 = found_table.bbox
        key = tuple(round(value, 1) for value in (x1, y1, x2, y2))
        if key in seen:
            continue
        seen.add(key)
        built = _build_table(found_table, page, document_id, start + len(output), width, height, options)
        if built is not None:
            output.append(built)
    return output


def _build_table(found_table, page: Page, document_id: str, ordinal: int, width: float, height: float, options: dict) -> Table | None:
    grid = list(getattr(found_table, "rows", None) or [])
    if not grid:
        return None
    n_cols = max(len(row.cells) for row in grid)
    texts = [[_clean(value) for value in row] for row in found_table.extract()]
    if len(texts) != len(grid):
        return None
    external = bool(getattr(found_table.header, "external", False))
    names = [_clean(name) for name in getattr(found_table.header, "names", [])]
    if external:
        # Header text sits above the table and is not part of ``rows``; keep it only if it lines up with the columns.
        header_texts = names if len(names) == n_cols else [""] * n_cols
        data_start = 0
    else:
        header_texts = texts[0] + [""] * (n_cols - len(texts[0]))
        data_start = 1
    data_rows = list(range(data_start, len(grid)))
    if n_cols < options["min_cols"] or len(data_rows) < options["min_data_rows"]:
        return None
    if options["max_header_chars"] and any(len(text) > options["max_header_chars"] for text in header_texts):
        return None

    columns = _columns(grid, n_cols, header_texts, found_table.bbox, width)
    table_bbox = _norm_bbox(*found_table.bbox, width, height)
    if _area(table_bbox) > options["max_page_area_ratio"]:
        return None
    rows = []
    for out_index, grid_index in enumerate(data_rows):
        row_geometry = grid[grid_index]
        cells = []
        for c in range(n_cols):
            raw = row_geometry.cells[c] if c < len(row_geometry.cells) else None
            if raw is not None:
                bbox = _norm_bbox(*raw, width, height)
            else:  # merged/missing cell: fall back to the column extent over the row extent
                top, bottom = _row_y(row_geometry, height)
                bbox = (columns[c].x1, top, columns[c].x2, bottom)
            text = texts[grid_index][c] if c < len(texts[grid_index]) else ""
            cells.append(Cell(text, bbox, row_idx=out_index, col_idx=c))
        if any(cell.text for cell in cells):
            rows.append(TableRow(len(rows), cells, (min(cell.bbox[0] for cell in cells), min(cell.bbox[1] for cell in cells), max(cell.bbox[2] for cell in cells), max(cell.bbox[3] for cell in cells))))
    if not rows:
        return None
    table_id = f"{document_id}-p{page.page_number}-t{ordinal}"
    return Table(table_id, page.page_number, page.page_number, "grid", "pymupdf_find_tables", columns, rows, table_bbox, conf=0.0, stitched_pages=[page.page_number], reason="pymupdf_find_tables")


def _row_y(row_geometry, height: float) -> tuple[float, float]:
    bbox = row_geometry.bbox
    return (max(0.0, min(1.0, bbox[1] / height)), max(0.0, min(1.0, bbox[3] / height)))


def _columns(grid, n_cols: int, header_texts: list[str], table_bbox, width: float) -> list[Column]:
    """Column x-extents from the real cell grid: the median left/right edge of the non-empty cells in each column."""
    columns = []
    for c in range(n_cols):
        lefts = [row.cells[c][0] for row in grid if c < len(row.cells) and row.cells[c] is not None]
        rights = [row.cells[c][2] for row in grid if c < len(row.cells) and row.cells[c] is not None]
        if lefts:
            x1, x2 = median(lefts), median(rights)
        else:  # fully merged column: split the table width evenly as a last resort
            span = (table_bbox[2] - table_bbox[0]) / n_cols
            x1, x2 = table_bbox[0] + c * span, table_bbox[0] + (c + 1) * span
        columns.append(Column(c, max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, x2 / width)), header_texts[c], [header_texts[c]]))
    return columns


def _norm_bbox(x1, y1, x2, y2, width, height):
    """Clip provider geometry to the established normalized page bounds."""
    left, right = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    top, bottom = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return (left / width, top / height, right / width, bottom / height)
