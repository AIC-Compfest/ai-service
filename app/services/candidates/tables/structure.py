"""Geometry-first table discovery: blocks -> header -> columns -> rows -> Table objects.

Replaces the page-global row clustering of the first token-grid implementation. See geometry.py, header.py,
columns.py and rows.py for the individual decisions. Works for exact vector tokens and for raster OCR tokens.
"""
from __future__ import annotations

from ..schema import Cell, Column, Page, Table, TableRow, Token
from .columns import ColumnSpec, infer_columns
from .geometry import Line, median_height, text_blocks, vertical_rules
from .header import Header, detect_header
from .rows import Body, RowDraft, segment_body, summary_pattern

DEFAULT_PREVIEW_LINES = 60


def _union(tokens: list[Token]) -> tuple[float, float, float, float]:
    return (min(t.bbox[0] for t in tokens), min(t.bbox[1] for t in tokens), max(t.bbox[2] for t in tokens), max(t.bbox[3] for t in tokens))


def _row_cells(draft: RowDraft, columns: list[ColumnSpec], row_idx: int) -> list[Cell]:
    cells = []
    for column in columns:
        tokens = draft.cells.get(column.idx, [])  # already in reading order: line by line, left to right
        text = " ".join(t.text for t in tokens)
        bbox = _union(tokens) if tokens else (column.x1, draft.y0, column.x2, draft.y1)
        cells.append(Cell(text, bbox, tokens, row_idx, column.idx))
    return cells


def _row(draft: RowDraft, columns: list[ColumnSpec], idx: int, kind: str) -> TableRow:
    cells = _row_cells(draft, columns, idx)
    bbox = (min(c.bbox[0] for c in cells), min(c.bbox[1] for c in cells), max(c.bbox[2] for c in cells), max(c.bbox[3] for c in cells))
    return TableRow(idx, cells, bbox, continuation_rows=max(draft.lines - 1, 0), kind_hint=kind)


def discover_tables(page: Page, matcher, config: dict, document_id: str = "document") -> list[Table]:
    """All tables of a page in reading order. ``config`` is the ``geometry`` + ``summary_labels`` YAML content."""
    geometry = config.get("geometry", {})
    labels = summary_pattern(config.get("summary_labels", []))
    header_opts = {k: v for k, v in geometry.items() if k.startswith(("header_", "alias_"))}
    row_opts = {k: v for k, v in geometry.items() if k in ("continuation_pitch_factor", "sparse_share", "prose_width_share")}
    rules = vertical_rules(page.source_page)
    blocks, line_height = text_blocks(page.tokens, page.aspect, geometry, rules)
    if not blocks:
        return []
    ratio = page.aspect or geometry.get("default_aspect", 0.7071)
    em_x = line_height / ratio

    found: list[tuple[float, float, Table]] = []
    for block in blocks:
        lines = block.lines
        start = 0
        while start < len(lines) - 1:
            header = detect_header(lines, start, em_x, line_height, matcher, header_opts)
            if header is None:
                break
            table, next_start = _build_table(lines, header, em_x, line_height, labels, row_opts, geometry, page, document_id, rules)
            if table is not None:
                found.append((table.bbox[1], table.bbox[0], table))
            start = max(next_start, header.last + 1)
    tables = [item[2] for item in sorted(found, key=lambda item: (round(item[0], 3), item[1]))]
    for number, table in enumerate(tables, start=1):
        table.table_id = f"{document_id}-p{page.page_number}-t{number}"
    return tables


def _build_table(lines: list[Line], header: Header, em_x: float, line_height: float, labels, row_opts: dict, geometry: dict, page: Page, document_id: str, rules=None):
    rest = lines[header.last + 1:]
    if not rest:
        return None, header.last + 1
    preview = rest[:DEFAULT_PREVIEW_LINES]
    columns = infer_columns(header, preview, em_x, geometry, rules)
    if len(columns) < 2:
        return None, header.last + 1
    body = segment_body(preview, columns, em_x, line_height, labels, row_opts)
    if not body.rows:
        return None, header.last + 1
    # Second pass: columns from the lines that really belong to the table, then rows again.
    used = preview[: body.consumed]
    columns = infer_columns(header, used, em_x, geometry, rules)
    body = segment_body(used, columns, em_x, line_height, labels, row_opts)
    if not body.rows or len(columns) < 2:
        return None, header.last + 1

    rows = [_row(draft, columns, i, "item") for i, draft in enumerate(body.rows)]
    summary = [_row(draft, columns, i, "total_like") for i, draft in enumerate(body.summary)]
    header_tokens = [t for line in header.lines for seg in line.segs for t in seg.tokens]
    item_tokens = [t for draft in body.rows for tokens in draft.cells.values() for t in tokens]
    bbox = _union(header_tokens + item_tokens)
    table_columns = [Column(c.idx, c.x1, c.x2, c.header_text, list(c.header_lines)) for c in columns]
    table = Table(
        f"{document_id}-p{page.page_number}-t0", page.page_number, page.page_number, "columnar", "token_grid",
        table_columns, rows, bbox, conf=0.0, stitched_pages=[page.page_number], reason="token_geometry_v2",
        summary_rows=summary,
    )
    table.validation["header_score"] = round(header.score, 4)
    return table, header.last + 1 + body.consumed
