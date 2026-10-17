"""Geometry-first table discovery: blocks -> header -> columns -> rows -> Table objects.

Replaces the page-global row clustering of the first token-grid implementation. See geometry.py, header.py,
columns.py and rows.py for the individual decisions. Works for exact vector tokens and for raster OCR tokens.
"""
from __future__ import annotations

from ..schema import Cell, Column, Page, Table, TableRow, Token
from .columns import ColumnSpec, infer_columns
from .geometry import Line, median_height, text_blocks, vertical_rules
from app.tools.table_eval.normalize import is_numeric

from .header import Header, _looks_like_key_value, detect_header, has_digit
from .rows import Body, RowDraft, segment_body, summary_pattern

DEFAULT_PREVIEW_LINES = 60
DEFAULT_HEADERLESS = {
    "enabled": True,
    "min_rows": 2,             # a header-less table needs at least this many rows ...
    "min_cols": 3,             # ... and this many columns
    "numeric_column_share": 0.8, # a column with this share of purely numeric cells (and at least 2) must exist
    "max_cell_words": 6,       # a cell with more words than this is a sentence (a note), so the line is not part of a grid
    "min_numeric_share": 0.6,  # share of rows that hold a number (an amount, a quantity, a date)
}

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

    headerless_opts = {**DEFAULT_HEADERLESS, **(config.get("headerless") or {})}
    found: list[tuple[float, float, Table]] = []
    for block in blocks:
        lines = block.lines
        start = 0
        while start < len(lines) - 1:
            header = detect_header(lines, start, em_x, line_height, matcher, header_opts)
            if headerless_opts["enabled"]:
                # Lines before the next header (or the rest of the block) may hold a table that has no header at all.
                limit = header.first if header is not None else len(lines)
                for table in _headerless_tables(lines[start:limit], em_x, line_height, labels, row_opts, geometry, headerless_opts, page, document_id, rules):
                    found.append((table.bbox[1], table.bbox[0], table))
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
    items = preview[: body.item_end or body.consumed]  # summary lines would stretch the columns (a total spans two of them)
    columns = infer_columns(header, items, em_x, geometry, rules)
    body = segment_body(preview[: body.consumed], columns, em_x, line_height, labels, row_opts)
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


def continue_table(page: Page, previous: Table, config: dict, document_id: str = "document") -> Table | None:
    """Rows of a table that continues on this page WITHOUT repeating its header (Fase 4; targets F11).

    The columns of ``previous`` are reused. The block must start near the top of the page, its first line must
    line up with those columns and carry digits (a body row, not prose); rows are then cut by the usual row rules.
    """
    geometry = config.get("geometry", {})
    labels = summary_pattern(config.get("summary_labels", []))
    row_opts = {k: v for k, v in geometry.items() if k in ("continuation_pitch_factor", "sparse_share", "prose_width_share")}
    rules = vertical_rules(page.source_page)
    blocks, line_height = text_blocks(page.tokens, page.aspect, geometry, rules)
    if not blocks or not previous.columns:
        return None
    ratio = page.aspect or geometry.get("default_aspect", 0.7071)
    em_x = line_height / ratio
    specs = [ColumnSpec(c.idx, c.x1, c.x2, c.header_text, [c.header_text], c.x1, c.x2) for c in previous.columns]
    tolerance = 0.5 * em_x
    for block in blocks:
        first = block.lines[0]
        if first.cy > 0.35:
            break  # continuation rows sit at the top of the page, under at most a running header
        under = sum(any(s.x0 >= c.ext0 - tolerance and s.x1 <= c.ext1 + tolerance or (s.cx >= c.ext0 and s.cx <= c.ext1) for c in specs) for s in first.segs)
        digits = any(ch.isdigit() for s in first.segs for ch in s.text)
        if len(first.segs) < 2 or under / len(first.segs) < 0.7 or not digits:
            continue
        body = segment_body(block.lines, specs, em_x, line_height, labels, row_opts)
        if not body.rows:
            return None
        rows = [_row(draft, specs, i, "item") for i, draft in enumerate(body.rows)]
        summary = [_row(draft, specs, i, "total_like") for i, draft in enumerate(body.summary)]
        tokens = [t for draft in body.rows for toks in draft.cells.values() for t in toks]
        table = Table(
            f"{document_id}-p{page.page_number}-c1", page.page_number, page.page_number, "columnar", previous.source,
            [Column(c.idx, c.x1, c.x2, c.header_text, list(c.header_lines)) for c in specs], rows, _union(tokens), conf=0.0,
            stitched_pages=[page.page_number], reason="continuation", summary_rows=summary,
        )
        table.validation["continued_from"] = previous.table_id
        return table
    return None


def _grid_like(line: Line, max_cell_words: int = 6) -> bool:
    """A line of at least three short cells; a sentence (a note beside a totals block) is not a cell."""
    if _looks_like_key_value(line.segs) or any(seg.text.rstrip().endswith(":") or " : " in seg.text or ": " in seg.text for seg in line.segs):
        return False
    return len(line.segs) >= 3 and all(len(seg.text.split()) <= max_cell_words for seg in line.segs)


def _headerless_tables(lines: list[Line], em_x: float, line_height: float, labels, row_opts: dict, geometry: dict, opts: dict, page: Page, document_id: str, rules) -> list[Table]:
    """Tables without a header row: runs of grid-like lines whose cells line up in columns and hold numbers.

    Columns come from the lines themselves (the widest line stands in for the header and is turned back into a data row), so
    nothing here depends on header words. Strict on purpose: at least ``min_rows`` rows of at least ``min_cols`` cells that
    align, most rows with a number, no label:value lines.
    """
    tables: list[Table] = []
    index = 0
    while index < len(lines):
        if not _grid_like(lines[index], opts["max_cell_words"]):
            index += 1
            continue
        end = index
        while end < len(lines) and (_grid_like(lines[end], opts["max_cell_words"]) or _summary_like(lines[end], labels)):
            end += 1
        run = lines[index:end]
        built = _build_headerless(run, em_x, line_height, labels, row_opts, geometry, opts, page, document_id, rules)
        if built is None:
            index += 1  # try again one line later: the run may start with a line that is not part of the table
            continue
        table, used = built
        tables.append(table)
        index += max(used, 1)
    return tables


def _summary_like(line: Line, labels) -> bool:
    return bool(line.segs) and bool(labels.match(line.segs[0].text.strip())) and len(line.segs) <= 3


def _build_headerless(run: list[Line], em_x: float, line_height: float, labels, row_opts: dict, geometry: dict, opts: dict, page: Page, document_id: str, rules):
    if len(run) < opts["min_rows"]:
        return None
    primary = max(range(len(run)), key=lambda i: (len(run[i].segs), -i))
    if len(run[primary].segs) < opts["min_cols"]:
        return None
    stand_in = Header(0, 0, 0, 0.0, [run[primary]])
    columns = infer_columns(stand_in, run, em_x, geometry, rules)
    if len(columns) < opts["min_cols"]:
        return None
    body = segment_body(run, columns, em_x, line_height, labels, row_opts)
    if len(body.rows) < opts["min_rows"]:
        return None

    numeric_rows = sum(any(has_digit(token.text) for tokens in draft.cells.values() for token in tokens) for draft in body.rows)
    if numeric_rows / len(body.rows) < opts["min_numeric_share"]:
        return None
    # At least one column must be (almost) all numbers: an amount, a quantity. Addresses and dates with a digit in them are not.
    numeric_column = any(
        sum(is_numeric(" ".join(t.text for t in draft.cells.get(col, []))) for draft in body.rows) >= max(2, opts["numeric_column_share"] * len(body.rows))
        for col in range(len(columns))
    )
    if not numeric_column:
        return None
    # Every row must use at least min_cols columns on average; a list of two-cell lines is not a table.
    mean_cells = sum(len(draft.cells) for draft in body.rows) / len(body.rows)
    if mean_cells < opts["min_cols"] - 0.5:
        return None
    rows = [_row(draft, columns, i, "item") for i, draft in enumerate(body.rows)]
    summary = [_row(draft, columns, i, "total_like") for i, draft in enumerate(body.summary)]
    item_tokens = [t for draft in body.rows for tokens in draft.cells.values() for t in tokens]
    if not item_tokens:
        return None
    table_columns = [Column(c.idx, c.x1, c.x2, "", [""]) for c in columns]
    table = Table(
        f"{document_id}-p{page.page_number}-t0", page.page_number, page.page_number, "columnar", "token_grid",
        table_columns, rows, _union(item_tokens), conf=0.0, stitched_pages=[page.page_number], reason="token_geometry_v2:headerless",
        summary_rows=summary,
    )
    return table, body.consumed