"""Clean-up of tables produced by the PyMuPDF provider (digital PDFs), applied before scoring.

``find_tables`` returns every ruled row as a data row and always treats the first row as the header. Two consequences,
both measured on the stress set (pages with exact gold): total/tax rows end up among the item rows, and a table WITHOUT a
header gets its first item row as header. Both are fixed here with structure only (no vendor rules):

* trailing rows whose label is in the YAML summary vocabulary, or that fill only the last two columns, are summary rows;
* a "header" that looks like data (mostly numbers) is turned into the first data row and the columns get no header text.
"""
from __future__ import annotations

import re

from ..schema import Cell, Table, TableRow

DEFAULT_REFINE = {
    "summary_max_fill_share": 0.5,   # a labelled summary row fills at most this share of the columns
    "header_data_share": 0.4,        # share of non-empty header cells that look like data to call the header a data row
    "structural_min_first_col": 0.6, # structural summary rule only if this share of the rows has a first-column cell
}
_NUMBERISH = re.compile(r"^[^\w]*(rp|usd|idr)?\s*[\d][\d.,/:%\-\s]*[a-zA-Z%]{0,3}$", re.IGNORECASE)


def _filled(row: TableRow) -> list[int]:
    return [i for i, cell in enumerate(row.cells) if cell.text.strip()]


def _is_number(text: str) -> bool:
    return bool(_NUMBERISH.match(text.strip())) and any(ch.isdigit() for ch in text)


def _is_summary_row(row: TableRow, n_cols: int, pattern: re.Pattern | None, opts: dict) -> bool:
    filled = _filled(row)
    if not filled or n_cols < 3:
        return False
    first_text = row.cells[filled[0]].text.strip()
    sparse = len(filled) <= max(2, int(opts["summary_max_fill_share"] * n_cols))
    if pattern is not None and pattern.match(first_text) and sparse:
        return True
    # Structural: only the last two columns hold anything (a label next to an amount) and one of them is a number.
    if not opts.get("structural", True):
        return False
    in_tail = all(i >= n_cols - 2 for i in filled)
    return in_tail and len(filled) <= 2 and any(_is_number(row.cells[i].text) for i in filled)


def split_summary_rows(table: Table, pattern: re.Pattern | None, options: dict | None = None) -> int:
    """Move the contiguous trailing summary rows to ``table.summary_rows``; returns how many were moved."""
    opts = {**DEFAULT_REFINE, **(options or {})}
    n_cols = len(table.columns)
    # The structural rule (label next to an amount in the last two columns) needs item rows that start in the first column.
    first_filled = sum(bool(row.cells and row.cells[0].text.strip()) for row in table.rows) / max(len(table.rows), 1)
    opts = {**opts, "structural": first_filled >= opts["structural_min_first_col"]}
    moved: list[TableRow] = []
    while len(table.rows) - len(moved) > 1 and _is_summary_row(table.rows[len(table.rows) - len(moved) - 1], n_cols, pattern, opts):
        moved.append(table.rows[len(table.rows) - len(moved) - 1])
    if not moved:
        return 0
    moved.reverse()
    table.rows = table.rows[: len(table.rows) - len(moved)]
    # The table box ends with its last ITEM row, like the box of the token-geometry provider (summary rows are not item rows).
    bottom = max(cell.bbox[3] for row in table.rows for cell in row.cells)
    table.bbox = (table.bbox[0], table.bbox[1], table.bbox[2], max(bottom, table.bbox[1]))
    for index, row in enumerate(moved):
        row.idx, row.kind_hint = index, "total_like"
        for cell in row.cells:
            cell.row_idx = index
    table.summary_rows = moved + table.summary_rows
    return len(moved)


def headerless_to_data(table: Table, options: dict | None = None) -> bool:
    """If the header row looks like data, make it the first data row and blank the header texts."""
    opts = {**DEFAULT_REFINE, **(options or {})}
    texts = [column.header_text.strip() for column in table.columns]
    filled = [t for t in texts if t]
    if len(filled) < 2:
        return False
    numeric = sum(_is_number(t) for t in filled)
    first_is_index = bool(texts[0]) and bool(re.fullmatch(r"\d{1,3}[.)]?", texts[0]))
    if numeric < 2 or (numeric / len(filled) < opts["header_data_share"] and not first_is_index):
        return False
    top = table.bbox[1]
    below = table.rows[0].bbox[1] if table.rows else top
    cells = [Cell(text, (column.x1, top, column.x2, max(below, top)), row_idx=0, col_idx=i) for i, (column, text) in enumerate(zip(table.columns, texts))]
    for row in table.rows:
        row.idx += 1
        for cell in row.cells:
            cell.row_idx = row.idx
    table.rows.insert(0, TableRow(0, cells, (min(c.bbox[0] for c in cells), top, max(c.bbox[2] for c in cells), max(below, top))))
    for column in table.columns:
        column.header_text, column.header_lines = "", [""]
    return True


def refine_pymupdf_table(table: Table, pattern: re.Pattern | None, options: dict | None = None) -> Table:
    headerless_to_data(table, options)
    split_summary_rows(table, pattern, options)
    return table
