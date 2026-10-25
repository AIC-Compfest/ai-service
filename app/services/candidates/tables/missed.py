"""Missed-table check (step 3 of the Docling flow).

Docling is trusted when its tables explain the numeric content of the page. A page is "probably missing a table"
when a vertical run of numeric lines is not covered by any accepted table, or when every table found is degenerate.
Structure and geometry only; every threshold comes from the ``docling.missed`` section of table_resolution.yaml.
"""
from __future__ import annotations

from ..schema import Page, Table
from .header import has_digit

DEFAULTS = {
    "min_numeric_tokens": 2,
    "min_line_tokens": 3,
    "cover_margin": 0.01,
    "min_uncovered_lines": 3,
    "max_run_gap_lines": 1,
}


def _covered(token, tables: list[Table], margin: float) -> bool:
    return any(
        t.bbox[0] - margin <= token.cx <= t.bbox[2] + margin and t.bbox[1] - margin <= token.cy <= t.bbox[3] + margin
        for t in tables
    )


def suspect_missed_tables(page: Page, tables: list[Table], options: dict | None = None) -> tuple[bool, str]:
    """Return ``(suspect, reason)``; ``reason`` is "" when the page passes."""
    opts = {**DEFAULTS, **(options or {})}
    # Summary rows are separated from the item grid, so their boxes must count as covered too.
    covering = list(tables) + [
        Table(t.table_id, t.page_start, t.page_end, t.layout_kind, t.source, t.columns, t.summary_rows, _rows_box(t), reason=t.reason)
        for t in tables
        if t.summary_rows
    ]
    run: list[int] = []
    last_numeric_line = None
    best = 0
    for line_number, line in enumerate(page.lines):
        numeric = sum(has_digit(tok.text) for tok in line)
        is_numeric_line = numeric >= opts["min_numeric_tokens"] and len(line) >= opts["min_line_tokens"]
        uncovered = is_numeric_line and not all(_covered(tok, covering, opts["cover_margin"]) for tok in line if has_digit(tok.text)) 
        if uncovered:
            if last_numeric_line is not None and line_number - last_numeric_line - 1 > opts["max_run_gap_lines"]:
                run = []
            run.append(line_number)
            last_numeric_line = line_number
            best = max(best, len(run))
    if best >= opts["min_uncovered_lines"]:
        return True, f"uncovered_numeric_block:{best}_lines"
    if tables and all(len(t.columns) < 2 or len(t.rows) < 1 for t in tables):
        return True, "degenerate_tables_only"
    return False, ""


def _rows_box(table: Table) -> tuple[float, float, float, float]:
    boxes = [row.bbox for row in table.summary_rows]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
