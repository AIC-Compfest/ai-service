"""Table quality, role classification and per-table arbitration (Fase 3.5 and 3.6; targets F6, F7).

* ``table_quality``: structural plausibility of ONE table (not an accuracy probability).
* ``classify_role``: line_items / summary / info / other, from structure and the YAML summary vocabulary only.
* ``arbitrate``: tables from several providers are grouped by overlap and the best of each group is kept, so a
  page can mix a ruled table (PyMuPDF) with an unruled one (token geometry) instead of one provider winning the page.
"""
from __future__ import annotations

import re

from ..schema import Table
from .header import has_digit

DEFAULT_SCORING = {
    "max_page_area_ratio": 0.5,
    "header_long_chars": 40,
    "overlap_same_table": 0.5,       # overlap (over the smaller table) at which two candidates are the same table
    "info_label_share": 0.5,         # share of rows with a "Label:" cell that makes a table a key-value block
    "info_max_cols": 3,
    "item_min_digit_share": 0.2,     # a line-item table needs at least this share of body cells with digits
    "summary_label_share": 0.6,
    "text_strategy_prior": 0.9,      # PyMuPDF "text" strategy splits columns by character gaps and is trusted less
}
_LABEL = re.compile(r"[A-Za-z)]:(\s|$)")


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_ratio(a, b) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller else 0.0


def _body_cells(table: Table) -> list[str]:
    return [cell.text for row in table.rows for cell in row.cells]


def table_quality(table: Table, options: dict | None = None) -> float:
    """0..1 plausibility of the table's own structure."""
    opts = {**DEFAULT_SCORING, **(options or {})}
    if not table.columns or not table.rows:
        return 0.0
    n_cols = len(table.columns)
    headers = [c.header_text.strip() for c in table.columns]
    header_ok = sum(bool(h) for h in headers) / n_cols
    if sum(len(h) for h in headers) / n_cols > opts["header_long_chars"]:
        header_ok *= 0.4
    cells = _body_cells(table)
    filled = [c for c in cells if c.strip()]
    grid = len(filled) / len(cells) if cells else 0.0
    empty_columns = sum(1 for c in range(n_cols) if not any(row.cells[c].text.strip() for row in table.rows if c < len(row.cells)))
    grid *= 1 - empty_columns / n_cols
    digits = sum(has_digit(c) for c in filled) / len(filled) if filled else 0.0
    numeric = min(digits / 0.6, 1.0)
    cohesion = 1.0 if len(table.rows) >= 2 else 0.6
    size = 1.0 if _area(table.bbox) <= opts["max_page_area_ratio"] else 0.0
    score = 0.30 * header_ok + 0.25 * grid + 0.20 * numeric + 0.15 * cohesion + 0.10 * size
    if table.reason.endswith(":text"):
        score *= opts["text_strategy_prior"]
    return round(score, 6)


def classify_role(table: Table, summary_pattern: re.Pattern | None, options: dict | None = None) -> tuple[str, str]:
    """Return ``(role, reason)``."""
    opts = {**DEFAULT_SCORING, **(options or {})}
    if not table.rows:
        return "other", "no_rows"
    n_cols = len(table.columns)
    first_cells = [row.cells[0].text.strip() for row in table.rows if row.cells and row.cells[0].text.strip()]
    if summary_pattern is not None and first_cells:
        hits = sum(bool(summary_pattern.match(text)) for text in first_cells)
        if hits / len(first_cells) >= opts["summary_label_share"] and n_cols <= 3:
            return "summary", "summary_labels"
    labelled = sum(any(_LABEL.search(cell.text) or cell.text.lstrip().startswith(":") for cell in row.cells) for row in table.rows)
    if n_cols <= opts["info_max_cols"] and labelled / len(table.rows) >= opts["info_label_share"]:
        return "info", "label_value_rows"
    headers = [c.header_text for c in table.columns]
    if any(_LABEL.search(h) or h.lstrip().startswith(":") for h in headers) and n_cols <= opts["info_max_cols"] + 1:
        return "info", "label_value_header"
    cells = [c for c in _body_cells(table) if c.strip()]
    digit_share = sum(has_digit(c) for c in cells) / len(cells) if cells else 0.0
    if digit_share < opts["item_min_digit_share"]:
        return "other", "no_numeric_body"
    return "line_items", "numeric_grid"


def arbitrate(tables: list[Table], options: dict | None = None) -> list[Table]:
    """Keep the best-scoring table of each group of overlapping candidates (any provider); reading order out."""
    opts = {**DEFAULT_SCORING, **(options or {})}
    ranked = sorted(tables, key=lambda t: t.structural_score, reverse=True)
    kept: list[Table] = []
    for table in ranked:
        if _area(table.bbox) > opts["max_page_area_ratio"] and any(_area(k.bbox) <= opts["max_page_area_ratio"] and overlap_ratio(table.bbox, k.bbox) > 0.5 for k in kept):
            continue
        if any(overlap_ratio(table.bbox, k.bbox) >= opts["overlap_same_table"] for k in kept):
            continue
        kept.append(table)
    kept.sort(key=lambda t: (round(t.bbox[1], 3), t.bbox[0]))
    return kept
