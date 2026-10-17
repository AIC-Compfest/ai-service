"""Row segmentation, summary rows and the end of a table (Fase 3.4; targets F3, F9, F10).

Order of decisions for every body line (this order fixes T-06, where a Total row was swallowed as a
continuation before it was ever tested for being a summary):
  1. prose that is not part of the grid ends the table;
  2. a summary label (YAML vocabulary, whole word) in a sparse line starts the summary block;
  3. after summary lines, the first non-summary line ends the table;
  4. a line starts a row when its row-number cell (anchor column) holds a number or most active columns are filled;
  5. otherwise it continues the current row only when it is tight below it and does not touch the anchor column.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import median

from app.tools.table_eval.normalize import is_numeric

from ..schema import Token
from .columns import ColumnSpec, split_segment
from .geometry import Line

DEFAULT_ROWS = {
    "continuation_pitch_factor": 1.25,  # a continuation line is at most this x the tightest line spacing below the row
    "anchor_min_fill": 0.6,
    "anchor_numeric_share": 0.7,
    "active_column_share": 0.3,
    "sparse_share": 0.5,                # a summary line fills at most this share of the columns
    "prose_width_share": 0.6,           # a single cell wider than this share of the table width is prose, not a cell
    "anchor_scan_lines": 30,
    "lead_line_rows": True,             # a description line above the numbers' line opens a row (two-line items)
    "lead_gap_ratio": 0.7,              # ... when its gap to the line below is at most this x its gap to the line above
}
ROW_NUMBER = re.compile(r"^\d{1,3}[.)]?$")


@dataclass
class RowDraft:
    cells: dict[int, list[Token]] = field(default_factory=dict)
    y0: float = 1.0
    y1: float = 0.0
    lines: int = 0

    def add(self, line: Line, pieces: dict[int, list[Token]]) -> None:
        for col, tokens in pieces.items():
            self.cells.setdefault(col, []).extend(tokens)
        self.y0, self.y1 = min(self.y0, line.y0), max(self.y1, line.y1)
        self.lines += 1


@dataclass
class Body:
    rows: list[RowDraft]
    summary: list[RowDraft]
    consumed: int  # number of body lines used (rows + summary rows + continuations)


def summary_pattern(labels: list[str]) -> re.Pattern:
    ordered = sorted((re.escape(label.strip()) for label in labels if label.strip()), key=len, reverse=True)
    return re.compile(r"^(?:" + "|".join(ordered) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def _pieces(line: Line, columns: list[ColumnSpec]) -> dict[int, list[Token]]:
    pieces: dict[int, list[Token]] = {}
    for seg in line.segs:
        for col, tokens in split_segment(seg, columns):
            pieces.setdefault(col, []).extend(tokens)
    return pieces


def _text(tokens: list[Token]) -> str:
    return " ".join(t.text for t in tokens)


def find_anchor(cells: list[dict[int, list[Token]]], n_cols: int, opts: dict) -> int | None:
    """The row-number column: one of the first two columns that is mostly filled with short integers."""
    sample = cells[: opts["anchor_scan_lines"]]
    if not sample:
        return None
    for col in range(min(2, n_cols)):
        texts = [_text(c[col]) for c in sample if col in c]
        if len(texts) / len(sample) >= opts["anchor_min_fill"] and sum(bool(ROW_NUMBER.match(t)) for t in texts) / len(texts) >= opts["anchor_numeric_share"]:
            return col
    return None


def _leads_next_row(body: list[Line], cells: list[dict[int, list[Token]]], index: int, anchor: int | None, start_fill: int, tight: float | None, opts: dict) -> bool:
    """True when line ``index`` is a sparse description line directly above (tight gap) a line that fills the row."""
    if index + 1 >= len(body):
        return False
    here, below = cells[index], cells[index + 1]
    if anchor is not None and anchor in here:
        return False
    if len(here) >= start_fill or len(below) < start_fill:
        return False
    gap_below = body[index + 1].cy - body[index].cy
    if index > 0:
        # It belongs with what follows only if it sits clearly closer to the line below than to the line above;
        # otherwise it is the wrapped tail of the previous row.
        return gap_below <= opts["lead_gap_ratio"] * (body[index].cy - body[index - 1].cy)
    return tight is None or gap_below <= opts["continuation_pitch_factor"] * tight


def segment_body(body: list[Line], columns: list[ColumnSpec], em_x: float, line_height: float, labels: re.Pattern, options: dict | None = None) -> Body:
    opts = {**DEFAULT_ROWS, **(options or {})}
    n_cols = len(columns)
    cells = [_pieces(line, columns) for line in body]
    anchor = find_anchor(cells, n_cols, opts)
    active = sum(1 for c in range(n_cols) if sum(c in cell for cell in cells) / max(len(cells), 1) >= opts["active_column_share"]) or 1
    start_fill = max(2, math.ceil(0.5 * active))
    gaps = [b.cy - a.cy for a, b in zip(body, body[1:]) if b.cy - a.cy >= 0.8 * line_height]
    tight = min(gaps) if gaps else None
    width = columns[-1].ext1 - columns[0].ext0

    rows: list[RowDraft] = []
    summary: list[RowDraft] = []
    current: RowDraft | None = None
    lead_open = False
    consumed = 0
    for index, (line, line_cells) in enumerate(zip(body, cells)):
        filled = set(line_cells)
        first_text = _text(line_cells[min(filled)]) if filled else ""
        sparse = len(filled) <= max(1, opts["sparse_share"] * n_cols)
        label_hit = bool(labels.match(first_text)) if first_text else False
        anchor_filled = anchor is not None and anchor in filled and bool(ROW_NUMBER.match(_text(line_cells[anchor])))

        if not filled:
            break
        wide = any((seg.x1 - seg.x0) > opts["prose_width_share"] * width for seg in line.segs) and len(line.segs) == 1
        if wide and not label_hit and len(filled) <= 1:
            break
        label_summary = label_hit and not anchor_filled and len(filled) < n_cols
        has_amount = any(is_numeric(seg.text) for seg in line.segs)
        if label_summary or (summary and sparse and not anchor_filled and has_amount):
            draft = RowDraft()
            draft.add(line, line_cells)
            summary.append(draft)
            current = None
            consumed = index + 1
            continue
        if summary:
            break
        gap = line.cy - body[index - 1].cy if index else None
        starts_row = anchor_filled or len(filled) >= start_fill and (anchor is None or anchor not in filled or anchor_filled)
        if starts_row and lead_open and current is not None:
            current.add(line, line_cells)  # the numbers of a row whose description line came first
            lead_open = False
            consumed = index + 1
            continue
        if starts_row:
            current = RowDraft()
            current.add(line, line_cells)
            rows.append(current)
            consumed = index + 1
            continue
        leads = opts["lead_line_rows"] and _leads_next_row(body, cells, index, anchor, start_fill, tight, opts)
        tight_enough = gap is None or tight is None or gap <= opts["continuation_pitch_factor"] * tight
        if current is not None and tight_enough and not leads and (anchor is None or anchor not in filled):
            current.add(line, line_cells)
            consumed = index + 1
            continue
        if leads:
            # A description line that precedes the line holding the numbers (two-line items, numbers centred between
            # the description lines): it opens the next row instead of ending the table.
            current = RowDraft()
            current.add(line, line_cells)
            rows.append(current)
            lead_open = True
            consumed = index + 1
            continue
        break
    return Body(rows, summary, consumed)


def median_or(values: list[float], default: float) -> float:
    return median(values) if values else default
