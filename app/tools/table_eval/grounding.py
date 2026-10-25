"""Structural gold bootstrap ("synthetic_grounded") from digital PDFs and ground-truth VALUES.

For every GT table (a list of rows of ``(key, value)`` pairs) the builder locates each row on the PDF page
using the GT values themselves (amounts, quantities, description tokens), then derives row lines, column
intervals, the header block and summary rows from the page words. The output follows the GOLD_REVIEW
schema with ``source_label="synthetic_grounded"`` and ``reviewed=false``.

Independence rule: this module MUST NOT import the production table providers or router. It reuses only
the evaluation-side number parser. Its heuristics (monotone row anchoring, key-guided column intervals)
are driven by GT values, which the providers never see.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import pymupdf

from .normalize import canonical_decimal, parse_number

NUM_RE = re.compile(r"\d[\d.,]*\d|\d")
TOKEN_RE = re.compile(r"[a-z0-9]+")
ROW_MIN_COVERAGE = 0.7
TABLE_MIN_ROWS_OK = 0.8
HEADER_MAX_LINES = 3
KEY_MIN_HIT_RATE = 0.3
RULED_HEADER_MAX_LINES = 6


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    nums: set[str] = field(default_factory=set)
    toks: set[str] = field(default_factory=set)
    line: int = 0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Needle:
    key: str
    kind: str  # "num" | "text"
    value: str
    toks: list[str]


def _word_numbers(text: str) -> set[str]:
    found = set()
    for match in NUM_RE.findall(text):
        parsed = parse_number(match).value
        if parsed is not None:
            found.add(canonical_decimal(parsed))
    return found


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def page_lines(page: pymupdf.Page) -> list[list[Word]]:
    width, height = page.rect.width, page.rect.height
    words = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        words.append(Word(x0 / width, y0 / height, x1 / width, y1 / height, text, _word_numbers(text), set(_tokens(text))))
    if not words:
        return []
    median_h = median(w.y1 - w.y0 for w in words)
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.cy, w.x0)):
        line = next((l for l in lines if abs(median(w.cy for w in l) - word.cy) <= median_h * 0.5), None)
        if line is None:
            lines.append([word])
        else:
            line.append(word)
    lines.sort(key=lambda l: median(w.cy for w in l))
    for index, line in enumerate(lines):
        line.sort(key=lambda w: w.x0)
        for word in line:
            word.line = index
    return lines


Rules = tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]


def page_rules(page: pymupdf.Page) -> Rules:
    """Vertical ``(x, y0, y1)`` and horizontal ``(y, x0, x1)`` drawn segments, normalized to the page."""
    width, height = page.rect.width, page.rect.height
    vertical: list[tuple[float, float, float]] = []
    horizontal: list[tuple[float, float, float]] = []

    def add(x0, y0, x1, y1):
        if abs(x1 - x0) <= 1.0 and abs(y1 - y0) > 3.0:
            vertical.append((((x0 + x1) / 2) / width, min(y0, y1) / height, max(y0, y1) / height))
        elif abs(y1 - y0) <= 1.0 and abs(x1 - x0) > 3.0:
            horizontal.append((((y0 + y1) / 2) / height, min(x0, x1) / width, max(x0, x1) / width))

    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                add(item[1].x, item[1].y, item[2].x, item[2].y)
            elif item[0] == "re":
                r = item[1]
                add(r.x0, r.y0, r.x0, r.y1)
                add(r.x1, r.y0, r.x1, r.y1)
                add(r.x0, r.y0, r.x1, r.y0)
                add(r.x0, r.y1, r.x1, r.y1)
    return vertical, horizontal


def _cluster(values: list[float], tolerance: float = 0.004) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(c) / len(c) for c in clusters]


def ruled_bounds(rules: Rules, key_x0: float, key_x1: float, row_y0: float, row_y1: float, centers: list[float]) -> tuple[list[float], float] | None:
    """Column boundaries from drawn vertical rules that enclose the key-matched words, plus the rules' top y."""
    span = max(row_y1 - row_y0, 1e-6)
    used = [(x, y0, y1) for x, y0, y1 in rules[0] if min(y1, row_y1) - max(y0, row_y0) >= 0.6 * span]
    xs = _cluster([x for x, _, _ in used])
    left = [x for x in xs if x <= key_x0 + 0.003]
    right = [x for x in xs if x >= key_x1 - 0.003]
    if not left or not right:
        return None
    lo, hi = max(left), min(right)
    # Extend over neighbouring ruled columns that hold row words but no GT key (e.g. a container column).
    while True:
        earlier = [x for x in xs if x < lo - 0.004]
        if earlier and any(max(earlier) < c < lo for c in centers):
            lo = max(earlier)
            continue
        break
    while True:
        later = [x for x in xs if x > hi + 0.004]
        if later and any(hi < c < min(later) for c in centers):
            hi = min(later)
            continue
        break
    inside = [x for x in xs if lo - 0.002 <= x <= hi + 0.002]
    if len(inside) < 3 or (hi - lo) > 1.8 * (key_x1 - key_x0) + 0.1:
        return None
    top = min(y0 for x, y0, _ in used if lo - 0.004 <= x <= hi + 0.004)
    return inside, top


def ruled_row_end(lines: list[list[Word]], anchor: int, rules: Rules, xs: list[float]) -> int:
    """Last line index of a row whose bottom is the next horizontal rule spanning the table width."""
    width = xs[-1] - xs[0]
    floor = max(w.y1 for w in lines[anchor])
    below = [y for y, x0, x1 in rules[1] if y > floor and min(x1, xs[-1]) - max(x0, xs[0]) >= 0.6 * width]
    if not below:
        return anchor
    limit = min(below)
    end = anchor
    for j in range(anchor + 1, len(lines)):
        if lines[j][0].cy < limit:
            end = j
        else:
            break
    return end


def make_needles(row: list[tuple[str, Any]]) -> list[Needle]:
    needles = []
    for key, value in row:
        if value is None or isinstance(value, (dict, list, bool)):
            continue
        text = str(value).strip()
        if not text or text == "-":
            continue
        parsed = parse_number(text)
        if parsed.value is not None:
            needles.append(Needle(key, "num", canonical_decimal(parsed.value), []))
        else:
            toks = _tokens(text)
            if toks:
                needles.append(Needle(key, "text", text, toks))
    return needles


def _line_score(line: list[Word], needles: list[Needle]) -> float:
    nums = set().union(*(w.nums for w in line)) if line else set()
    toks = set().union(*(w.toks for w in line)) if line else set()
    score = 0.0
    for needle in needles:
        if needle.kind == "num":
            big = len(needle.value.lstrip("-").replace(".", "")) >= 4
            if needle.value in nums:
                score += 2.0 if big else 0.5
        else:
            fraction = sum(token in toks for token in needle.toks) / len(needle.toks)
            if fraction >= 0.5:
                score += 1.5 * fraction
    return score


def anchor_rows(lines: list[list[Word]], rows: list[list[Needle]], first_row: int) -> list[int]:
    """Anchor consecutive rows (monotone) starting at ``first_row``; returns anchor line indices."""
    anchors: list[int] = []
    previous = -1
    for needles in rows[first_row:]:
        best, best_score = -1, 0.0
        minimum = 1.5 if len(needles) >= 2 else 0.5
        for index in range(previous + 1, len(lines)):
            score = _line_score(lines[index], needles)
            if score > best_score + 1e-9:
                best, best_score = index, score
            if previous >= 0 and index > previous + 12 and best >= 0:
                break
        if best < 0 or best_score < minimum:
            break
        anchors.append(best)
        previous = best
    return anchors


def _assign_row_words(row_words: list[Word], needles: list[Needle]) -> dict[str, list[Word]]:
    """Assign words to GT keys. Keys are processed in GT order with a left-to-right cursor preference."""
    assigned: dict[str, list[Word]] = {}
    used: set[int] = set()
    cursor = -1.0
    ordered = sorted(row_words, key=lambda w: (w.line, w.x0))
    for needle in needles:
        pick: list[Word] = []
        if needle.kind == "num":
            candidates = [w for w in ordered if id(w) not in used and needle.value in w.nums]
            ahead = [w for w in candidates if w.x0 >= cursor - 0.005]
            chosen = (ahead or candidates)[:1]
            pick = chosen
        else:
            remaining = set(needle.toks)
            candidates = [w for w in ordered if id(w) not in used and w.toks and w.toks <= set(needle.toks)]
            ahead = [w for w in candidates if w.x0 >= cursor - 0.005]
            for word in (ahead or candidates):
                if word.toks & remaining:
                    pick.append(word)
                    remaining -= word.toks
                if not remaining:
                    break
        if pick:
            for word in pick:
                used.add(id(word))
            assigned[needle.key] = pick
            cursor = max(w.x1 for w in pick)
    return assigned


def _union(words: list[Word]) -> tuple[float, float, float, float]:
    return (min(w.x0 for w in words), min(w.y0 for w in words), max(w.x1 for w in words), max(w.y1 for w in words))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] - 0.5 * min(hi - lo, merged[-1][1] - merged[-1][0]):
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def _norm_box(box) -> list[float]:
    return [round(min(max(v, 0.0), 1.0), 6) for v in box]


def _cell(index: int, words: list[Word], fallback_box) -> dict:
    ordered = sorted(words, key=lambda w: (w.line, w.x0))
    text = " ".join(w.text for w in ordered)
    return {"column_index": index, "raw_text": text, "normalized_text": text, "bbox_norm": _norm_box(_union(words) if words else fallback_box), "legibility": "clear"}


def _header_like(words: list[Word], ruled: bool) -> bool:
    """Gold v2: a header line carries no contact data, label:value pairs or running text (those were swallowed by v1).

    Inside a ruled grid the rules already delimit the header, so only contact data and label:value are refused there.
    """
    text = " ".join(w.text for w in words)
    if "@" in text or ":" in text:
        return False
    if ruled:
        return True
    if any(ch.isdigit() for ch in text):
        return False
    lower = sum(1 for w in words if next((c for c in w.text if c.isalpha()), "A").islower())
    return lower < max(2, 0.3 * len(words))


def ground_table(lines: list[list[Word]], rows: list[list[Needle]], anchors: list[int], first_row: int, totals: set[str], table_id: str, rules: Rules | None = None) -> tuple[dict | None, str]:
    """Build one gold table for rows ``first_row .. first_row+len(anchors)`` on a page, or return a reason."""
    if not anchors:
        return None, "no_anchor"
    n = len(anchors)
    gaps = [lines[anchors[i + 1]][0].cy - lines[anchors[i]][0].cy for i in range(n - 1)]
    base_gap = min(gaps) if gaps else 0.0
    # Row line ranges: in-block non-anchor lines are continuation of the preceding anchor row.
    ranges = []
    for i in range(n):
        end = anchors[i + 1] - 1 if i + 1 < n else anchors[i]
        ranges.append([anchors[i], end])
    # Continuation after the last row: lines carrying not-yet-seen description tokens of that row.
    last_needles = rows[first_row + n - 1]
    seen = set().union(*(w.toks for w in lines[anchors[-1]])) if lines[anchors[-1]] else set()
    wanted = {t for needle in last_needles if needle.kind == "text" for t in needle.toks} - seen
    j = anchors[-1] + 1
    while j < len(lines) and wanted and (not base_gap or lines[j][0].cy - lines[j - 1][0].cy <= 2.5 * base_gap):
        hit = {t for w in lines[j] for t in w.toks} & wanted
        if not hit:
            break
        ranges[-1][1] = j
        wanted -= hit
        j += 1

    per_row_words: list[list[Word]] = []
    all_assigned: list[dict[str, list[Word]]] = []
    for i, (start, end) in enumerate(ranges):
        row_words = [w for line in lines[start:end + 1] for w in line]
        per_row_words.append(row_words)
        all_assigned.append(_assign_row_words(row_words, rows[first_row + i]))

    # Keys that the document never shows (e.g. a GT row number printed nowhere) do not count against coverage.
    key_hits: dict[str, int] = {}
    for assigned in all_assigned:
        for key in assigned:
            key_hits[key] = key_hits.get(key, 0) + 1
    displayed = {key for key, hits in key_hits.items() if hits / n >= KEY_MIN_HIT_RATE}
    coverage_ok = 0
    for i, assigned in enumerate(all_assigned):
        expected = [needle for needle in rows[first_row + i] if needle.key in displayed]
        if expected and sum(needle.key in assigned for needle in expected) / len(expected) >= ROW_MIN_COVERAGE:
            coverage_ok += 1
    if coverage_ok / n < TABLE_MIN_ROWS_OK:
        return None, f"weak_coverage {coverage_ok}/{n}"

    # Column intervals from key-assigned words (merge overlapping intervals of different keys).
    per_key: dict[str, list[Word]] = {}
    for assigned in all_assigned:
        for key, words in assigned.items():
            per_key.setdefault(key, []).extend(words)
    columns = _merge_intervals([(min(w.x0 for w in ws), max(w.x1 for w in ws)) for ws in per_key.values()])
    if len(columns) < 2:
        return None, "single_column"
    table_x0, table_x1 = columns[0][0] - 0.01, columns[-1][1] + 0.01
    bounds = [table_x0] + [(columns[i][1] + columns[i + 1][0]) / 2 for i in range(len(columns) - 1)] + [table_x1]
    rule_top = None
    if rules is not None:
        flat = [w for words in per_row_words for w in words]
        ruled = ruled_bounds(rules, columns[0][0], columns[-1][1], min(w.y0 for w in flat), max(w.y1 for w in flat), [w.cx for w in flat])
        if ruled is not None:
            bounds, rule_top = ruled
            columns = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
            table_x0, table_x1 = bounds[0], bounds[-1]
            end = ruled_row_end(lines, anchors[-1], rules, bounds)
            if end > ranges[-1][1]:
                ranges[-1][1] = end
                per_row_words[-1] = [w for line in lines[ranges[-1][0]:end + 1] for w in line]

    def column_of(word: Word) -> int | None:
        if word.cx < bounds[0] or word.cx > bounds[-1]:
            return None
        return next(i for i in range(len(columns)) if word.cx <= bounds[i + 1])

    row_dicts, item_words = [], []
    for i, row_words in enumerate(per_row_words):
        buckets: list[list[Word]] = [[] for _ in columns]
        for word in row_words:
            c = column_of(word)
            if c is not None:
                buckets[c].append(word)
        y0 = min(w.y0 for w in row_words)
        y1 = max(w.y1 for w in row_words)
        cells = [_cell(c, ws, (bounds[c], y0, bounds[c + 1], y1)) for c, ws in enumerate(buckets)]
        item_words.extend(w for ws in buckets for w in ws)
        row_dicts.append({"row_id": f"r{i + 1}", "kind": "item", "cells": cells})
    if not item_words:
        return None, "no_words"

    # Header block: up to HEADER_MAX_LINES lines above the first row that carry no row numbers.
    first_line = ranges[0][0]
    header_lines: list[list[Word]] = []
    row_numbers = set().union(*(w.nums for w in item_words)) if item_words else set()
    k = first_line - 1
    while k >= 0 and len(header_lines) < (RULED_HEADER_MAX_LINES if rule_top is not None else HEADER_MAX_LINES):
        in_table = [w for w in lines[k] if table_x0 <= w.cx <= table_x1]
        if rule_top is not None:
            if lines[k][0].cy < rule_top - 0.002 or not in_table:
                break
        else:
            step = lines[k + 1][0].cy - lines[k][0].cy
            if len(in_table) < 2 or (base_gap and step > 2.2 * base_gap) or any(len(num) >= 4 and num in row_numbers for w in in_table for num in w.nums):
                break
        if not _header_like(in_table, rule_top is not None):
            break
        header_lines.insert(0, in_table)
        k -= 1
    headers, header_found = [], False
    if header_lines:
        buckets = [[] for _ in columns]
        for line in header_lines:
            for word in line:
                c = column_of(word)
                if c is not None:
                    buckets[c].append(word)
        if sum(bool(ws) for ws in buckets) / len(columns) >= 0.5:
            header_found = True
            headers = [_cell(c, ws, (bounds[c], 0, bounds[c + 1], 0)) for c, ws in enumerate(buckets)]
            item_words.extend(w for ws in buckets for w in ws)

    # Summary rows: lines after the item block that carry a GT totals value.
    summary_rows, misses, j = [], 0, ranges[-1][1] + 1
    while j < len(lines) and misses < 2 and len(summary_rows) < 8:
        in_table = [w for w in lines[j] if table_x0 - 0.2 <= w.cx <= table_x1 + 0.2]
        if in_table and any(num in totals for w in in_table for num in w.nums):
            heights = median(w.y1 - w.y0 for w in in_table)
            cells, current = [], [in_table[0]]
            for word in in_table[1:]:
                if word.x0 - current[-1].x1 > heights * 1.0:
                    cells.append(current)
                    current = [word]
                else:
                    current.append(word)
            cells.append(current)
            summary_rows.append({"row_id": f"s{len(summary_rows) + 1}", "kind": "summary", "cells": [_cell(ci, ws, (0, 0, 0, 0)) for ci, ws in enumerate(cells)]})
            misses = 0
        else:
            misses += 1
        j += 1

    table = {
        "table_id": table_id,
        "role": "line_items",
        "bbox_norm": _norm_box(_union(item_words)),
        "headers": headers,
        "rows": row_dicts,
        "summary_rows": summary_rows,
        "logical_id": None,
        "quality": {"header_found": header_found, "row_coverage_ok": f"{coverage_ok}/{n}", "columns": len(columns)},
    }
    return table, "ok"


def ground_document(pdf_path, gt_tables: list[dict], totals: list[str], document_id: str) -> tuple[dict, list[dict]]:
    """Return ``(gold_pages_by_number, report)`` for a PDF.

    ``gt_tables``: ``[{"path": str, "rows": [[(key, value), ...], ...]}, ...]``.
    """
    document = pymupdf.open(pdf_path)
    page_data = [(index + 1, page_lines(page), page_rules(page)) for index, page in enumerate(document)]
    totals_set = set()
    for value in totals:
        parsed = parse_number(value)
        if parsed.value is not None:
            totals_set.add(canonical_decimal(parsed.value))
    pages: dict[int, list[dict]] = {}
    report: list[dict] = []
    for t_index, gt in enumerate(gt_tables, start=1):
        needle_rows = [make_needles(row) for row in gt["rows"]]
        if not needle_rows or all(not n for n in needle_rows):
            report.append({"path": gt["path"], "status": "skipped", "reason": "empty_gt_table"})
            continue
        needle_rows = [n for n in needle_rows if n]
        built, next_row, reason = [], 0, "ok"
        for page_number, lines, page_rule_set in page_data:
            if next_row >= len(needle_rows):
                break
            anchors = anchor_rows(lines, needle_rows, next_row)
            if not anchors:
                continue
            table, why = ground_table(lines, needle_rows, anchors, next_row, totals_set, f"{document_id}-p{page_number}-g{t_index}", page_rule_set)
            if table is None:
                reason = why
                break
            built.append((page_number, table))
            next_row += len(anchors)
        if next_row < len(needle_rows) or not built:
            report.append({"path": gt["path"], "status": "failed", "reason": reason if reason != "ok" else f"rows_unanchored {next_row}/{len(needle_rows)}"})
            continue
        logical = f"{document_id}#g{t_index}" if len(built) > 1 else None
        for page_number, table in built:
            table["logical_id"] = logical
            pages.setdefault(page_number, []).append(table)
        report.append({"path": gt["path"], "status": "ok", "pages": [p for p, _ in built], "rows": len(needle_rows)})
    return pages, report
