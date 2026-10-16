"""Header detection by scoring candidates (Fase 3.2; targets F1, and removes the fake headers of T-02).

A header line is a line of at least two WORD cells that sit directly above lines whose cells line up under
them and are mostly numeric. Candidates are scored and the best one wins; nothing here depends on the
document type (an alias matcher, when available, only adds a bonus).
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

import re


from .geometry import Line, Seg

DEFAULT_HEADER = {
    "header_search_lines": 400,
    "header_lookahead": 6,
    "header_min_score": 0.55,
    "header_max_extra_lines": 2,
    "header_line_pitch": 1.7,        # max vertical distance between header lines, in line heights
    "header_tolerance_em": 0.5,      # horizontal slack when asking whether a body cell lies under a header cell
    "header_long_text_chars": 40,    # average cell length above which a line reads as prose, not as a header
    "alias_bonus": 0.25,
    "alias_min_score": 0.8,
}


@dataclass
class Header:
    first: int  # index of the first header line (inclusive)
    last: int  # index of the last header line (inclusive)
    primary: int  # the scored line; its cells define the columns
    score: float
    lines: list[Line]


_LABEL_VALUE = re.compile(r"[A-Za-z)]:\s+\S")


def has_digit(text: str) -> bool:
    """Header words carry no digits; row numbers, quantities with units, dates and amounts all do."""
    return any(ch.isdigit() for ch in text)


def _lies_under(seg: Seg, head: Seg, tolerance: float) -> bool:
    low, high = head.x0 - tolerance, head.x1 + tolerance
    overlap = max(0.0, min(seg.x1, head.x1) - max(seg.x0, head.x0))
    return low <= seg.cx <= high or overlap >= 0.3 * min(seg.x1 - seg.x0, head.x1 - head.x0)


def candidate_score(lines: list[Line], index: int, em_x: float, matcher, opts: dict) -> float:
    segs = lines[index].segs
    n = len(segs)
    if n < 2:
        return 0.0
    numeric = sum(has_digit(s.text) for s in segs) / n
    if numeric > 0.34 or _looks_like_key_value(segs) or _prose_like(segs):
        return 0.0
    body = lines[index + 1:index + 1 + opts["header_lookahead"]]
    if not body:
        return 0.0
    tolerance = opts["header_tolerance_em"] * em_x
    aligned, covered, body_numeric = [], [], []
    for line in body:
        hit_heads = set()
        hits = 0
        for seg in line.segs:
            matches = [h for h, head in enumerate(segs) if _lies_under(seg, head, tolerance)]
            hits += bool(matches)
            hit_heads.update(matches)
        aligned.append(hits / len(line.segs))
        covered.append(len(hit_heads) / n)
        body_numeric.append(sum(has_digit(s.text) for s in line.segs) / len(line.segs))
    gain = min(max(mean(body_numeric) - numeric, 0.0), 1.0)
    score = 0.30 * mean(aligned) + 0.25 * mean(covered) + 0.20 * gain + 0.15 * (1 - numeric) + 0.10 * min(n / 4, 1.0)
    if mean(len(s.text) for s in segs) > opts["header_long_text_chars"]:
        score *= 0.5
    if matcher is not None:
        fields = set()
        for seg in segs:
            match = matcher.match(seg.text)
            if match.candidates and matcher.best_score(seg.text) >= opts["alias_min_score"]:
                fields.add(match.candidates[0].field)
        score += opts["alias_bonus"] * min(len(fields) / 3, 1.0)
    return score


def _prose_like(segs: list[Seg]) -> bool:
    """Running text, not a header: many cells that start with a lowercase letter (words of a sentence)."""
    lower = sum(1 for s in segs if next((c for c in s.text if c.isalpha()), "A").islower())
    return lower >= max(2, 0.3 * len(segs))


def _looks_like_key_value(segs: list[Seg]) -> bool:
    """``Label : value`` pairs are not headers: a cell that starts with a colon, or a short line of labels ending in one."""
    if any(s.text.lstrip().startswith(":") or _LABEL_VALUE.search(s.text) for s in segs):
        return True
    return len(segs) <= 3 and any(s.text.rstrip().endswith(":") for s in segs)


def _short_words(line: Line, limit: int = 5, chars: int = 40) -> bool:
    return all(len(s.text.split()) <= limit and len(s.text) <= chars for s in line.segs)


def detect_header(lines: list[Line], start: int, em_x: float, line_height: float, matcher, options: dict | None = None) -> Header | None:
    """Best header among the first ``header_search_lines`` lines from ``start``; None when no candidate is convincing."""
    opts = {**DEFAULT_HEADER, **(options or {})}
    scores = {}
    for index in range(start, min(len(lines) - 1, start + opts["header_search_lines"])):
        scores[index] = candidate_score(lines, index, em_x, matcher, opts)
    accepted = [i for i, score in scores.items() if score >= opts["header_min_score"]]
    if not accepted:
        return None
    # The FIRST convincing line opens the table; a stronger line further down belongs to the next table. Only the
    # tightly spaced lines that follow it (a wrapped header) may take over as the line that defines the columns.
    gaps = [b.cy - a.cy for a, b in zip(lines[start:start + 12], lines[start + 1:start + 12])]
    pitch_limit = opts["header_line_pitch"] * max(line_height, median(gaps) if gaps else 0.0)
    first = last = accepted[0]
    tolerance = opts["header_tolerance_em"] * em_x
    extra = 0
    # Lines below the first header line that are still header (wrapped text, units, bilingual labels): no numbers in
    # them, tightly spaced, and numbers in the line after them (so the first body row is not swallowed).
    while extra < opts["header_max_extra_lines"] and last + 2 < len(lines):
        below, after = lines[last + 1], lines[last + 2]
        if (
            below.cy - lines[last].cy <= pitch_limit
            and not any(has_digit(s.text) for s in below.segs)
            and not _looks_like_key_value(below.segs)
            and _short_words(below)
            and any(_lies_under(s, h, tolerance) for s in below.segs for h in lines[first].segs)
            and any(has_digit(s.text) for s in after.segs)
        ):
            last += 1
            extra += 1
        else:
            break
    # Lines directly above that sit over the same columns (group labels, first half of a wrapped header).
    while extra < opts["header_max_extra_lines"] and first - 1 >= start:
        above = lines[first - 1]
        if (
            lines[first].cy - above.cy <= pitch_limit
            and not any(has_digit(s.text) for s in above.segs)
            and not _looks_like_key_value(above.segs)
            and _short_words(above)
            and any(_lies_under(s, h, tolerance) for s in above.segs for h in lines[first].segs)
        ):
            first -= 1
            extra += 1
        else:
            break
    block = list(range(first, last + 1))
    primary = max(block, key=lambda i: (len(lines[i].segs), i))  # the widest line defines the columns
    return Header(first, last, primary, scores[accepted[0]], lines[first:last + 1])
