"""Column inference from header cells AND body data (Fase 3.3; targets F2).

Header cells give the anchors; body cells extend them, new columns appear where several rows have cells that
sit under no header, and column boundaries lie in the empty space between the real extents of the data (not at
0.0/1.0 and not at the midpoint of header words).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Token
from .geometry import Line, Seg
from .header import Header, _lies_under

DEFAULT_COLUMNS = {
    "orphan_attach_em": 3.0,      # a cell this close (in em) to a column is attached to it instead of being dropped
    "orphan_min_lines": 2,        # a header-less column needs cells on at least this many lines
    "attach_gap_em": 1.5,         # a cell under no header closer than this to its left neighbour joins that neighbour's column
}


@dataclass
class ColumnSpec:
    idx: int
    ext0: float
    ext1: float
    header_text: str = ""
    header_lines: list[str] = field(default_factory=list)
    x1: float = 0.0  # boundaries after finalisation
    x2: float = 1.0


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _ruled_columns(header: Header, body: list[Line], columns: list[ColumnSpec], rules: list[tuple[float, float, float]], em_x: float) -> list[ColumnSpec] | None:
    """Columns taken from drawn vertical rules when they form a grid over the whole table (digital PDFs only).

    Used only if the rules enclose every header cell, so a stray vertical line can never override good geometry.
    """
    if not rules or not body or len(columns) < 2:
        return None
    y0 = min(l.y0 for l in header.lines)
    y1 = max(l.y1 for l in body)
    x_lo, x_hi = min(c.ext0 for c in columns), max(c.ext1 for c in columns)
    span = max(y1 - y0, 1e-6)
    xs = sorted(x for x, ya, yb in rules if min(yb, y1) - max(ya, y0) >= 0.8 * span and x_lo - 0.08 <= x <= x_hi + 0.08)
    merged: list[float] = []
    for x in xs:
        if merged and x - merged[-1] < 0.006:
            continue
        merged.append(x)
    inside = [x for x in merged if x_lo - 0.08 <= x <= x_hi + 0.08]
    if len(inside) < 3:
        return None
    intervals = [(a, b) for a, b in zip(inside, inside[1:]) if b - a >= 2 * em_x * 0.6]
    if len(intervals) < 2:
        return None
    result = [ColumnSpec(i, a, b) for i, (a, b) in enumerate(intervals)]
    primary = header.lines[header.primary - header.first]
    for line in header.lines:
        for seg in line.segs:
            owner = next((c for c in result if c.ext0 - 1e-9 <= seg.cx <= c.ext1 + 1e-9), None)
            if owner is None:
                if line is primary:
                    return None  # a header cell outside the ruled grid: the rules are not this table's grid
                continue
            owner.header_lines.append(seg.text)
    if sum(bool(c.header_lines) for c in result) < max(2, len(columns) - 1):
        return None
    for c in result:
        c.header_text = " ".join(c.header_lines).strip()
        c.x1, c.x2 = c.ext0, c.ext1
    return result


def _split_estimated_headers(header: Header, body: list[Line]) -> Header:
    """Raster OCR may return several header cells as ONE phrase box with estimated word positions. When the data
    below clearly forms several separate clusters inside that box, cut the phrase at the clusters (Fase 3.7)."""
    clusters: list[list[float]] = []
    for line in body:
        for seg in line.segs:
            for cluster in clusters:
                if _overlap(seg.x0, seg.x1, cluster[0], cluster[1]) > 0:
                    cluster[0], cluster[1] = min(cluster[0], seg.x0), max(cluster[1], seg.x1)
                    break
            else:
                clusters.append([seg.x0, seg.x1])
    changed = False
    new_lines = []
    for line in header.lines:
        segs = []
        for seg in line.segs:
            parts = _cut_by_clusters(seg, clusters) if seg.estimated else [seg]
            changed |= len(parts) > 1
            segs.extend(parts)
        new_lines.append(Line(segs, line.cy))
    if not changed:
        return header
    return Header(header.first, header.last, header.primary, header.score, new_lines)


def _cut_by_clusters(seg: Seg, clusters: list[list[float]]) -> list[Seg]:
    inside = [c for c in clusters if c[0] >= seg.x0 - 1e-6 and c[1] <= seg.x1 + 1e-6 or _overlap(seg.x0, seg.x1, c[0], c[1]) > 0.5 * (c[1] - c[0])]
    inside.sort()
    if len(inside) < 2:
        return [seg]
    centres = [(c[0] + c[1]) / 2 for c in inside]
    groups: list[list[Token]] = [[] for _ in inside]
    for token in seg.tokens:
        centre = (token.bbox[0] + token.bbox[2]) / 2
        groups[min(range(len(inside)), key=lambda i: abs(centre - centres[i]))].append(token)
    parts = [Seg(tokens) for tokens in groups if tokens]
    return parts if len(parts) > 1 else [seg]


def _best_anchor(seg: Seg, anchors: list[ColumnSpec], tolerance: float) -> int | None:
    candidates = [
        (_overlap(seg.x0, seg.x1, a.ext0, a.ext1), -abs(seg.cx - (a.ext0 + a.ext1) / 2), i)
        for i, a in enumerate(anchors)
        if seg.x0 - tolerance <= a.ext1 and seg.x1 + tolerance >= a.ext0 and (a.ext0 - tolerance <= seg.cx <= a.ext1 + tolerance or _overlap(seg.x0, seg.x1, a.ext0, a.ext1) > 0)
    ]
    return max(candidates)[2] if candidates else None


def infer_columns(header: Header, body: list[Line], em_x: float, options: dict | None = None, rules: list[tuple[float, float, float]] | None = None) -> list[ColumnSpec]:
    opts = {**DEFAULT_COLUMNS, **(options or {})}
    tolerance = 0.5 * em_x
    header = _split_estimated_headers(header, body)
    primary = header.lines[header.primary - header.first]
    anchors = [ColumnSpec(i, s.x0, s.x1) for i, s in enumerate(primary.segs)]

    # Header text: every header line contributes to each anchor it sits over, top to bottom. Labels ABOVE the primary header
    # line (group labels, first half of a wrapped header) are placed after the body has extended the anchors, because a
    # group label is centred over the CELLS and the header WORDS alone do not tell where the cell boundary is.
    pending: list[tuple[Seg, list[ColumnSpec]]] = []
    for line_index, line in enumerate(header.lines):
        above_primary = header.first + line_index < header.primary
        for seg in line.segs:
            hit = [a for a in anchors if _lies_under(seg, Seg_like(a), tolerance)]
            if above_primary:
                pending.append((seg, hit))
                continue
            if not hit and anchors:
                nearest = min(anchors, key=lambda a: abs(seg.cx - (a.ext0 + a.ext1) / 2))
                if abs(seg.cx - (nearest.ext0 + nearest.ext1) / 2) <= opts["orphan_attach_em"] * em_x:
                    hit = [nearest]
            for anchor in hit:
                anchor.header_lines.append(seg.text)

    # Body cells extend anchors; cells under no anchor become candidates for header-less columns.
    orphans: list[tuple[int, Seg]] = []
    for line_index, line in enumerate(body):
        previous: tuple[int, Seg] | None = None
        for seg in line.segs:
            i = _best_anchor(seg, anchors, tolerance)
            if i is None and previous is not None and seg.x0 - previous[1].x1 <= opts["attach_gap_em"] * em_x:
                i = previous[0]  # glued to the cell on its left (e.g. a quote mark split off a word): same column
            if i is None:
                orphans.append((line_index, seg))
                previous = None
                continue
            a = anchors[i]
            spans_two = sum(_overlap(seg.x0, seg.x1, o.ext0, o.ext1) > 0.3 * (seg.x1 - seg.x0) for o in anchors) > 1
            if not spans_two:
                a.ext0, a.ext1 = min(a.ext0, seg.x0), max(a.ext1, seg.x1)
            previous = (i, seg)

    for seg, hit in reversed(pending):  # bottom-most label first, so each insert at the front keeps top-to-bottom order
        hit = _group_span(seg, anchors) or hit
        if not hit and anchors:
            nearest = min(anchors, key=lambda a: abs(seg.cx - (a.ext0 + a.ext1) / 2))
            if abs(seg.cx - (nearest.ext0 + nearest.ext1) / 2) <= opts["orphan_attach_em"] * em_x:
                hit = [nearest]
        for anchor in hit:
            anchor.header_lines.insert(0, seg.text)
    for anchor in anchors:
        anchor.header_text = " ".join(anchor.header_lines).strip()

    # Cells under no header that overlap each other on several lines form a header-less column.
    groups: list[dict] = []
    for line_index, seg in sorted(orphans, key=lambda item: item[1].cx):
        group = next((g for g in groups if _overlap(seg.x0, seg.x1, g["x0"], g["x1"]) > 0), None)
        if group is None:
            groups.append({"x0": seg.x0, "x1": seg.x1, "lines": {line_index}})
        else:
            group["x0"], group["x1"] = min(group["x0"], seg.x0), max(group["x1"], seg.x1)
            group["lines"].add(line_index)
    columns = list(anchors)
    for group in groups:
        if len(group["lines"]) >= opts["orphan_min_lines"]:
            columns.append(ColumnSpec(-1, group["x0"], group["x1"]))

    columns.sort(key=lambda c: c.ext0)
    ruled = _ruled_columns(header, body, columns, rules or [], em_x)
    if ruled is not None:
        columns = ruled
    for i, column in enumerate(columns):
        column.idx = i
    # Boundaries: the middle of the empty space between neighbouring extents (or of their overlap).
    for i, column in enumerate(columns):
        column.x1 = columns[0].ext0 if i == 0 else (columns[i - 1].ext1 + column.ext0) / 2
        column.x2 = column.ext1 if i == len(columns) - 1 else (column.ext1 + columns[i + 1].ext0) / 2
    return columns


def _group_span(seg: Seg, anchors: list[ColumnSpec]) -> list[ColumnSpec]:
    """A label above the header that is centred over the cells of two neighbouring columns spans both of them.

    The anchors are extended by the body here, so they approximate the CELL contents. Text alignment moves the content
    inside its cell, so two readings are accepted: the label centre is near the CENTRE of the union of the two columns
    (left aligned content), or near the BOUNDARY between them (right or centre aligned content). A label over a single
    column sits at that column's centre, which is far from both for every pair.
    """
    best: tuple[float, list[ColumnSpec]] | None = None
    for left, right in zip(anchors, anchors[1:]):
        union0, union1 = left.ext0, right.ext1
        width = union1 - union0
        if width <= 0 or (seg.x1 - seg.x0) > 1.2 * width:
            continue
        to_centre = abs(seg.cx - (union0 + union1) / 2)
        to_boundary = abs(seg.cx - (left.ext1 + right.ext0) / 2) if union0 <= seg.cx <= union1 else float("inf")
        distance = min(to_centre / 0.12, to_boundary / 0.2) / width  # 1.0 or less = accepted by one of the readings
        if distance <= 1.0 and (best is None or distance < best[0]):
            best = (distance, [left, right])
    return best[1] if best else []

class Seg_like:  # minimal adapter so ``_lies_under`` can compare a Seg against an anchor extent
    def __init__(self, anchor: ColumnSpec):
        self.x0, self.x1 = anchor.ext0, anchor.ext1


def column_of_x(columns: list[ColumnSpec], x: float) -> int:
    for column in columns:
        if column.x1 <= x <= column.x2:
            return column.idx
    return min(columns, key=lambda c: min(abs(x - c.x1), abs(x - c.x2))).idx


def split_segment(seg: Seg, columns: list[ColumnSpec]) -> list[tuple[int, list[Token]]]:
    """Cell pieces of a segment. Real segments belong to the column of their centre; ESTIMATED segments
    (raster phrase boxes whose word boxes are only proportions) are cut at the column boundaries so one OCR
    phrase can feed several cells (Fase 3.7)."""
    if not seg.estimated:
        return [(column_of_x(columns, seg.cx), seg.tokens)]
    pieces: dict[int, list[Token]] = {}
    for token in seg.tokens:
        pieces.setdefault(column_of_x(columns, (token.bbox[0] + token.bbox[2]) / 2), []).append(token)
    return sorted(pieces.items())
