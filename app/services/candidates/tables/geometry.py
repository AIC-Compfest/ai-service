"""Text geometry for table discovery: visual lines, cell-like segments, text blocks, side-by-side split.

Every distance is relative to the text size (the em = median token height) so the logic is independent of
DPI and page size. No vendor, template or document-type knowledge lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from app.tools.table_eval.normalize import is_numeric, norm_text

from ..schema import Token

DEFAULT_GEOMETRY = {
    "line_tolerance": 0.5,
    "segment_gap_em": 0.55,
    "block_gap_factor": 2.2,
    "block_pitch_window": 4,
    "block_periodic_min": 2,
    "block_periodic_tolerance": 0.3,
    "block_periodic_window": 6,
    "side_gutter_min": 0.05,
    "side_header_lines": 4,
    "side_header_repeats": 2,
    "side_header_jaccard": 0.5,
    "side_min_lines": 3,
    "default_aspect": 0.7071,
}


@dataclass
class Seg:
    """A run of tokens on one visual line that reads as one cell."""

    tokens: list[Token]

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens)

    @property
    def x0(self) -> float:
        return min(token.bbox[0] for token in self.tokens)

    @property
    def x1(self) -> float:
        return max(token.bbox[2] for token in self.tokens)

    @property
    def y0(self) -> float:
        return min(token.bbox[1] for token in self.tokens)

    @property
    def y1(self) -> float:
        return max(token.bbox[3] for token in self.tokens)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def estimated(self) -> bool:
        return all(token.box_estimated for token in self.tokens) and len(self.tokens) > 1


@dataclass
class Line:
    segs: list[Seg]
    cy: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segs)

    @property
    def y0(self) -> float:
        return min(seg.y0 for seg in self.segs)

    @property
    def y1(self) -> float:
        return max(seg.y1 for seg in self.segs)

    @property
    def x0(self) -> float:
        return min(seg.x0 for seg in self.segs)

    @property
    def x1(self) -> float:
        return max(seg.x1 for seg in self.segs)


@dataclass
class Block:
    lines: list[Line] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(l.x0 for l in self.lines), min(l.y0 for l in self.lines), max(l.x1 for l in self.lines), max(l.y1 for l in self.lines))


def median_height(tokens: list[Token]) -> float:
    return median(token.bbox[3] - token.bbox[1] for token in tokens) if tokens else 0.0


def vertical_rules(source_page, min_length: float = 0.015) -> list[tuple[float, float, float]]:
    """Drawn vertical rules ``(x, y0, y1)`` of a live PDF page, normalized to 0..1. Empty when unavailable."""
    if source_page is None or not hasattr(source_page, "get_drawings"):
        return []
    try:
        width, height = float(source_page.rect.width), float(source_page.rect.height)
        rules: list[tuple[float, float, float]] = []

        def add(x0, y0, x1, y1):
            if abs(x1 - x0) <= 1.0 and abs(y1 - y0) / height >= min_length:
                rules.append((((x0 + x1) / 2) / width, min(y0, y1) / height, max(y0, y1) / height))

        for drawing in source_page.get_drawings():
            for item in drawing["items"]:
                if item[0] == "l":
                    add(item[1].x, item[1].y, item[2].x, item[2].y)
                elif item[0] == "re":
                    rect = item[1]
                    add(rect.x0, rect.y0, rect.x0, rect.y1)
                    add(rect.x1, rect.y0, rect.x1, rect.y1)
        return rules
    except Exception:  # a drawing the library cannot parse must not break table discovery
        return []


def build_lines(tokens: list[Token], aspect: float | None, options: dict | None = None, rules: list[tuple[float, float, float]] | None = None) -> list[Line]:
    """Cluster tokens into visual lines (by centre y against the running line centre) and split lines into segments.

    Segments also split where a drawn vertical rule runs between two tokens (cells of a ruled table never merge).
    """
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    tokens = [t for t in tokens if t.text.strip()]
    if not tokens:
        return []
    height = median_height(tokens)
    clusters: list[list[Token]] = []
    centres: list[float] = []
    for token in sorted(tokens, key=lambda t: (t.cy, t.bbox[0])):
        if clusters and abs(token.cy - centres[-1]) <= opts["line_tolerance"] * height:
            clusters[-1].append(token)
            centres[-1] = sum(t.cy for t in clusters[-1]) / len(clusters[-1])
        else:
            clusters.append([token])
            centres.append(token.cy)
    ratio = aspect or opts["default_aspect"]
    lines = []
    for cluster, centre in zip(clusters, centres):
        cluster.sort(key=lambda t: t.bbox[0])
        em_x = median_height(cluster) / ratio  # one em measured in page-width units
        segs, current = [], [cluster[0]]
        for token in cluster[1:]:
            left = max(t.bbox[2] for t in current)
            ruled = any(left - 1e-4 <= x <= token.bbox[0] + 1e-4 and y0 <= centre <= y1 for x, y0, y1 in (rules or ()))
            if ruled or token.bbox[0] - left > opts["segment_gap_em"] * em_x:
                segs.append(Seg(current))
                current = [token]
            else:
                current.append(token)
        segs.append(Seg(current))
        lines.append(Line(segs, centre))
    return lines


def _is_row_spacing(gaps: list[float], i: int, pitch: float, opts: dict, below: Line | None = None) -> bool:
    """A large gap that repeats is the spacing BETWEEN multi-line rows, not the end of a block.

    Items made of two text lines (description + wrapped description, with the numbers centred between them) have small
    gaps inside a row and a larger, regular gap between rows; the local pitch only sees the small ones. A gap counts as
    row spacing when at least ``block_periodic_min`` other large gaps of about the same size sit within
    ``block_periodic_window`` gaps of it. A single large gap (a real break) never qualifies.
    """
    needed = int(opts["block_periodic_min"])
    if needed <= 0:
        return False
    if below is not None and len(below.segs) >= 2 and not any(ch.isdigit() for seg in below.segs for ch in seg.text):
        return False  # a multi-cell line without digits right after the gap is the header of ANOTHER table
    large = opts["block_gap_factor"] * pitch
    window = int(opts["block_periodic_window"])
    tolerance = opts["block_periodic_tolerance"] * gaps[i]
    similar = sum(
        1 for j in range(max(0, i - window), min(len(gaps), i + window + 1))
        if j != i and gaps[j] > large and abs(gaps[j] - gaps[i]) <= tolerance
    )
    return similar >= needed


def split_blocks(lines: list[Line], height: float, options: dict | None = None) -> list[Block]:
    """Group consecutive lines into blocks.

    A vertical gap starts a new block when it exceeds ``block_gap_factor`` times the LOCAL line pitch (median of the
    neighbouring gaps, never less than one line height). A fixed multiple of the text height would split tables
    whose rows are simply spaced widely.
    """
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    ordered = sorted(lines, key=lambda l: l.cy)
    gaps = [b.cy - a.cy for a, b in zip(ordered, ordered[1:])]
    window = opts["block_pitch_window"]
    blocks: list[Block] = []
    for index, line in enumerate(ordered):
        if index == 0:
            blocks.append(Block([line]))
            continue
        near = gaps[max(0, index - 1 - window): index + window]
        pitch = max(height, median(near))
        if gaps[index - 1] <= opts["block_gap_factor"] * pitch or _is_row_spacing(gaps, index - 1, pitch, opts, line):
            blocks[-1].lines.append(line)
        else:
            blocks.append(Block([line]))
    return blocks


def _gutters(block: Block) -> list[tuple[float, float]]:
    """Vertical bands (x0, x1) that no segment of any line in the block covers."""
    spans = sorted((seg.x0, seg.x1) for line in block.lines for seg in line.segs)
    gutters, reach = [], spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 > reach:
            gutters.append((reach, x0))
        reach = max(reach, x1)
    return gutters


def _side_lines(block: Block, x0: float, x1: float) -> list[list[Line]]:
    sides = []
    for keep in (lambda s: s.x1 <= x0 + 1e-9, lambda s: s.x0 >= x1 - 1e-9):
        lines = []
        for line in block.lines:
            segs = [s for s in line.segs if keep(s)]
            if segs:
                lines.append(Line(segs, sum(s.y0 + s.y1 for s in segs) / (2 * len(segs))))
        sides.append(lines)
    return sides


def _repeated_header(block: Block, x0: float, x1: float, lines_checked: int, min_repeats: int, min_jaccard: float) -> bool:
    """True when the same non-numeric cell texts appear on the same line on both sides of the gutter.

    Two tables placed next to each other typically repeat header words ("No", "Qty", "Tarif") on one visual line,
    which a single table never does. A wide empty gutter alone is NOT evidence (short descriptions create one).
    """
    for line in block.lines[:lines_checked]:
        # Word level (not segment level) so raster phrase boxes that merge neighbouring cells still compare correctly.
        left = {norm_text(t.text) for s in line.segs if s.x1 <= x0 + 1e-9 for t in s.tokens if not is_numeric(t.text)}
        right = {norm_text(t.text) for s in line.segs if s.x0 >= x1 - 1e-9 for t in s.tokens if not is_numeric(t.text)}
        shared = (left & right) - {""}
        union = (left | right) - {""}
        if len(shared) >= min_repeats and len(shared) / len(union) >= min_jaccard:
            return True
    return False


def split_side_by_side(block: Block, height: float, options: dict | None = None, depth: int = 0) -> list[Block]:
    """Split a block at an empty vertical gutter that separates two tables with a repeated header row."""
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    if depth >= 3 or len(block.lines) < opts["side_min_lines"]:
        return [block]
    for x0, x1 in sorted(_gutters(block), key=lambda g: g[0] - g[1]):
        if x1 - x0 < opts["side_gutter_min"]:
            break
        if not _repeated_header(block, x0, x1, opts["side_header_lines"], opts["side_header_repeats"], opts["side_header_jaccard"]):
            continue
        sides = _side_lines(block, x0, x1)
        if any(len(lines) < opts["side_min_lines"] for lines in sides):
            continue
        result: list[Block] = []
        for lines in sides:
            for sub in split_blocks(lines, height, opts):
                result.extend(split_side_by_side(sub, height, opts, depth + 1))
        return result
    return [block]


def text_blocks(tokens: list[Token], aspect: float | None, options: dict | None = None, rules: list[tuple[float, float, float]] | None = None) -> tuple[list[Block], float]:
    """Visual lines -> vertical blocks -> side-by-side split. Returns the blocks and the median token height."""
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    lines = build_lines(tokens, aspect, opts, rules)
    if not lines:
        return [], 0.0
    height = median_height([t for t in tokens if t.text.strip()])
    blocks: list[Block] = []
    for block in split_blocks(lines, height, opts):
        blocks.extend(split_side_by_side(block, height, opts))
    return blocks, height
