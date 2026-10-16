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
    "segment_gap_em": 0.8,
    "block_gap_factor": 2.2,
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


def build_lines(tokens: list[Token], aspect: float | None, options: dict | None = None) -> list[Line]:
    """Cluster tokens into visual lines (by centre y against the running line centre) and split lines into segments."""
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
            if token.bbox[0] - max(t.bbox[2] for t in current) > opts["segment_gap_em"] * em_x:
                segs.append(Seg(current))
                current = [token]
            else:
                current.append(token)
        segs.append(Seg(current))
        lines.append(Line(segs, centre))
    return lines


def split_blocks(lines: list[Line], height: float, options: dict | None = None) -> list[Block]:
    """Group consecutive lines; a vertical gap above ``block_gap_factor`` line heights starts a new block."""
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    blocks: list[Block] = []
    for line in sorted(lines, key=lambda l: l.cy):
        if blocks and line.cy - blocks[-1].lines[-1].cy <= opts["block_gap_factor"] * height:
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


def text_blocks(tokens: list[Token], aspect: float | None, options: dict | None = None) -> tuple[list[Block], float]:
    """Visual lines -> vertical blocks -> side-by-side split. Returns the blocks and the median token height."""
    opts = {**DEFAULT_GEOMETRY, **(options or {})}
    lines = build_lines(tokens, aspect, opts)
    if not lines:
        return [], 0.0
    height = median_height([t for t in tokens if t.text.strip()])
    blocks: list[Block] = []
    for block in split_blocks(lines, height, opts):
        blocks.extend(split_side_by_side(block, height, opts))
    return blocks, height
