from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Token:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0
    line_idx: int = 0
    tok_idx: int = 0
    box_estimated: bool = False
    row_idx: int = 0

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class Page:
    page_number: int
    tokens: list[Token]
    lines: list[list[Token]]
    engine_used: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Cell:
    text: str
    bbox: tuple[float, float, float, float]
    tokens: list[Token] = field(default_factory=list)
    row_idx: int = 0
    col_idx: int = 0
    is_header: bool = False


@dataclass
class Column:
    idx: int
    x1: float
    x2: float
    header_text: str
    header_lines: list[str] = field(default_factory=list)


@dataclass
class TableRow:
    idx: int
    cells: list[Cell]
    bbox: tuple[float, float, float, float]
    continuation_rows: int = 0
    kind_hint: str = "item"


@dataclass
class Table:
    table_id: str
    page_start: int
    page_end: int
    layout_kind: str
    source: str
    columns: list[Column]
    rows: list[TableRow]
    bbox: tuple[float, float, float, float]
    conf: float = 0.0
    stitched_pages: list[int] = field(default_factory=list)
    reason: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    structural_score: float = 0.0


@dataclass
class Candidate:
    cand_type: str
    page: Page
    tokens: list[Token]
    raw_text: str
    parsed_value: str = ""
    parsed_norm: str = ""
    reading_idx: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(t.bbox[0] for t in self.tokens), min(t.bbox[1] for t in self.tokens), max(t.bbox[2] for t in self.tokens), max(t.bbox[3] for t in self.tokens))

    @property
    def line_idx(self) -> int:
        return self.tokens[0].line_idx


@dataclass(frozen=True)
class NumberParse:
    value: Decimal | None
    fmt: str
    ambiguous: bool = False
    value_alt: Decimal | None = None
    negative: bool = False
    sym: str | None = None


@dataclass(frozen=True)
class DateParse:
    iso: str | None
    fmt: str = "invalid"
    month_lang: str = ""
    month_fuzzy: bool = False
    two_digit_year: bool = False
    ambiguous_dm: bool = False
    sep: str = ""
