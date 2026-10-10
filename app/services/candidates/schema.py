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