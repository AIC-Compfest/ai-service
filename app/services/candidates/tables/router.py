"""Feature-flagged, scored routing for internal table providers."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml

from ..schema import Page, Table
from .token_grid import TokenGridTableProvider
from .pymupdf_provider import PyMuPDFTableProvider
from app.services.validation.arithmetic import validate_table


class TableProvider(Protocol):
    """Discover tables without requiring the public OCR evidence contract."""

    name: str

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]: ...


@lru_cache(maxsize=8)
def _cached_extractor(config_dir: str, provider_order: tuple[str, ...]) -> "TableExtractor":
    return TableExtractor.from_config(Path(config_dir), provider_order)


class TableExtractor:
    """Select a provider by table structure, never merely provider order."""

    def __init__(self, providers: Iterable[TableProvider]):
        self.providers = {provider.name: provider for provider in providers}

    @classmethod
    def from_config(cls, config_dir, provider_order: Iterable[str]) -> "TableExtractor":
        config_dir = Path(config_dir)
        provider = TokenGridTableProvider(config_dir / "line_item_aliases.yaml", config_dir / "abbreviations.yaml")
        resolution = yaml.safe_load((config_dir / "table_resolution.yaml").read_text(encoding="utf-8")) or {}
        # Unknown providers intentionally remain unavailable until their optional
        # dependencies are installed and registered in later steps.
        return cls([PyMuPDFTableProvider(resolution.get("pymupdf")), provider]).ordered(provider_order)

    @classmethod
    def cached(cls, config_dir, provider_order: Iterable[str]) -> "TableExtractor":
        """Process-wide extractor per (config dir, provider order): YAML is read once, not once per page (T-25)."""
        return _cached_extractor(str(Path(config_dir).resolve()), tuple(provider_order))

    def ordered(self, provider_order: Iterable[str]) -> "TableExtractor":
        requested = list(provider_order)
        self.order = [name for name in requested if name in self.providers]
        return self

    def extract(self, page: Page, doc_type: str | None, document_id: str = "document", *, arithmetic_validation: bool = False, relative_tolerance=None, absolute_tolerance=None) -> list[Table]:
        candidates: list[tuple[float, int, list[Table]]] = []
        for position, name in enumerate(self.order):
            tables = self.providers[name].tables(page, doc_type, document_id)
            if not tables:
                continue
            for table in tables:
                table.structural_score = structural_score(table)
                table.reason = table.reason or "structural_score"
                table.validation.setdefault("structural", {"score": round(table.structural_score, 6)})
                if arithmetic_validation:
                    validation = validate_table(table, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance)
                    table.validation["arithmetic"] = _json_validation(validation)
                    if validation["available"]:
                        table.structural_score = round(0.8 * table.structural_score + 0.2 * float(validation["score"]), 6)
            candidates.append((sum(table.structural_score for table in tables) / len(tables), position, tables))
        if not candidates:
            return []
        # Provider order is only a deterministic tie-breaker.
        return max(candidates, key=lambda item: (item[0], -item[1]))[2]


def structural_score(table: Table) -> float:
    """Heuristic completeness score; it is not an accuracy probability."""
    if not table.columns or not table.rows:
        return 0.0
    expected = len(table.columns) * len(table.rows)
    filled = sum(bool(cell.text.strip()) for row in table.rows for cell in row.cells)
    header_ratio = sum(bool(column.header_text.strip()) for column in table.columns) / len(table.columns)
    fill_ratio = filled / expected if expected else 0.0
    return round(0.45 * header_ratio + 0.55 * fill_ratio, 6)


def _json_validation(value):
    if isinstance(value, dict):
        return {key: _json_validation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_validation(item) for item in value]
    return str(value) if hasattr(value, "as_tuple") else value
