"""Scored routing for internal table providers (Docling TableFormer is the only provider)."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml

from ..schema import Page, Table
from .docling_provider import DoclingTableProvider
from .rows import summary_pattern
from .scoring import arbitrate, classify_role, table_quality


class TableProvider(Protocol):
    """Discover tables without requiring the public OCR evidence contract."""

    name: str

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]: ...


@lru_cache(maxsize=8)
def _cached_extractor(config_dir: str, provider_order: tuple[str, ...]) -> "TableExtractor":
    return TableExtractor.from_config(Path(config_dir), provider_order)


class TableExtractor:
    """Collect tables from the registered providers, keep the best table of each overlapping group, classify roles.

    Arbitration is per TABLE, not per provider; provider order is only a deterministic tie-breaker.
    """

    def __init__(self, providers: Iterable[TableProvider], scoring: dict | None = None, summary_labels: Iterable[str] = (), docling: dict | None = None):
        self.docling = docling or {}
        self.providers = {provider.name: provider for provider in providers}
        self.order = list(self.providers)
        self.scoring = scoring or {}
        self.summary_pattern = summary_pattern(list(summary_labels)) if summary_labels else None

    @classmethod
    def from_config(cls, config_dir, provider_order: Iterable[str]) -> "TableExtractor":
        config_dir = Path(config_dir)
        resolution = yaml.safe_load((config_dir / "table_resolution.yaml").read_text(encoding="utf-8")) or {}
        docling = resolution.get("docling") or {}
        docling_options = {**docling, "summary_labels": resolution.get("summary_labels", ()), "refine": resolution.get("refine")}
        # Docling is the only provider; its models load lazily on the first conversion.
        return cls([DoclingTableProvider(docling_options)], resolution.get("scoring"), resolution.get("summary_labels", ()), docling).ordered(provider_order)

    @classmethod
    def cached(cls, config_dir, provider_order: Iterable[str]) -> "TableExtractor":
        """Process-wide extractor per (config dir, provider order): YAML is read once, not once per page (T-25)."""
        return _cached_extractor(str(Path(config_dir).resolve()), tuple(provider_order))

    def ordered(self, provider_order: Iterable[str]) -> "TableExtractor":
        requested = list(provider_order)
        self.order = [name for name in requested if name in self.providers]
        return self

    def extract(self, page: Page, doc_type: str | None, document_id: str = "document", *, arithmetic_validation: bool = False, relative_tolerance=None, absolute_tolerance=None) -> list[Table]:
        candidates: list[Table] = []
        for position, name in enumerate(self.order):
            found = self.providers[name].tables(page, doc_type, document_id)
            for table in found:
                table.structural_score = table_quality(table, self.scoring) - position * 1e-6  # order breaks exact ties
                table.reason = table.reason or "structural_score"
                table.validation.setdefault("structural", {"score": round(table.structural_score, 6)})
                if arithmetic_validation:
                    from app.services.validation.arithmetic import validate_table  # lazy: validation imports the candidates package

                    validation = validate_table(table, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance)
                    table.validation["arithmetic"] = _json_validation(validation)
                    if validation["available"]:
                        table.structural_score = round(0.8 * table.structural_score + 0.2 * float(validation["score"]), 6)
                candidates.append(table)

        tables = arbitrate(candidates, self.scoring)
        for number, table in enumerate(tables, start=1):
            table.table_id = f"{document_id}-p{page.page_number}-t{number}"
            table.role, table.role_reason = classify_role(table, self.summary_pattern, self.scoring)
        return tables


def structural_score(table: Table) -> float:
    """Structural plausibility score of one table (kept as the public name; see ``scoring.table_quality``)."""
    return table_quality(table)


def _json_validation(value):
    if isinstance(value, dict):
        return {key: _json_validation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_validation(item) for item in value]
    return str(value) if hasattr(value, "as_tuple") else value
