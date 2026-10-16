"""Token-geometry table provider.

Works from OCR/PDF word boxes only (no ruling lines, no PDF object), so it serves digital PDFs and raster OCR
alike. The pipeline is: text blocks -> scored header -> columns from header AND data -> rows with summary and
end-of-table rules. See ``structure.py``. ``doc_type`` only adds an alias bonus to the header score.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..fuzzy_fields import FuzzyFieldMatcher
from ..schema import Page, Table
from .structure import discover_tables


class TokenGridTableProvider:
    """Build columnar tables directly from OCR token geometry."""

    name = "token_grid"

    def __init__(self, aliases_path, abbreviations_path):
        self.aliases_path = aliases_path
        self.abbreviations_path = abbreviations_path
        self.config = yaml.safe_load((Path(aliases_path).parent / "table_resolution.yaml").read_text(encoding="utf-8")) or {}
        self._matchers: dict[str, FuzzyFieldMatcher] = {}  # one matcher per doc type; YAML is read once per type

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        return discover_tables(page, self._matcher(doc_type), self.config, document_id)

    def _matcher(self, doc_type: str | None) -> FuzzyFieldMatcher | None:
        if not doc_type:
            return None
        if doc_type not in self._matchers:
            self._matchers[doc_type] = FuzzyFieldMatcher(doc_type, self.aliases_path, self.abbreviations_path)
        return self._matchers[doc_type]
