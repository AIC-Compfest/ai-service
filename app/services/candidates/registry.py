from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FieldSpec:
    doc_type: str
    field: str
    cand_type: str
    cardinality: str = "one"
    model: str = "lgbm"


class FieldRegistry:
    def __init__(self, fields: list[FieldSpec]):
        self.fields = fields

    @classmethod
    def load(cls, path: str | Path) -> "FieldRegistry":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw_fields = data.get("fields")
        if raw_fields is None:
            raw_fields = []
            for doc_type, doc_fields in data.items():
                if not isinstance(doc_fields, dict):
                    continue
                for field, spec in doc_fields.items():
                    spec = spec or {}
                    raw_fields.append({"doc_type": doc_type, "field": field, **spec})
        fields = [FieldSpec(**item) for item in raw_fields]
        if not fields:
            raise ValueError("field registry contains no fields")
        return cls(fields)

    def for_doc(self, doc_type: str) -> list[FieldSpec]:
        return [field for field in self.fields if field.doc_type == doc_type and field.model != "none"]

    def fields_for_type(self, doc_type: str, cand_type: str) -> list[str]:
        return [field.field for field in self.for_doc(doc_type) if field.cand_type == cand_type]

    def field_specs_for_type(self, doc_type: str, cand_type: str) -> list[FieldSpec]:
        return [field for field in self.for_doc(doc_type) if field.cand_type == cand_type]
