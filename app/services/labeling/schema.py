from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FieldStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    MISSED = "missed_by_generator"
    UNREADABLE = "unreadable"


class ReviewState(StrEnum):
    PENDING = "pending_review"
    AUTO = "auto_accepted"
    CONFIRMED = "human_confirmed"
    CORRECTED = "human_corrected"


class LabelSource(StrEnum):
    LLM_BASE = "llm_base"
    TRANSFER = "transfer"
    LLM_VERIFY = "llm_verify"
    HUMAN = "human"


@dataclass
class TruthField:
    status: str
    value_text: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    anchor_text: str | None = None
    evidence: str | None = None
    confidence: float = 0.0
    also_at: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TruthRecord:
    document_id: str
    doc_type: str
    prompt_version: str = ""
    llm_model: str = ""
    fields: dict[str, TruthField] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    m_pipeline_version: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TruthRecord":
        fields = {name: TruthField(**value) for name, value in (data.get("fields") or {}).items()}
        return cls(data["document_id"], data["doc_type"], data.get("prompt_version", ""), data.get("llm_model", ""), fields, data.get("tables", []), data.get("m_pipeline_version", ""))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fields"] = {name: asdict(value) for name, value in self.fields.items()}
        return data


@dataclass
class FieldLabel:
    label_version: str
    document_id: str
    doc_type: str
    template_group: str
    is_base: int
    field: str
    status: str
    review_state: str
    label_source: str
    confidence: float = 0.0
    value_text: str = ""
    value_norm: str = ""
    page: int | None = None
    positive_candidate_ids: str = ""
    closest_candidate_id: str = ""
    grounding_method: str = "none"
    grounding_score: float = 0.0
    transfer_score: float = 0.0
    transfer_margin: float = 0.0
    anchor_text: str = ""
    notes: str = ""
    llm_model: str = ""
    prompt_version: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    m_pipeline_version: str = ""

    def key(self) -> tuple[str, str]:
        return self.document_id, self.field
