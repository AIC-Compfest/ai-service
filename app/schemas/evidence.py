"""
Evidence Model — PRD §18.

Setiap extracted value HARUS memiliki value, confidence, source
(document_id, page, bbox), dan extraction method. Tidak ada critical value
tanpa evidence yang bisa ditelusuri (PRD §18, §60 No Hallucination Rule).
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    OCR = "ocr"
    LAYOUT = "layout"
    VLM = "vlm"
    RULE_BASED = "rule_based"
    MANUAL = "manual"
    ML_MODEL = "ml_model"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class EvidenceSource(BaseModel):
    document_id: str
    page: int
    bbox: BoundingBox | None = None


class ExtractedField(BaseModel):
    """
    Satu field hasil ekstraksi + evidence-nya.
    Jika value tidak dapat dipastikan, gunakan value=None dan
    confidence rendah — JANGAN mengarang nilai (PRD §60).
    """

    field: str
    value: Any | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: EvidenceSource
    method: ExtractionMethod
    is_unknown: bool = False  # true jika evidence tidak cukup (INSUFFICIENT_EVIDENCE)
