"""Candidate generation + LightGBM scoring on the extraction path (additive to the rule-based extractors).

``run_ml_extraction`` never raises: any failure is logged and reported as ``status="error"`` so the rule-based result is
always returned. ``promote_ml_fields`` is deliberately conservative: an ML value replaces a rule field only when that
rule field is unknown, has the same name, and the ML decoder marked it "accepted" (score >= the bundle's t_auto).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField, ExtractionMethod

from .predictor import MLFieldExtractor

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _extractor(model_set_dir: str) -> MLFieldExtractor | None:
    path = Path(model_set_dir)
    if not path.is_dir() or not any(path.glob("*/bundle.json")):
        return None
    return MLFieldExtractor(path)


def reset_cache() -> None:
    _extractor.cache_clear()


def run_ml_extraction(document_id: str, doc_type: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Score field candidates of ``evidence`` (pages of one segment) with the LightGBM bundles of ``doc_type``."""
    if not settings.enable_ml_extraction:
        return {"status": "disabled", "doc_type": doc_type, "fields": []}
    try:
        extractor = _extractor(str(settings.ml_model_set_dir))
        if extractor is None:
            return {"status": "no_model_set", "doc_type": doc_type, "fields": []}
        fields_with_bundle = sorted(field for (bundle_doc, field) in extractor.bundles if bundle_doc == doc_type)
        if not fields_with_bundle:
            return {"status": "no_model_for_doc_type", "doc_type": doc_type, "model_set": extractor.model_set_dir.name, "fields": []}
        result = extractor.extract(document_id, doc_type, evidence)
        fields = []
        for name in fields_with_bundle:
            entry = result.fields.get(name, {"status": "unknown", "candidates": [], "score": 0.0})
            chosen = entry["candidates"][0][0] if entry.get("candidates") else None
            fields.append(
                {
                    "field": name,
                    "status": entry["status"],  # accepted | review | unknown
                    "score": round(float(entry.get("score", 0.0)), 6),
                    "value": chosen.get("m_parsed_value") if chosen else None,
                    "raw_text": chosen.get("m_raw_text") if chosen else None,
                    "page": int(chosen.get("m_page", 1)) if chosen else None,
                    "bbox_norm": [float(chosen[k]) for k in ("m_x1", "m_y1", "m_x2", "m_y2")] if chosen else None,
                }
            )
        return {"status": "ok", "doc_type": doc_type, "model_set": result.model_set_id, "fields": fields}
    except Exception as error:  # the ML path is additive; it must never break extraction
        logger.warning("ml_extraction_failed", document_id=document_id, doc_type=doc_type, error=str(error))
        return {"status": "error", "doc_type": doc_type, "error": str(error), "fields": []}


def promote_ml_fields(rule_fields: list[ExtractedField], ml: dict[str, Any], document_id: str) -> list[ExtractedField]:
    """Return ``rule_fields`` with unknown same-name fields replaced by accepted ML values; other fields are untouched."""
    accepted = {f["field"]: f for f in ml.get("fields", []) if f["status"] == "accepted" and f.get("bbox_norm")}
    if not accepted:
        return rule_fields
    result = []
    for field in rule_fields:
        ml_field = accepted.get(field.field)
        if ml_field is not None and field.is_unknown:
            x1, y1, x2, y2 = ml_field["bbox_norm"]
            field = ExtractedField(
                field=field.field, value=ml_field["value"], confidence=min(max(ml_field["score"], 0.0), 1.0),
                source=EvidenceSource(document_id=document_id, page=int(ml_field["page"]), bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)),
                method=ExtractionMethod.ML_MODEL, is_unknown=False,
            )
        result.append(field)
    return result
