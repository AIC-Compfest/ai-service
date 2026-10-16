from __future__ import annotations

import json
from pathlib import Path

from app.core.config import candidates_config_dir
from typing import Any

import numpy as np

from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField, ExtractionMethod
from app.services.candidates.pipeline import CandidatePipeline
from app.services.candidates.registry import FieldRegistry
from app.services.training.bundle import load_bundle

from .decode import decode


class MLExtractionResult:
    def __init__(self, document_id: str, doc_type: str, fields: dict[str, dict[str, Any]], model_set_id: str = ""):
        self.document_id, self.doc_type, self.fields, self.model_set_id = document_id, doc_type, fields, model_set_id

    def to_extracted_fields(self) -> list[ExtractedField]:
        result = []
        for field, value in self.fields.items():
            chosen = value.get("candidates", [])[0][0] if value.get("candidates") else None
            score = float(value.get("score", 0))
            result.append(ExtractedField(field=field, value=chosen.get("m_parsed_value") if chosen and value["status"] != "unknown" else None, confidence=score, source=EvidenceSource(document_id=self.document_id, page=int(chosen.get("m_page", 1)) if chosen else 1, bbox=BoundingBox(x1=float(chosen["m_x1"],), y1=float(chosen["m_y1"]), x2=float(chosen["m_x2"]), y2=float(chosen["m_y2"])) if chosen else None), method=ExtractionMethod.ML_MODEL, is_unknown=value["status"] == "unknown"))
        return result


class MLFieldExtractor:
    def __init__(self, model_set_dir: str | Path, candidate_config_dir: str | Path | None = None, *, allow_mismatch: bool = False):
        self.model_set_dir = Path(model_set_dir)
        candidate_config_dir = Path(candidate_config_dir) if candidate_config_dir is not None else candidates_config_dir()
        self.pipeline = CandidatePipeline(candidate_config_dir)
        self.registry = FieldRegistry.load(candidate_config_dir / "field_registry.yaml")
        self.bundles = {}
        for path in self.model_set_dir.glob("*/bundle.json"):
            model, metadata = load_bundle(path.parent, allow_mismatch=allow_mismatch)
            self.bundles[(metadata["doc_type"], metadata["field"])] = (model, metadata)

    def extract(self, document_id: str, doc_type: str, ocr_result: dict[str, Any]) -> MLExtractionResult:
        rows = self.pipeline.generate(document_id, doc_type, ocr_result, "inference")
        scores = {}
        thresholds = {}
        for (bundle_doc, field), (model, metadata) in self.bundles.items():
            if bundle_doc != doc_type:
                continue
            candidates = [row for row in rows if row["m_cand_type"] == metadata["cand_type"]]
            matrix = np.asarray([[_predict_value(row.get(column, ""), metadata.get("feature_vocabs", {}).get(column, [])) for column in metadata["feature_columns"]] for row in candidates], dtype=float)
            probabilities = model.predict_proba(matrix)[:, 1] if candidates else []
            scores[field] = list(zip(candidates, probabilities))
            thresholds[field] = metadata.get("thresholds", {"t_unknown": .4, "t_auto": .9})
        specs = {field.field: {"cardinality": field.cardinality} for field in self.registry.for_doc(doc_type)}
        return MLExtractionResult(document_id, doc_type, decode(scores, specs, thresholds), self.model_set_dir.name)


def _predict_value(value: Any, vocab: list[str]) -> float:
    if value in {None, ""}:
        return float("nan")
    if vocab:
        return float(vocab.index(value) if value in vocab else vocab.index("other") if "other" in vocab else -1)
    return float(value)