"""Candidate generation + LightGBM on the extraction path: additive, conservative, never fatal."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.ml import service
from app.services.extraction.ml.service import promote_ml_fields, run_ml_extraction
from app.tools.table_eval.synthetic import build_fixture

POD_EVIDENCE = Path(settings.config_dir).parent / "data" / "pod_phase2" / "evidence" / "pod-1-v1.json"


def _rule_field(name: str, unknown: bool) -> ExtractedField:
    return ExtractedField(field=name, value=None if unknown else "Budi", confidence=0.0 if unknown else 0.9, source=EvidenceSource(document_id="d", page=1), method=ExtractionMethod.RULE_BASED, is_unknown=unknown)


def _ml(status: str, name: str = "pihak_pertama_nama") -> dict:
    return {"status": "ok", "fields": [{"field": name, "status": status, "score": 0.95, "value": "PT Maju", "raw_text": "PT Maju", "page": 1, "bbox_norm": [0.1, 0.1, 0.3, 0.12]}]}


@pytest.fixture(autouse=True)
def _fresh_cache():
    service.reset_cache()
    yield
    service.reset_cache()


@pytest.mark.skipif(not POD_EVIDENCE.exists(), reason="POD evidence sample not available")
def test_pod_candidates_are_scored_by_the_lightgbm_bundle():
    result = run_ml_extraction("t-pod", "POD", json.loads(POD_EVIDENCE.read_text(encoding="utf-8")))
    assert result["status"] == "ok" and result["model_set"] == "pod_phase2"
    (field,) = [f for f in result["fields"] if f["field"] == "pihak_pertama_nama"]
    assert field["status"] in {"accepted", "review", "unknown"} and 0.0 <= field["score"] <= 1.0


def test_doc_type_without_a_bundle_is_reported_not_guessed():
    assert run_ml_extraction("t", "INVOICE", {"pages": []})["status"] == "no_model_for_doc_type"


def test_flag_off_disables_the_ml_path(monkeypatch):
    monkeypatch.setattr(settings, "enable_ml_extraction", False)
    assert run_ml_extraction("t", "POD", {"pages": []})["status"] == "disabled"


def test_failure_in_the_ml_path_is_contained(monkeypatch):
    def boom(_path):
        raise RuntimeError("corrupt bundle")

    monkeypatch.setattr(service, "_extractor", boom)
    result = run_ml_extraction("t", "POD", {"pages": []})
    assert result["status"] == "error" and "corrupt bundle" in result["error"]


def test_only_accepted_ml_values_fill_unknown_fields_of_the_same_name():
    unknown, known = _rule_field("pihak_pertama_nama", True), _rule_field("pihak_pertama_nama", False)
    promoted = promote_ml_fields([unknown], _ml("accepted"), "d")[0]
    assert promoted.value == "PT Maju" and promoted.method == ExtractionMethod.ML_MODEL and not promoted.is_unknown
    assert promote_ml_fields([unknown], _ml("review"), "d")[0].is_unknown  # review is not enough
    assert promote_ml_fields([known], _ml("accepted"), "d")[0].value == "Budi"  # rule value wins
    assert promote_ml_fields([_rule_field("other", True)], _ml("accepted"), "d")[0].is_unknown  # different name


def test_full_pipeline_response_carries_ml_and_table_groups_additively():
    pdf = build_fixture("multipage_repeated_header").pdf
    response = TestClient(app).post("/v1/extract/pipeline/full", files={"file": ("mp.pdf", pdf, "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert "ml_extraction" in body and "table_groups" in body
    assert {"document_id", "status", "classification", "extracted_fields", "pipeline_version"} <= body.keys()  # legacy keys intact
