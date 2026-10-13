from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from rapidfuzz.fuzz import token_set_ratio

from app.services.candidates.parsers import parse_date, parse_number, parse_time
from app.services.candidates.text_norm import norm_alnum, norm_text

from .schema import FieldLabel, FieldStatus, LabelSource, ReviewState, TruthField, TruthRecord


def normalize_truth_value(cand_type: str, value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if cand_type == "date":
        return parse_date(value).iso or norm_text(value)
    if cand_type in {"amount", "qty", "weight", "percent", "days"}:
        parsed = parse_number(value.rstrip("%"))
        return str(parsed.value) if parsed.value is not None else norm_text(value)
    if cand_type == "time":
        parsed = parse_time(value)
        return parsed["value"] if parsed else norm_text(value)
    if cand_type == "id":
        return norm_alnum(value)
    if cand_type == "enum":
        normalized = norm_text(value)
        return {"baik": "GOOD", "good": "GOOD", "lengkap": "GOOD", "rusak": "DAMAGED", "damaged": "DAMAGED", "kurang": "SHORTAGE", "senin": "MONDAY", "selasa": "TUESDAY", "rabu": "WEDNESDAY", "kamis": "THURSDAY", "jumat": "FRIDAY", "sabtu": "SATURDAY", "minggu": "SUNDAY"}.get(normalized, value.upper())
    return norm_text(value)


def ground_truth_field(truth: TruthRecord, field: str, cand_type: str, candidates: list[dict[str, Any]], *, template_group: str = "", is_base: int = 1, pipeline_version: str = "", label_version: str = "1") -> FieldLabel:
    truth_field = truth.fields.get(field, TruthField(FieldStatus.ABSENT))
    base = FieldLabel(label_version, truth.document_id, truth.doc_type, template_group, is_base, field, truth_field.status, ReviewState.PENDING, LabelSource.LLM_BASE, truth_field.confidence, truth_field.value_text or "", normalize_truth_value(cand_type, truth_field.value_text), truth_field.page, "", "", "none", 0.0, 0.0, 0.0, truth_field.anchor_text or "", "", truth.llm_model, truth.prompt_version, "", "", pipeline_version)
    if truth_field.status != FieldStatus.PRESENT or not truth_field.value_text:
        return base
    matches = []
    for candidate in candidates:
        candidate_norm = candidate.get("m_parsed_norm", "")
        if candidate_norm == base.value_norm:
            matches.append((candidate, 1.0, "exact"))
        elif cand_type in {"party", "address", "city", "text"}:
            score = token_set_ratio(norm_text(truth_field.value_text), norm_text(candidate.get("m_raw_text", ""))) / 100
            if score >= 0.9:
                matches.append((candidate, score, "fuzzy"))
        elif cand_type == "id" and candidate_norm != base.value_norm and len(base.value_norm) >= 6 and _levenshtein(base.value_norm, candidate_norm) <= 1:
            matches.append((candidate, 0.9, "ocr_tolerant"))
    if truth_field.page is not None:
        page_matches = [match for match in matches if int(match[0].get("m_page", 0)) == truth_field.page]
        if page_matches:
            matches = page_matches
    if not matches:
        base.status = FieldStatus.MISSED
        base.closest_candidate_id = min(candidates, key=lambda item: token_set_ratio(norm_text(truth_field.value_text), norm_text(item.get("m_raw_text", ""))), default={}).get("m_candidate_id", "")
        base.notes = "no candidate matched normalized truth"
        return base
    best_score = max(score for _, score, _ in matches)
    positives = [candidate["m_candidate_id"] for candidate, score, _ in matches if score == best_score or (best_score == 1 and score == 1)]
    best = next(match for match in matches if match[1] == best_score)
    base.positive_candidate_ids = "|".join(sorted(set(positives)))
    base.closest_candidate_id = best[0]["m_candidate_id"]
    base.grounding_method = "duplicate_all" if len(positives) > 1 else best[2]
    base.grounding_score = best_score
    base.confidence = truth_field.confidence * best_score
    return base


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]