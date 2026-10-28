"""Config-driven, evidence-backed key/value extraction for review registry fields."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.utils import parse_date_string


@lru_cache(maxsize=1)
def _review_policy() -> dict:
    path = Path(settings.config_dir) / "extraction" / "review_policy.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def extract_label_values(document_id: str, evidence: dict[str, Any], document_type: str) -> list[ExtractedField]:
    config_dir = Path(settings.config_dir) / "extraction"
    labels_doc = yaml.safe_load((config_dir / "labels.yaml").read_text(encoding="utf-8")) or {}
    policy = _review_policy()
    noise = yaml.safe_load((config_dir / "noise_patterns.yaml").read_text(encoding="utf-8")) or {}
    field_specs = labels_doc.get(document_type, {})
    registry = yaml.safe_load((Path(settings.config_dir) / "candidates" / "field_registry.yaml").read_text(encoding="utf-8")) or {}
    registry_fields = registry.get(document_type, {})
    all_labels = sorted({label for name, spec in field_specs.items() if not name.startswith("_") for label in spec.get("labels", [])} | set(field_specs.get("_stop_labels", [])), key=len, reverse=True)
    label_re = _label_regex(all_labels)
    noise_patterns = [re.compile(item["regex"], re.IGNORECASE) for item in noise.get("patterns", [])]
    score_cfg = policy.get("rule_scores", {})
    limits = policy.get("field_limits", {})
    threshold = float((policy.get("status") or {}).get("warning_threshold", 0.7))
    lines = [(page, line) for page in evidence.get("pages", []) for line in page.get("lines", [])]
    output: list[ExtractedField] = []

    for field, registry_spec in registry_fields.items():
        if registry_spec.get("cand_type") == "table":
            table_field = _table_candidate(document_id, evidence, document_type, field, policy)
            output.append(table_field or _unknown(document_id, field))
            continue
        spec = field_specs.get(field)
        if not spec:
            output.append(_unknown(document_id, field))
            continue
        hits = _field_candidates(lines, spec, label_re, all_labels, noise_patterns)
        if not hits:
            output.append(ExtractedField(
                field=field, value=None, confidence=0.0,
                source=EvidenceSource(document_id=document_id, page=1), method=ExtractionMethod.RULE_BASED,
                is_unknown=True, raw_text=None, review_status="UNKNOWN", source_kind="rule",
            ))
            continue
        raw, bbox, page, source_kind = hits[0]
        score = float(score_cfg.get(source_kind, score_cfg.get("rule_label_match", 0.8)))
        value: Any = raw
        if spec.get("kind") == "date":
            value = parse_date_string(raw)
            if value is None:
                output.append(_unknown(document_id, field))
                continue
        if not isinstance(value, (dict, list)):
            cap = int(limits.get("address" if spec.get("kind") == "address" else "scalar", 160))
            value = " ".join(str(value).split())[:cap]
        source = EvidenceSource(document_id=document_id, page=page, bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        status = "ACCEPTED" if score >= float((policy.get("status") or {}).get("accepted_threshold", 0.9)) else "REVIEW"
        output.append(ExtractedField(
            field=field, value=value, confidence=score, source=source, method=ExtractionMethod.RULE_BASED,
            is_unknown=False, raw_text=raw, review_status=status, low_confidence=score < threshold,
            confidence_source=source_kind, source_kind="rule", decoder_metadata={"rule_id": source_kind, "label_match": True},
        ))
    return output


def merge_review_fields(document_id: str, evidence: dict[str, Any], document_type: str, rule_fields: list[ExtractedField], ml: dict[str, Any]) -> list[ExtractedField]:
    """Combine registry values from label rules and ML without hiding suggestions."""
    selected = {field.field: field for field in extract_label_values(document_id, evidence, document_type)}
    # The label policy is an additive review aid, not the source of truth for
    # extraction. It intentionally returns UNKNOWN when a label is absent. In
    # that case the typed extractor has often already found a value from OCR
    # geometry/regex/table analysis. Previously the pre-created UNKNOWN entry
    # shadowed that real value, producing empty HITL mappings even though the
    # evidence lines and line-items table were present.
    for extracted in rule_fields:
        current = selected.get(extracted.field)
        if current is not None and current.is_unknown and extracted.value is not None:
            selected[extracted.field] = extracted
    ml_fields = {item.get("field"): item for item in ml.get("fields", []) if isinstance(item, dict) and item.get("field")}
    label_config = yaml.safe_load((Path(settings.config_dir) / "extraction" / "labels.yaml").read_text(encoding="utf-8")) or {}
    field_specs = label_config.get(document_type, {})
    labels = sorted({label for name, spec in field_specs.items() if not name.startswith("_") for label in spec.get("labels", [])}, key=len, reverse=True)
    registry = yaml.safe_load((Path(settings.config_dir) / "candidates" / "field_registry.yaml").read_text(encoding="utf-8")) or {}
    registry_specs = registry.get(document_type, {})
    for name, current in list(selected.items()):
        candidate = ml_fields.get(name)
        if not candidate or candidate.get("value") is None or not candidate.get("bbox_norm"):
            continue
        try:
            score = max(0.0, min(1.0, float(candidate.get("score", 0.0))))
            box = [float(value) for value in candidate["bbox_norm"]]
            page = int(candidate.get("page") or 1)
        except (TypeError, ValueError):
            continue
        kind = field_specs.get(name, {}).get("kind", registry_specs.get(name, {}).get("cand_type", "text"))
        if len(box) != 4 or not _sane(str(candidate["value"]), {"kind": kind}, labels):
            continue
        alternative = {"value": current.value, "raw_text": current.raw_text, "score": current.confidence, "source_kind": current.source_kind, "source": current.source.model_dump()}
        if current.is_unknown or score > current.confidence:
            status = "ACCEPTED" if score >= 0.9 else "REVIEW"
            selected[name] = ExtractedField(
                field=name, value=candidate["value"], confidence=score,
                source=EvidenceSource(document_id=document_id, page=page, bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])),
                method=ExtractionMethod.ML_MODEL, is_unknown=False, raw_text=str(candidate.get("raw_text") or candidate["value"]),
                confidence_source="ml_model_score", review_status=status, low_confidence=score < 0.7,
                source_kind="ml", alternatives=[alternative], decoder_metadata={"ml_status": candidate.get("status")},
            )
        else:
            current.alternatives = [{"value": candidate["value"], "raw_text": candidate.get("raw_text"), "score": score, "source_kind": "ml", "source": {"page": page, "bbox": box}}]

    # Preserve legacy fields for existing reconciliation consumers. The review
    # UI filters these using the generated registry metadata.
    for field in rule_fields:
        if field.field not in selected:
            score = max(0.0, min(1.0, float(field.confidence or 0.0)))
            field.raw_text = str(field.value) if field.value is not None else None
            field.review_status = "UNKNOWN" if field.value is None else ("ACCEPTED" if score >= 0.9 else "REVIEW")
            field.low_confidence = field.value is not None and score < 0.7
            field.source_kind = "rule"
            selected[field.field] = field

    # Registry order is stable and is also the order used by generated clients.
    return list(selected.values())


def _unknown(document_id: str, field: str) -> ExtractedField:
    return ExtractedField(field=field, confidence=0.0, source=EvidenceSource(document_id=document_id, page=1),
                          method=ExtractionMethod.RULE_BASED, is_unknown=True, review_status="UNKNOWN", source_kind="rule")


def _label_regex(labels: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(label).replace(r"\ ", r"\s+") for label in labels)
    return re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])(?:\s*[:：–-]\s*|\s+)?", re.IGNORECASE)


def _field_candidates(lines, spec, label_re, all_labels, noise_patterns):
    field_labels = {label.casefold() for label in spec.get("labels", [])}
    matches = []
    for page, line in lines:
        text = " ".join(str(line.get("text", "")).split())
        box = line.get("bbox_norm") or [0, 0, 0, 0]
        if len(box) != 4:
            continue
        for hit in label_re.finditer(text):
            label = hit.group(0).strip(" :：-–").casefold()
            if label not in field_labels:
                continue
            if spec.get("continuation_only"):
                value = ""
            else:
                end = label_re.search(text, hit.end())
                value = text[hit.end():end.start() if end else len(text)].strip(" :：-–\t")
            if value and any(pattern.search(value) for pattern in noise_patterns):
                value = ""
            value_box = list(box)
            if not value and not spec.get("continuation_only"):
                adjacent = _same_row_value(lines, page, box, label_re)
                if adjacent:
                    value, value_box = adjacent
            if value and _sane(value, spec, all_labels):
                matches.append((value, value_box, int(page.get("page_number", 1)), "rule_label_match"))
            if spec.get("continuation_only") or not value:
                continuation = _continuation(lines, page, box, label_re, noise_patterns, all_labels)
                if continuation:
                    ctext, cbbox, cpage = continuation
                    if _sane(ctext, spec, all_labels):
                        matches.append((ctext, cbbox, cpage, "rule_continuation"))
    return matches


def _same_row_value(lines, page, anchor_box, label_re):
    anchor_center = (float(anchor_box[1]) + float(anchor_box[3])) / 2
    anchor_height = float(anchor_box[3]) - float(anchor_box[1])
    candidates = []
    policy = _review_policy().get("label_value", {})
    for candidate_page, line in lines:
        if candidate_page is not page:
            continue
        box = line.get("bbox_norm") or []
        if len(box) != 4 or box[0] < float(anchor_box[2]) - float(policy.get("adjacent_value_x_margin", 0.01)):
            continue
        center = (float(box[1]) + float(box[3])) / 2
        if abs(center - anchor_center) > float(policy.get("same_row_center_tolerance", 0.6)) * max(anchor_height, float(box[3]) - float(box[1])):
            continue
        text = " ".join(str(line.get("text", "")).split())
        value = text[:label_re.search(text).start()] if label_re.search(text) else text
        value = value.strip(" :：-–\t")
        if value:
            candidates.append((float(box[0]), value, list(box)))
    if not candidates:
        return None
    _, value, box = min(candidates, key=lambda item: item[0])
    return value, box


def _continuation(lines, page, anchor_box, label_re, noise_patterns, labels):
    policy = _review_policy().get("label_value", {})
    page_number = int(page.get("page_number", 1))
    page_lines = [line for p, line in lines if int(p.get("page_number", 1)) == page_number]
    anchor_line = next((i for i, line in enumerate(page_lines) if line.get("bbox_norm") == anchor_box), None)
    if anchor_line is None:
        return None
    base_x = float(anchor_box[0])
    collected, boxes = [], []
    last_y = float(anchor_box[3])
    for line in page_lines[anchor_line + 1:]:
        text = " ".join(str(line.get("text", "")).split())
        box = line.get("bbox_norm") or [0, 0, 0, 0]
        if len(box) != 4:
            continue
        center = (float(box[1]) + float(box[3])) / 2
        anchor_center = (float(anchor_box[1]) + float(anchor_box[3])) / 2
        if abs(center - anchor_center) <= float(policy.get("same_row_center_tolerance", 0.6)) * max(float(box[3]) - float(box[1]), float(anchor_box[3]) - float(anchor_box[1])):
            continue
        if label_re.search(text):
            break
        if float(box[1]) - last_y > float(policy.get("continuation_max_vertical_gap", 0.045)) or abs(float(box[0]) - base_x) > float(policy.get("continuation_x_tolerance", 0.25)):
            break
        if any(pattern.search(text) for pattern in noise_patterns):
            continue
        if text:
            collected.append(text.strip(" :：-–\t"))
            boxes.append(box)
            last_y = float(box[3])
    if not collected:
        return None
    bbox = [min(float(box[0]) for box in boxes), min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes), max(float(box[3]) for box in boxes)]
    return " ".join(collected), bbox, page_number


def _sane(value: str, spec: dict, labels: list[str] | None = None) -> bool:
    value = " ".join(value.split()).strip(" :：-–")
    if not value or len(value) > (240 if spec.get("kind") == "address" else 160):
        return False
    if re.fullmatch(r"\(?\s*[.·_\-]{3,}\s*\)?", value):
        return False
    for label in labels or []:
        label_pattern = re.escape(label).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![\w]){label_pattern}(?![\w])", value, re.IGNORECASE):
            return False
    kind = spec.get("kind")
    if kind == "id" and not re.search(r"\d", value):
        return False
    if kind == "party" and not any(char.isalpha() for char in value):
        return False
    if kind == "address":
        try:
            markers = (_review_policy().get("sanity") or {}).get("address_markers", [])
        except Exception:
            markers = []
        if not re.search(r"\d", value) and not any(re.search(rf"(?<!\w){re.escape(str(marker))}(?!\w)", value, re.IGNORECASE) for marker in markers):
            return False
    return True


def _table_candidate(document_id: str, evidence: dict[str, Any], document_type: str, field: str, policy: dict) -> ExtractedField | None:
    aliases_path = Path(settings.config_dir) / "candidates" / "line_item_aliases.yaml"
    abbreviations_path = Path(settings.config_dir) / "candidates" / "abbreviations.yaml"
    try:
        from app.services.candidates.fuzzy_fields import FuzzyFieldMatcher
        matcher = FuzzyFieldMatcher(document_type, aliases_path, abbreviations_path)
    except Exception:
        matcher = None
    ranked = []
    for page in evidence.get("pages", []):
        tables = list(page.get("tables", []))
        if not tables:
            fallback = _text_table(page, matcher)
            if fallback:
                tables.append(fallback)
                page.setdefault("tables", []).append(fallback)
        for table in tables:
            headers = [str(value or "").strip() for value in table.get("headers", [])]
            rows = [[str(value or "").strip() for value in row] for row in table.get("rows", [])]
            rows = [row for row in rows if any(cell for cell in row) and not _summary_row(row)]
            if not headers or not rows:
                continue
            matches = [matcher.match(header).candidates if matcher else [] for header in headers]
            matched = sum(bool(candidates and candidates[0].score >= float((policy.get("tables") or {}).get("header_match_threshold", 0.55))) for candidates in matches)
            table_policy = _review_policy().get("tables", {})
            score = float(table_policy.get("header_weight", 0.65)) * matched / max(len(headers), 1) + float(table_policy.get("row_count_weight", 0.15)) * min(len(rows), 3) / 3 + float(table_policy.get("role_weight", 0.20)) * float(table.get("role") == "line_items")
            ranked.append((score, int(page.get("page_number", 1)), page, table, headers, rows, matches))
    for group in evidence.get("table_groups", []):
        headers = [str(value or "").strip() for value in group.get("headers", [])]
        rows = [[str(value or "").strip() for value in row] for row in group.get("rows", [])]
        rows = [row for row in rows if any(cell for cell in row) and not _summary_row(row)]
        if not headers or not rows:
            continue
        matches = [matcher.match(header).candidates if matcher else [] for header in headers]
        table_policy = _review_policy().get("tables", {})
        threshold = float(table_policy.get("header_match_threshold", 0.55))
        matched = sum(bool(candidates and candidates[0].score >= threshold) for candidates in matches)
        score = float(table_policy.get("header_weight", 0.65)) * matched / max(len(headers), 1) + float(table_policy.get("row_count_weight", 0.15)) * min(len(rows), 3) / 3 + float(table_policy.get("role_weight", 0.20))
        table_ids = set(group.get("table_ids", []))
        member_boxes = [table.get("bbox_norm") for page in evidence.get("pages", []) for table in page.get("tables", []) if table.get("table_id") in table_ids and table.get("bbox_norm")]
        bbox = [min(box[0] for box in member_boxes), min(box[1] for box in member_boxes), max(box[2] for box in member_boxes), max(box[3] for box in member_boxes)] if member_boxes else [0, 0, 1, 1]
        table = {"table_id": group.get("group_id"), "role": "line_items", "source": "docling_table_group", "headers": headers, "rows": rows,
                 "pages": group.get("pages", []), "bbox_norm": bbox, "selection_score": score}
        page_number = min(group.get("pages", [1]))
        page = next((item for item in evidence.get("pages", []) if item.get("page_number") == page_number), {"page_number": page_number})
        ranked.append((score, page_number, page, table, headers, rows, matches))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1], str(item[3].get("table_id", ""))))
    score, page_number, page, table, headers, rows, matches = ranked[0]
    max_rows = int((policy.get("tables") or {}).get("max_rows", 500))
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    columns, canonical_rows = [], []
    for index, header in enumerate(headers):
        match = matches[index][0] if matches[index] else None
        columns.append({"index": index, "header": header, "field": match.field if match else None, "match_score": round(match.score, 4) if match else 0.0})
    for index, cells in enumerate(rows, 1):
        cells = (cells + [""] * len(headers))[:len(headers)]
        values = {column["field"]: cells[column["index"]] for column in columns if column["field"]}
        canonical_rows.append({"index": index, "cells": cells, "values": values, "is_summary": False})
    bbox = table.get("bbox_norm") or [0, 0, 1, 1]
    payload = {"kind": "table", "role": table.get("role") or "line_items", "table_id": table.get("table_id"),
               "pages": table.get("pages") or [page_number], "source": table.get("source", "docling"), "bbox_norm": bbox,
               "selection": {"rank": 1, "score": round(score, 4), "features": ["header_alias_match"] if score else []},
               "headers": headers, "columns": columns, "rows": canonical_rows, "summary_rows": [], "truncated": truncated,
               "confidence_source": "table_structural_score"}
    summary = f"Tabel p{page_number} · {len(canonical_rows)} baris × {len(headers)} kolom"
    return ExtractedField(field=field, value=payload, confidence=round(score, 4),
                          source=EvidenceSource(document_id=document_id, page=page_number,
                                                bbox=BoundingBox(x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]))),
                          method=ExtractionMethod.LAYOUT, is_unknown=False, raw_text=summary,
                          confidence_source="table_structural_score", review_status="ACCEPTED" if score >= 0.9 else "REVIEW", low_confidence=score < 0.7,
                          source_kind="table", decoder_metadata={"table_id": table.get("table_id"), "headers": headers})


def _summary_row(row: list[str]) -> bool:
    joined = " ".join(row).casefold()
    return any(word in joined for word in ("subtotal", "grand total", "total amount", "total barang"))


def _text_table(page: dict, matcher) -> dict | None:
    """Recover a simple whitespace-aligned table from positioned vector-text rows."""
    visual_rows: list[list[dict]] = []
    for line in sorted(page.get("lines", []), key=lambda item: (item["bbox_norm"][1], item["bbox_norm"][0])):
        box = line.get("bbox_norm") or []
        if len(box) != 4 or not str(line.get("text", "")).strip():
            continue
        center = (float(box[1]) + float(box[3])) / 2
        row_tolerance = float(_review_policy().get("line_construction", {}).get("table_row_y_tolerance", 0.006))
        row = next((r for r in visual_rows if abs(center - sum((float(x["bbox_norm"][1]) + float(x["bbox_norm"][3])) / 2 for x in r) / len(r)) <= row_tolerance), None)
        if row is None:
            visual_rows.append([line])
        else:
            row.append(line)
    visual_rows = [sorted(row, key=lambda item: item["bbox_norm"][0]) for row in visual_rows]

    header_index, header_matches = None, None
    for index, row in enumerate(visual_rows):
        texts = [str(line["text"]).strip() for line in row]
        matches = [matcher.match(text).candidates if matcher else [] for text in texts]
        matched = sum(bool(candidates and candidates[0].score >= 0.55) for candidates in matches)
        if matched >= 2:
            header_index, header_matches = index, matches
            break
    if header_index is None:
        return None
    header_row = visual_rows[header_index]
    headers = [str(line["text"]).strip() for line in header_row]
    centers = [(float(line["bbox_norm"][0]) + float(line["bbox_norm"][2])) / 2 for line in header_row]
    data_rows = []
    last_y = max(float(line["bbox_norm"][3]) for line in header_row)
    source_boxes = [line["bbox_norm"] for line in header_row]
    for row in visual_rows[header_index + 1:]:
        top = min(float(line["bbox_norm"][1]) for line in row)
        if top - last_y > 0.055:
            break
        cells = [""] * len(headers)
        for line in row:
            text = str(line["text"]).strip()
            if not text or _summary_row([text]):
                continue
            center = (float(line["bbox_norm"][0]) + float(line["bbox_norm"][2])) / 2
            index = min(range(len(centers)), key=lambda i: abs(center - centers[i]))
            cells[index] = (cells[index] + " " + text).strip()
            source_boxes.append(line["bbox_norm"])
        if any(cells):
            data_rows.append(cells)
            last_y = max(float(line["bbox_norm"][3]) for line in row)
    if not data_rows:
        return None
    bbox = [min(float(box[0]) for box in source_boxes), min(float(box[1]) for box in source_boxes),
            max(float(box[2]) for box in source_boxes), max(float(box[3]) for box in source_boxes)]
    return {"table_id": f"page-{page.get('page_number', 1)}-text-table-1", "role": "line_items", "role_reason": "header_alias_match",
            "source": "pymupdf_positioned_text", "bbox_norm": bbox, "headers": headers, "rows": data_rows,
            "row_count": len(data_rows), "col_count": len(headers), "summary_rows": [], "confidence": float(_review_policy().get("tables", {}).get("text_table_confidence", 0.65))}
