from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from .context import build_page, context_for
from .parsers import parse_date, parse_number, parse_time
from .registry import FieldRegistry
from .schema import Candidate, Page, Token
from .text_norm import norm_alnum, norm_text


class CandidatePipeline:
    """Generate literal OCR spans and deterministic, label-ready feature rows."""

    def __init__(self, config_dir: str | Path):
        config_dir = Path(config_dir)
        self.registry = FieldRegistry.load(config_dir / "field_registry.yaml")
        self.caps = {"amount": 80, "date": 30, "time": 20, "id": 60, "party": 40, "enum": 20, "qty": 40, "weight": 30, "percent": 20, "days": 20, "text": 25}
        generation = config_dir / "candidate_generation.yaml"
        if generation.exists():
            import yaml
            self.caps.update((yaml.safe_load(generation.read_text(encoding="utf-8")) or {}).get("caps", {}))

    def generate(self, document_id: str, doc_type: str, ocr_result: dict[str, Any], run_id: str = "") -> list[dict[str, Any]]:
        enabled = {field.cand_type for field in self.registry.for_doc(doc_type)}
        pages = [build_page(page) for page in ocr_result.get("pages", [])]
        candidates: list[Candidate] = []
        for page in pages:
            for cand_type in sorted(enabled):
                candidates.extend(self._generate_type(page, cand_type))
        candidates.sort(key=lambda candidate: (candidate.page.page_number, candidate.bbox[1], candidate.bbox[0], candidate.cand_type, candidate.raw_text))
        type_counts: dict[str, int] = {}
        for candidate in candidates:
            candidate.reading_idx = type_counts.get(candidate.cand_type, 0) + 1
            type_counts[candidate.cand_type] = candidate.reading_idx
            candidate.context = context_for(candidate)
            candidate.features = self._features(candidate, candidates)
        return [self._row(document_id, doc_type, run_id, candidate) for candidate in candidates]

    def _generate_type(self, page: Page, cand_type: str) -> list[Candidate]:
        if cand_type == "amount":
            return self._number_candidates(page, cand_type, amount=True)
        if cand_type == "date":
            return self._date_candidates(page)
        if cand_type == "time":
            return self._time_candidates(page)
        if cand_type == "id":
            return self._id_candidates(page)
        if cand_type == "party":
            return self._party_candidates(page)
        if cand_type == "enum":
            return [self._candidate("enum", page, [token], "IDR" if norm_text(token.text) in {"rp", "rp.", "idr", "rupiah"} else token.text.upper()) for token in page.tokens if norm_text(token.text) in {"rp", "rp.", "idr", "usd", "sgd", "eur", "$", "rupiah"}]
        if cand_type == "percent":
            return self._percent_candidates(page)
        if cand_type == "days":
            return self._days_candidates(page)
        return []

    def _number_candidates(self, page: Page, cand_type: str, amount: bool) -> list[Candidate]:
        result = []
        for token in page.tokens:
            parsed = parse_number(token.text)
            if parsed.value is None:
                continue
            text = norm_text(token.text)
            if amount:
                if re.search(r"%$", token.text) or re.fullmatch(r"\d{4}", text) or (parsed.value < 1000 and not parsed.sym and not any(word in text for word in (".", ","))):
                    continue
                if re.fullmatch(r"(?:\+?62|0)8\d{8,12}", text):
                    continue
            result.append(self._candidate(cand_type, page, [token], str(parsed.value), norm_alnum(token.text)))
        return self._dedupe(result)

    def _date_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for start, token in enumerate(page.tokens):
            for end in range(min(len(page.tokens), start + 4), start, -1):
                raw = " ".join(item.text for item in page.tokens[start:end])
                parsed = parse_date(raw)
                if parsed.iso:
                    result.append(self._candidate("date", page, page.tokens[start:end], parsed.iso, parsed.iso))
                    break
        return self._dedupe(result)

    def _time_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for token in page.tokens:
            parsed = parse_time(token.text)
            if parsed:
                result.append(self._candidate("time", page, [token], parsed["value"], parsed["value"]))
        return result

    def _percent_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for token in page.tokens:
            match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*%", token.text.strip())
            if match:
                result.append(self._candidate("percent", page, [token], match.group(1).replace(",", ".")))
        return self._dedupe(result)

    def _days_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for index, token in enumerate(page.tokens):
            if not re.fullmatch(r"\d{1,3}", token.text.strip()):
                continue
            next_token = page.tokens[index + 1] if index + 1 < len(page.tokens) else None
            if next_token and norm_text(next_token.text) in {"day", "days", "hari"}:
                result.append(self._candidate("days", page, [token, next_token], token.text.strip()))
        return self._dedupe(result)

    def _id_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for token in page.tokens:
            raw = token.text.strip("[]")
            if len(raw) < 4 or len(raw) > 32 or not re.search(r"\d", raw) or (raw.isdigit() and len(raw) < 5) or parse_date(raw).iso:
                continue
            if re.fullmatch(r"[\d.,]+", raw):
                continue
            result.append(self._candidate("id", page, [token], raw, norm_alnum(raw)))
        return self._dedupe(result)

    def _party_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for line in page.lines:
            raw = " ".join(token.text for token in line)
            words = raw.split()
            letters = sum(char.isalpha() for char in raw)
            if len(words) < 1 or letters < 3 or letters / max(len(raw), 1) < .5 or re.fullmatch(r"invoice|faktur|surat jalan|proof of delivery", norm_text(raw)):
                continue
            start = 0
            if ":" in raw:
                colon = next((index for index, token in enumerate(line) if ":" in token.text), None)
                if colon is not None and colon + 1 < len(line):
                    start = colon + 1
            value_tokens = line[start:]
            value = " ".join(token.text for token in value_tokens)
            if sum(char.isalpha() for char in value) / max(len(value), 1) >= .5:
                result.append(self._candidate("party", page, value_tokens, value, norm_text(value)))
        return self._dedupe(result)

    def _candidate(self, cand_type: str, page: Page, tokens: list[Token], parsed: str, normalized: str = "") -> Candidate:
        return Candidate(cand_type, page, tokens, " ".join(token.text for token in tokens), parsed, normalized or norm_text(parsed))

    def _dedupe(self, candidates: list[Candidate]) -> list[Candidate]:
        seen: set[tuple[str, int, tuple[float, ...]]] = set()
        result = []
        for candidate in candidates:
            key = (candidate.cand_type, candidate.page.page_number, candidate.bbox)
            if key not in seen:
                seen.add(key)
                result.append(candidate)
        return result

    def _features(self, candidate: Candidate, all_candidates: list[Candidate]) -> dict[str, Any]:
        x1, y1, x2, y2 = candidate.bbox
        same = [item for item in all_candidates if item.cand_type == candidate.cand_type]
        values = [item for item in same if item.parsed_norm == candidate.parsed_norm]
        features: dict[str, Any] = {"g_x1": round(x1, 4), "g_y1": round(y1, 4), "g_x2": round(x2, 4), "g_y2": round(y2, 4), "g_cx": round((x1 + x2) / 2, 4), "g_cy": round((y1 + y2) / 2, 4), "g_w": round(x2 - x1, 4), "g_h": round(y2 - y1, 4), "q_conf_mean": round(sum(token.confidence for token in candidate.tokens) / len(candidate.tokens), 6), "q_box_estimated": int(any(token.box_estimated for token in candidate.tokens)), "q_is_vector": int(candidate.page.engine_used == "pymupdf_vector"), "d_n_same_type_doc": len(same), "d_n_same_type_page": sum(item.page.page_number == candidate.page.page_number for item in same), "d_dup_count_doc": len(values) - 1, "d_is_first_occurrence": int(values and values[0] is candidate), "d_rank_reading": candidate.reading_idx, "d_rank_reading_rel": (candidate.reading_idx - 1) / max(len(same) - 1, 1)}
        if candidate.cand_type == "amount":
            parsed = Decimal(candidate.parsed_value)
            features.update({"a_value": float(parsed), "a_log10": 0 if parsed == 0 else float(abs(parsed).ln() / Decimal(10).ln()), "a_ambiguous": int("." in candidate.raw_text or "," in candidate.raw_text), "a_is_max": int(parsed == max(Decimal(item.parsed_value) for item in same)), "a_rank_desc_doc": sorted((Decimal(item.parsed_value) for item in same), reverse=True).index(parsed) + 1})
        return features

    def _row(self, document_id: str, doc_type: str, run_id: str, candidate: Candidate) -> dict[str, Any]:
        x1, y1, x2, y2 = candidate.bbox
        candidate_id = hashlib.sha1(f"{document_id}|{candidate.page.page_number}|{candidate.cand_type}|{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}|{candidate.raw_text}".encode()).hexdigest()[:16]
        row = {"m_schema_version": "1.0.0", "m_run_id": run_id, "m_candidate_id": candidate_id, "m_doc_type": doc_type, "m_document_id": document_id, "m_vendor_group": "UNKNOWN", "m_template_group": "UNKNOWN", "m_page": candidate.page.page_number, "m_cand_type": candidate.cand_type, "m_raw_text": candidate.raw_text, "m_parsed_value": candidate.parsed_value, "m_parsed_norm": candidate.parsed_norm, "m_x1": round(x1, 4), "m_y1": round(y1, 4), "m_x2": round(x2, 4), "m_y2": round(y2, 4), "m_line_idx": candidate.line_idx, "m_tok_start": candidate.tokens[0].tok_idx, "m_tok_end": candidate.tokens[-1].tok_idx, "m_overlap_types": "", **candidate.features, **candidate.context}
        for field in self.registry.fields_for_type(doc_type, candidate.cand_type):
            row[f"label__{field}"] = ""
        row.update({"label_status": "", "label_notes": ""})
        return row
