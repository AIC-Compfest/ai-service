from __future__ import annotations

import copy
import re

import hashlib
import re
import statistics
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .context import build_page, context_for
from app.services.evidence.docling_tables import normalize_docling_evidence
from .parsers import parse_date, parse_number, parse_time
from .registry import FieldRegistry
from .schema import Candidate, Page, Token
from .text_norm import norm_alnum, norm_text, ocr_fold
from .fuzzy_fields import _window_similarity
from .version import candidate_pipeline_version


class CandidatePipeline:
    """Generate literal OCR spans and deterministic, label-ready feature rows."""

    def __init__(self, config_dir: str | Path):
        config_dir = Path(config_dir)
        self.config_dir = config_dir
        self.pipeline_version = candidate_pipeline_version(config_dir)
        lexicon_path = config_dir / "lexicon.yaml"
        self.lexicon = yaml.safe_load(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.exists() else {}
        self.registry = FieldRegistry.load(config_dir / "field_registry.yaml")
        self.registry.validate()
        self.caps = {"amount": 80, "date": 30, "time": 20, "id": 60, "party": 40, "enum": 20, "qty": 40, "weight": 30, "percent": 20, "days": 20, "text": 25}
        self.last_report: dict[str, Any] = {"empty_pages": [], "cap_overflows": [], "timings_ms": {}}
        generation = config_dir / "candidate_generation.yaml"
        if generation.exists():
            self.caps.update((yaml.safe_load(generation.read_text(encoding="utf-8")) or {}).get("caps", {}))
        self.last_run_report: dict[str, Any] = {"cap_overflow": {}}

    def generate(self, document_id: str, doc_type: str, ocr_result: dict[str, Any], run_id: str = "", vendor_group: str = "UNKNOWN", template_group: str = "UNKNOWN", include_tables: bool = True) -> list[dict[str, Any]]:
        started = time.perf_counter()
        self.last_report = {"empty_pages": [], "cap_overflows": [], "timings_ms": {}}
        enabled = {field.cand_type for field in self.registry.for_doc(doc_type)}
        evidence, table_warnings = normalize_docling_evidence(ocr_result)
        pages = [build_page(page) for page in evidence.get("pages", [])]
        table_rows = _table_candidate_rows(self, document_id, doc_type, run_id, vendor_group, template_group, evidence) if include_tables else []
        self.last_report["tables"] = [
            {key: value for key, value in row.items() if key not in {"m_candidate_id"}}
            for row in table_rows
        ]
        if table_warnings:
            self.last_report["table_warnings"] = table_warnings
        candidates: list[Candidate] = []
        for page in pages:
            if not page.lines:
                self.last_report["empty_pages"].append({"document_id": document_id, "page": page.page_number})
            for cand_type in sorted(enabled):
                generated = self._generate_type(page, cand_type)
                candidates.extend(generated)
        candidates.sort(key=lambda candidate: (candidate.page.page_number, candidate.bbox[1], candidate.bbox[0], candidate.cand_type, candidate.raw_text))
        candidates = self._apply_caps(candidates)
        type_counts: dict[str, int] = {}
        for candidate in candidates:
            candidate.reading_idx = type_counts.get(candidate.cand_type, 0) + 1
            type_counts[candidate.cand_type] = candidate.reading_idx
            candidate.context = context_for(candidate)
            candidate.features = self._features(candidate, candidates)
            overlap_types = {candidate.cand_type}
            overlap_types.update(other.cand_type for other in candidates if other is not candidate and _overlaps(candidate.bbox, other.bbox))
            candidate.features["m_overlap_types"] = "|".join(sorted(overlap_types))
        self.last_report["timings_ms"][f"{document_id}:{doc_type}"] = round((time.perf_counter() - started) * 1000, 3)
        return table_rows + [self._row(document_id, doc_type, run_id, vendor_group, template_group, candidate) for candidate in candidates]
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
            return self._enum_candidates(page)
        if cand_type == "qty":
            return self._unit_number_candidates(page, "qty", {"dus", "koli", "pcs", "box", "colly", "ctn", "paket", "unit"}, 99999)
        if cand_type == "weight":
            return self._unit_number_candidates(page, "weight", {"kg", "kgs", "kilogram", "g", "gram", "ton"}, 50000)
        if cand_type == "percent":
            return self._percent_candidates(page)
        if cand_type == "days":
            return self._unit_number_candidates(page, "days", {"hari", "days", "working", "business"}, 365)
        if cand_type == "city":
            return self._city_candidates(page)
        if cand_type == "address":
            return self._address_candidates(page)
        if cand_type == "text":
            return self._text_candidates(page)
        return []

    def _enum_candidates(self, page: Page) -> list[Candidate]:
        values = {"rp": "IDR", "rp.": "IDR", "idr": "IDR", "rupiah": "IDR", "usd": "USD", "sgd": "SGD", "eur": "EUR", "$": "USD", "baik": "GOOD", "good": "GOOD", "ok": "GOOD", "lengkap": "GOOD", "complete": "GOOD", "rusak": "DAMAGED", "damaged": "DAMAGED", "pecah": "DAMAGED", "penyok": "DAMAGED", "kurang": "SHORTAGE", "shortage": "SHORTAGE"}
        return [self._candidate("enum", page, [token], values[norm_text(token.text)]) for token in page.tokens if norm_text(token.text) in values]

    def _unit_number_candidates(self, page: Page, cand_type: str, units: set[str], max_value: int | float) -> list[Candidate]:
        result = []
        for index, token in enumerate(page.tokens):
            parsed = parse_number(token.text.rstrip("%"))
            if parsed.value is None or parsed.value <= 0 or parsed.value > max_value:
                continue
            right = norm_text(page.tokens[index + 1].text) if index + 1 < len(page.tokens) else ""
            left = norm_text(page.tokens[index - 1].text) if index else ""
            attached = next((unit for unit in units if token.text.lower().endswith(unit) and token.text.lower() != unit), None)
            unit = attached or (right if right in units else left if left in units else None)
            if cand_type == "qty" and parsed.value > 99999:
                continue
            if cand_type == "weight" and not unit:
                continue
            result.append(self._candidate(cand_type, page, [token], str(parsed.value), norm_alnum(token.text)))
        return self._dedupe(result)

    def _percent_candidates(self, page: Page) -> list[Candidate]:
        result = []
        for index, token in enumerate(page.tokens):
            raw = token.text.rstrip("%")
            parsed = parse_number(raw)
            next_text = norm_text(page.tokens[index + 1].text) if index + 1 < len(page.tokens) else ""
            if parsed.value is None or not (token.text.endswith("%") or next_text == "persen"):
                continue
            result.append(self._candidate("percent", page, [token], str(parsed.value), norm_alnum(raw)))
        return self._dedupe(result)

    def _city_candidates(self, page: Page) -> list[Candidate]:
        gazetteer = self._gazetteer()
        result = []
        for token in page.tokens:
            key = norm_text(token.text).strip(".,")
            if key in gazetteer:
                result.append(self._candidate("city", page, [token], gazetteer[key], norm_text(gazetteer[key])))
        return self._dedupe(result)

    def _address_candidates(self, page: Page) -> list[Candidate]:
        result = []
        markers = {"jl", "jalan", "jln", "street", "rt", "rw", "kel", "kelurahan", "kec", "kecamatan", "kota", "kabupaten"}
        for line in page.lines:
            text = norm_text(" ".join(token.text for token in line))
            if any(marker in text.split() for marker in markers) or re.search(r"\b\d{5}\b", text):
                result.append(self._candidate("address", page, line, " ".join(token.text for token in line), text))
        return self._dedupe(result)

    def _text_candidates(self, page: Page) -> list[Candidate]:
        result = []
        labels = {"description", "deskripsi", "uraian", "commodity", "isi", "keterangan barang", "nama barang"}
        for line in page.lines:
            tokens = line
            raw = " ".join(token.text for token in tokens)
            normalized = norm_text(raw)
            split = next((index for index, token in enumerate(tokens) if norm_text(token.text).rstrip(":") in labels), None)
            if split is not None and split + 1 < len(tokens):
                value_tokens = tokens[split + 1:]
                result.append(self._candidate("text", page, value_tokens, " ".join(token.text for token in value_tokens), norm_text(" ".join(token.text for token in value_tokens))))
            elif normalized in labels:
                continue
        return self._dedupe(result)

    def _gazetteer(self) -> dict[str, str]:
        path = self.config_dir / "gazetteer_cities.csv"
        if not path.exists():
            return {}
        import csv
        with path.open(encoding="utf-8", newline="") as handle:
            return {norm_text(row["alias"]): row["canonical"] for row in csv.DictReader(handle)}

    def _number_candidates(self, page: Page, cand_type: str, amount: bool) -> list[Candidate]:
        result = []
        for token in page.tokens:
            parsed = parse_number(token.text)
            if parsed.value is None:
                continue
            text = norm_text(token.text)
            if amount:
                digits = sum(char.isdigit() for char in token.text)
                phone_pattern = re.fullmatch(r"(?:\+?62|0)8\d{8,12}", re.sub(r"[\s().-]", "", token.text))
                # Unformatted 9+ digit strings are phone/account/reference-like in this
                # corpus. Manually labeled true amounts use separators or currency markers.
                if phone_pattern or (digits >= 9 and not parsed.sym and not re.search(r"[.,]", token.text)):
                    continue
                if re.search(r"%$", token.text) or re.fullmatch(r"\d{4}", text) or (parsed.value < 1000 and not parsed.sym and not any(word in text for word in (".", ","))):
                    continue
            result.append(self._candidate(cand_type, page, [token], str(parsed.value), norm_alnum(token.text)))
        if amount:
            # OCR often splits grouped thousands into individual tokens ("25", ",000", ",000").
            # Generate only contiguous visual-line spans, retaining exact evidence provenance.
            for line in page.lines:
                for start in range(len(line)):
                    for end in range(start + 2, min(len(line), start + 4) + 1):
                        tokens = line[start:end]
                        if not self._reasonable_token_span(tokens):
                            continue
                        raw = " ".join(token.text for token in tokens)
                        numeric = re.sub(r"^(?:rp\.?|idr|usd|\$)\s*", "", raw, flags=re.I)
                        # A currency token plus an already intact amount is covered by the
                        # legacy single-token candidate; this generator is for fragmented
                        # numeric groups only.
                        if sum(bool(re.search(r"\d", token.text)) for token in tokens) < 2:
                            continue
                        if not re.fullmatch(r"\d{1,3}(?:\s*[.,]\s?\d{3})+", numeric):
                            continue
                        parsed = parse_number(raw)
                        if parsed.value is not None:
                            result.append(self._candidate(cand_type, page, tokens, str(parsed.value), norm_alnum(raw)))
        return self._dedupe(result)

    @staticmethod
    def _reasonable_token_span(tokens: list[Token]) -> bool:
        return all(next_token.bbox[0] - token.bbox[2] <= max(.03, 2.5 * (token.bbox[2] - token.bbox[0]))
                   for token, next_token in zip(tokens, tokens[1:]))

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
            if re.fullmatch(r"[\d.,]+", raw) and not re.fullmatch(r"\d{5,20}", raw):
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
        chosen: dict[tuple[str, int, tuple[float, ...], str], Candidate] = {}
        for candidate in candidates:
            key = (candidate.cand_type, candidate.page.page_number, candidate.bbox, candidate.parsed_value)
            old = chosen.get(key)
            if old is None or len(candidate.tokens) > len(old.tokens):
                chosen[key] = candidate
        return list(chosen.values())

    def _apply_caps(self, candidates: list[Candidate]) -> list[Candidate]:
        """Apply configured caps deterministically, retaining an auditable overflow count."""
        grouped: dict[tuple[int, str], list[Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault((candidate.page.page_number, candidate.cand_type), []).append(candidate)
        kept: list[Candidate] = []
        overflow: dict[str, dict[str, int]] = {}
        for (page, cand_type), group in sorted(grouped.items()):
            ranked = sorted(group, key=lambda item: (-(_party_cap_priority(item) if cand_type == "party" else 0), -sum(t.confidence for t in item.tokens) / len(item.tokens), item.bbox[1], item.bbox[0], item.raw_text))
            cap = int(self.caps.get(cand_type, len(ranked)))
            kept.extend(ranked[:cap])
            if len(ranked) > cap:
                overflow.setdefault(str(page), {})[cand_type] = len(ranked) - cap
        self.last_run_report = {"cap_overflow": overflow}
        for page, types in overflow.items():
            for cand_type, dropped in types.items():
                self.last_report["cap_overflows"].append({"document_id": "", "page": int(page), "type": cand_type, "kept": int(self.caps.get(cand_type, 0)), "dropped": dropped})
        return kept

    def _features(self, candidate: Candidate, all_candidates: list[Candidate]) -> dict[str, Any]:
        x1, y1, x2, y2 = candidate.bbox
        same = [item for item in all_candidates if item.cand_type == candidate.cand_type]
        values = [item for item in same if item.parsed_norm == candidate.parsed_norm]
        features: dict[str, Any] = {"g_x1": round(x1, 4), "g_y1": round(y1, 4), "g_x2": round(x2, 4), "g_y2": round(y2, 4), "g_cx": round((x1 + x2) / 2, 4), "g_cy": round((y1 + y2) / 2, 4), "g_w": round(x2 - x1, 4), "g_h": round(y2 - y1, 4), "q_conf_mean": round(sum(token.confidence for token in candidate.tokens) / len(candidate.tokens), 6), "q_box_estimated": int(any(token.box_estimated for token in candidate.tokens)), "q_is_vector": int(candidate.page.engine_used == "pymupdf_vector"), "d_n_same_type_doc": len(same), "d_n_same_type_page": sum(item.page.page_number == candidate.page.page_number for item in same), "d_dup_count_doc": len(values) - 1, "d_is_first_occurrence": int(values and values[0] is candidate), "d_rank_reading": candidate.reading_idx, "d_rank_reading_rel": (candidate.reading_idx - 1) / max(len(same) - 1, 1)}
        features.update(self._lexical_features(candidate))
        if candidate.cand_type == "amount":
            parsed = Decimal(candidate.parsed_value)
            features.update({"a_value": float(parsed), "a_log10": 0 if parsed == 0 else float(abs(parsed).ln() / Decimal(10).ln()), "a_ambiguous": int("." in candidate.raw_text or "," in candidate.raw_text), "a_n_tokens": len(candidate.tokens), "a_has_currency_sym": int(bool(re.search(r"(?:^|\s)(?:rp\.?|idr|usd|\$)(?:\s|$)", candidate.raw_text, re.I))), "a_is_max": int(parsed == max(Decimal(item.parsed_value) for item in same)), "a_rank_desc_doc": sorted((Decimal(item.parsed_value) for item in same), reverse=True).index(parsed) + 1})
        elif candidate.cand_type == "date":
            parsed = parse_date(candidate.raw_text)
            if parsed.iso:
                from datetime import date
                value = date.fromisoformat(parsed.iso)
                dates = sorted(date.fromisoformat(item.parsed_value) for item in same)
                features.update({"dt_fmt": parsed.fmt, "dt_month_lang": parsed.month_lang, "dt_month_fuzzy": int(parsed.month_fuzzy), "dt_two_digit_year": int(parsed.two_digit_year), "dt_ambiguous_dm": int(parsed.ambiguous_dm), "dt_year": value.year, "dt_month": value.month, "dt_day": value.day, "dt_weekday": value.weekday(), "dt_is_month_start": int(value.day == 1), "dt_is_month_end": int(value.day == (date(value.year, value.month + 1, 1) - value).days if value.month < 12 else value.day == 31), "dt_rank_asc": dates.index(value) + 1, "dt_rank_desc": sorted(dates, reverse=True).index(value) + 1, "dt_is_min": int(value == min(dates)), "dt_is_max": int(value == max(dates)), "dt_days_from_min": (value - min(dates)).days, "dt_days_to_max": (max(dates) - value).days, "dt_n_dates_doc": len(dates)})
        elif candidate.cand_type == "time":
            parsed = parse_time(candidate.raw_text)
            if parsed:
                hour, minute = map(int, parsed["value"].split(":")[:2])
                features.update({"tm_hour": hour, "tm_minute": minute, "tm_has_seconds": int(parsed["value"].count(":") == 2), "tm_has_ampm": int(bool(re.search(r"\b(?:am|pm)\b", candidate.raw_text, re.I))), "tm_tz": parsed["tz"], "tm_sep": parsed["sep"], "tm_business_hours": int(7 <= hour <= 20), "tm_n_times_doc": len(same)})
        elif candidate.cand_type == "id":
            raw = candidate.raw_text.strip("[]")
            letters = [char for char in raw if char.isalpha()]
            features.update({"id_len": len(raw), "id_n_digits": sum(char.isdigit() for char in raw), "id_n_alpha": len(letters), "id_n_sep": sum(raw.count(char) for char in "- /._"), "id_digit_ratio": sum(char.isdigit() for char in raw) / max(len(raw), 1), "id_upper_ratio": sum(char.isupper() for char in letters) / len(letters) if letters else "", "id_has_dash": int("-" in raw), "id_has_slash": int("/" in raw), "id_has_dot": int("." in raw), "id_has_underscore": int("_" in raw), "id_starts_alpha": int(bool(raw) and raw[0].isalpha()), "id_prefix_alpha": re.match(r"[A-Za-z]{1,6}", raw).group(0).lower() if re.match(r"[A-Za-z]{1,6}", raw) else "", "id_pure_digits": int(raw.isdigit()), "id_pure_digits_len": len(raw) if raw.isdigit() else "", "id_has_year_segment": int(bool(re.search(r"(?:^|[-/])20\d\d(?:$|[-/])", raw))), "id_looks_like_phone": int(bool(re.fullmatch(r"(?:\+?62|0)8\d{8,12}", raw))), "id_in_brackets": int(candidate.raw_text != raw), "id_is_merged": int(len(candidate.tokens) > 1), "id_shape_compact": _shape_compact(raw)})
        elif candidate.cand_type in {"qty", "weight", "percent", "days"}:
            parsed = parse_number(candidate.raw_text.rstrip("%"))
            if parsed.value is not None:
                value = float(parsed.value)
                prefix = {"qty": "qt", "weight": "wt", "percent": "pc", "days": "dy"}[candidate.cand_type]
                features.update({f"{prefix}_value": value, f"{prefix}_plausible": int(0 < value <= (365 if candidate.cand_type == "days" else 50000)), f"{prefix}_n_doc": len(same)})
                if candidate.cand_type == "weight":
                    features.update({"wt_value_alt": float(parsed.value_alt) if parsed.value_alt is not None else "", "wt_ambiguous": int(parsed.ambiguous), "wt_unit": _near_unit(candidate.page, candidate.tokens[0], {"kg", "kgs", "kilogram", "g", "gram", "ton"}), "wt_unit_attached": int(bool(re.search(r"\d(?:kg|kgs|g|gram|ton)$", candidate.raw_text, re.I)))})
                elif candidate.cand_type == "percent":
                    features.update({"pc_decimals": max(len(candidate.raw_text.rstrip("%").split(".")[-1]), 0) if "." in candidate.raw_text else 0, "pc_sign_attached": int(candidate.raw_text.endswith("%")), "pc_word_form": int("persen" in candidate.context.get("ctx_right", "").lower())})
                elif candidate.cand_type == "days":
                    features.update({"dy_unit": _near_unit(candidate.page, candidate.tokens[0], {"hari", "days", "working", "business"}), "dy_plausible": int(0 < value <= 365)})
                else:
                    features.update({"qt_log": 0 if value <= 0 else __import__("math").log10(value), "qt_is_small": int(value <= 10), "qt_has_unit": int(bool(_near_unit(candidate.page, candidate.tokens[0], {"dus", "koli", "pcs", "box", "colly", "ctn", "paket", "unit"})))})
        elif candidate.cand_type == "party":
            raw = candidate.raw_text
            letters = sum(char.isalpha() for char in raw)
            words = raw.split()
            legal = next((marker for marker in ("pt", "cv", "ud", "pd", "tbk", "koperasi", "yayasan") if marker in norm_text(raw).split()), "none")
            features.update({"pt_n_tokens": len(words), "pt_n_chars": len(raw), "pt_upper_ratio": sum(char.isupper() for char in raw) / max(letters, 1), "pt_title_ratio": sum(bool(word) and word[0].isupper() for word in words) / max(len(words), 1), "pt_digit_ratio": sum(char.isdigit() for char in raw) / max(len(raw.replace(" ", "")), 1), "pt_alpha_ratio": letters / max(len(raw.replace(" ", "")), 1), "pt_legal_marker": legal, "pt_legal_pos": "prefix" if legal != "none" and norm_text(raw).split()[0] == legal else "suffix" if legal != "none" else "none", "pt_has_email": int("@" in raw), "pt_has_url": int(bool(re.search(r"https?://|www\.", raw, re.I))), "pt_has_phone": int(bool(re.search(r"(?:\+?62|0)8\d{8,12}", raw))), "pt_has_npwp": int(bool(re.search(r"\d{2}\.\d{3}", raw))), "pt_is_centered": int(abs((candidate.bbox[0] + candidate.bbox[2]) / 2 - .5) < .12), "pt_is_left_aligned": int(candidate.bbox[0] < .15), "pt_person_like": int(legal == "none" and all(word[:1].isupper() for word in words) and 1 <= len(words) <= 4), **_party_role_features(candidate)})
        elif candidate.cand_type == "enum":
            features.update({"en_norm": candidate.parsed_value, "en_match_type": "exact", "en_in_signature_region": int(candidate.bbox[1] > .7), "en_symbol_vs_code": "symbol" if candidate.raw_text in {"Rp", "Rp.", "$"} else "code" if candidate.raw_text.upper() in {"IDR", "USD", "SGD", "EUR"} else "word"})
        elif candidate.cand_type == "text":
            raw = candidate.raw_text
            features.update({"tx_n_tokens": len(raw.split()), "tx_n_chars": len(raw), "tx_n_lines": len(candidate.tokens), "tx_alpha_ratio": sum(char.isalpha() for char in raw) / max(len(raw.replace(" ", "")), 1), "tx_digit_ratio": sum(char.isdigit() for char in raw) / max(len(raw.replace(" ", "")), 1), "tx_upper_ratio": sum(char.isupper() for char in raw) / max(sum(char.isalpha() for char in raw), 1), "tx_has_amount": int(any(char.isdigit() for char in raw))})
        return features

    def _lexical_features(self, candidate: Candidate) -> dict[str, Any]:
        features = {}
        windows = {"left": candidate.context.get("ctx_left", ""), "above": candidate.context.get("ctx_above", ""), "right": candidate.context.get("ctx_right", ""), "line": candidate.context.get("ctx_line", ""), "below": candidate.context.get("ctx_below", ""), "sent": candidate.context.get("ctx_sent", "")}
        relevant = {
            "amount": {"subtotal", "tax_amount", "grand_total", "total_row", "table_end"},
            "party": {"customer_name", "party_roles", "titles"},
            "id": {"invoice_number", "titles"},
            "date": {"invoice_date", "due_date"},
            "table": {"total_row", "table_end", "titles"},
            "address": {"party_roles"},
            "city": {"party_roles"},
            "enum": set(),
        }.get(candidate.cand_type, set())
        for group, phrases in self.lexicon.items():
            if group.startswith("_") or group not in relevant or not isinstance(phrases, list):
                continue
            for window, value in windows.items():
                if window == "sent" and candidate.page.page_number < 0:
                    continue
                score = max((_window_similarity(phrase, value) for phrase in phrases), default=0.0)
                features[f"kw_{group}__{window}"] = score
                features[f"kw_{group}__{window}__gap"] = 1.0 if score == 0 else min(6, len(value.split())) / 6
            features[f"kw_{group}__left_sep"] = int(bool(re.search(r"[:=\-]", candidate.context.get("ctx_left", ""))) and features.get(f"kw_{group}__left", 0) > 0)
        return features

    def _row(self, document_id: str, doc_type: str, run_id: str, vendor_group: str, template_group: str, candidate: Candidate) -> dict[str, Any]:
        x1, y1, x2, y2 = candidate.bbox
        candidate_id = hashlib.sha1(f"{document_id}|{candidate.page.page_number}|{candidate.cand_type}|{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}|{candidate.raw_text}".encode()).hexdigest()[:16]
        row = {"m_schema_version": "1.0.0", "m_pipeline_version": self.pipeline_version, "m_run_id": run_id, "m_candidate_id": candidate_id, "m_doc_type": doc_type, "m_document_id": document_id, "m_vendor_group": vendor_group or "UNKNOWN", "m_template_group": template_group or "UNKNOWN", "m_page": candidate.page.page_number, "m_cand_type": candidate.cand_type, "m_raw_text": candidate.raw_text, "m_parsed_value": candidate.parsed_value, "m_parsed_norm": candidate.parsed_norm, "m_x1": round(x1, 4), "m_y1": round(y1, 4), "m_x2": round(x2, 4), "m_y2": round(y2, 4), "m_line_idx": candidate.line_idx, "m_tok_start": candidate.tokens[0].tok_idx, "m_tok_end": candidate.tokens[-1].tok_idx, "m_overlap_types": candidate.features.pop("m_overlap_types", ""), **candidate.features, **candidate.context}
        for field in self.registry.fields_for_type(doc_type, candidate.cand_type):
            row[f"label__{field}"] = ""
        row.update({"label_status": "", "label_notes": ""})
        return row


def _overlaps(left, right) -> bool:
    return left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]


def _page_line_texts(page: Page) -> list[str]:
    return [" ".join(token.text for token in line) for line in page.lines]


def _party_cap_priority(candidate: Candidate) -> int:
    """Keep party values near explicit name/role labels when a page hits its cap."""
    lines = _page_line_texts(candidate.page)
    index = candidate.line_idx
    nearby = " ".join(lines[max(0, index - 2): min(len(lines), index + 3)])
    current = norm_text(lines[index]) if 0 <= index < len(lines) else ""
    priority = 0
    if any(re.search(r"\bnama\b", norm_text(line)) for line in lines[max(0, index - 1): min(len(lines), index + 2)]):
        priority += 5
    if re.search(r"\b(?:pihak pertama|pihak kedua|yang menyerahkan|yang menerima|pengirim|penerima|sender|receiver|shipper|consignee)\b", norm_text(nearby)):
        priority += 3
    if current.startswith("(") and re.search(r"[A-Za-z]{3,}", current):
        priority += 1
    return priority


def _party_role_features(candidate: Candidate) -> dict[str, Any]:
    lines = _page_line_texts(candidate.page)
    index = candidate.line_idx
    normalized = [norm_text(line) for line in lines]
    first_positions = [i for i, line in enumerate(normalized) if re.search(r"\bpihak pertama\b|\byang menyerahkan\b|\bpemberi\b|\bsender\b|\bshipper\b", line)]
    second_positions = [i for i, line in enumerate(normalized) if re.search(r"\bpihak kedua\b|\byang menerima\b|\bpenerima\b|\breceiver\b|\bconsignee\b", line)]
    name_window = normalized[max(0, index - 1): min(len(normalized), index + 2)]
    n_lines = max(len(lines), 1)
    return {
        "pt_near_name_label": int(any(re.search(r"\bnama\b", line) for line in name_window)),
        "pt_near_first_party_marker": int(any(abs(index - marker) <= 5 for marker in first_positions)),
        "pt_near_second_party_marker": int(any(abs(index - marker) <= 5 for marker in second_positions)),
        "pt_dist_first_party_marker": min((abs(index - marker) / n_lines for marker in first_positions), default=""),
        "pt_dist_second_party_marker": min((abs(index - marker) / n_lines for marker in second_positions), default=""),
        "pt_before_first_party_marker": int(bool(first_positions) and index < min(first_positions)),
        "pt_before_second_party_marker": int(bool(second_positions) and index < min(second_positions)),
    }


def _table_candidate_rows(pipeline, document_id: str, doc_type: str, run_id: str, vendor_group: str, template_group: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Build table candidates from serialized Docling evidence only.

    Table detection belongs to evidence production.  Candidate generation must
    not reopen page images or invoke a table provider; ``pages[].tables[]`` is
    the table source of truth here.
    """
    matcher = None
    aliases = pipeline.config_dir / "line_item_aliases.yaml"
    abbreviations = pipeline.config_dir / "abbreviations.yaml"
    if aliases.exists() and abbreviations.exists():
        from .fuzzy_fields import FuzzyFieldMatcher
        matcher = FuzzyFieldMatcher(doc_type, aliases, abbreviations)
    pages = {int(page.get("page_number", 1)): page for page in evidence.get("pages", [])}
    candidates: list[dict[str, Any]] = []
    for page_number in sorted(pages):
        page = pages[page_number]
        for ordinal, table in enumerate(page.get("tables", []), start=1):
            candidates.append(_make_table_candidate(pipeline, document_id, doc_type, run_id, vendor_group, template_group, page, table, matcher, ordinal))
    scores = sorted(((float(row.get("tb_heuristic_score", 0.0)), index) for index, row in enumerate(candidates)), key=lambda item: (-item[0], item[1]))
    for rank, (_, index) in enumerate(scores, start=1):
        candidates[index]["tb_heuristic_rank"] = rank
    for left_index, left in enumerate(candidates):
        overlaps = {"table"}
        for right_index, right in enumerate(candidates):
            if left_index == right_index:
                continue
            if left["m_page"] == right["m_page"] and _overlaps((left["m_x1"], left["m_y1"], left["m_x2"], left["m_y2"]), (right["m_x1"], right["m_y1"], right["m_x2"], right["m_y2"])):
                overlaps.add("table")
        left["m_overlap_types"] = "|".join(sorted(overlaps)) if len(overlaps) > 1 else "table"
    return candidates


def _make_table_candidate(pipeline, document_id, doc_type, run_id, vendor_group, template_group, page, table, matcher, ordinal):
    bbox = [float(value) for value in table.get("bbox_norm", [0, 0, 1, 1])[:4]]
    headers = [str(value or "") for value in table.get("headers", [])]
    rows = [[str(value or "") for value in row] for row in table.get("rows", [])]
    summary_rows = [[str(value or "") for value in row] for row in table.get("summary_rows", [])]
    n_rows = int(table.get("row_count", len(rows)) or 0)
    n_cols = int(table.get("col_count", len(headers) or max((len(row) for row in rows), default=0)) or 0)
    raw_text = " | ".join(headers) if any(headers) else (" | ".join(rows[0]) if rows else "")
    features = _table_features(pipeline, doc_type, page, table, headers, rows, summary_rows, n_rows, n_cols, matcher)
    x1, y1, x2, y2 = bbox
    candidate_id = hashlib.sha1(f"{document_id}|{page.get('page_number', 1)}|table|{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}|{raw_text}".encode()).hexdigest()[:16]
    row = {
        "m_schema_version": "1.0.0", "m_pipeline_version": pipeline.pipeline_version, "m_run_id": run_id,
        "m_candidate_id": candidate_id, "m_doc_type": doc_type, "m_document_id": document_id,
        "m_vendor_group": vendor_group or "UNKNOWN", "m_template_group": template_group or "UNKNOWN",
        "m_page": int(page.get("page_number", 1)), "m_page_end": int(table.get("page_end", page.get("page_number", 1)) or page.get("page_number", 1)),
        "m_table_id": str(table.get("table_id") or f"{document_id}-p{page.get('page_number', 1)}-t{ordinal}"),
        "m_cand_type": "table", "m_raw_text": raw_text, "m_parsed_value": "", "m_parsed_norm": f"{n_rows}x{n_cols}",
        "m_x1": round(x1, 4), "m_y1": round(y1, 4), "m_x2": round(x2, 4), "m_y2": round(y2, 4),
        "m_line_idx": "", "m_tok_start": "", "m_tok_end": "", "m_evidence_role": table.get("role") or "", "m_evidence_role_reason": table.get("role_reason") or "",
        "label_status": "", "label_notes": "",
        **features,
    }
    for field in pipeline.registry.fields_for_type(doc_type, "table"):
        row[f"label__{field}"] = ""
    return row


def _table_features(pipeline, doc_type, page, table, headers, rows, summary_rows, n_rows, n_cols, matcher):
    nonempty = lambda value: bool(str(value).strip())
    fills = [sum(nonempty(value) for value in row[:n_cols]) / max(n_cols, 1) for row in rows]
    col_values = [[row[index] if index < len(row) else "" for row in rows] for index in range(n_cols)]
    numeric = []
    amount_cols = integer_cols = 0
    for values in col_values:
        parsed = [number for value in values if nonempty(value) for number in [_number(value)] if number is not None]
        ratio = len(parsed) / max(sum(nonempty(value) for value in values), 1)
        numeric.append(ratio >= .6)
        if ratio >= .6 and parsed:
            if all(value == value.to_integral_value() for value in parsed):
                integer_cols += 1
            if any(re.search(r"(?:rp|idr|usd|\$|[.,]\d{3})", value, re.I) for value in values):
                amount_cols += 1
    header_scores = [matcher.best_score(header) if matcher and header else 0.0 for header in headers]
    cells = table.get("cells", [])
    y_centers = sorted({round((float(cell.get("bbox_norm", [0, 0, 0, 0])[1]) + float(cell.get("bbox_norm", [0, 0, 0, 0])[3])) / 2, 4) for cell in cells if len(cell.get("bbox_norm", [])) >= 4})
    row_gaps = [right - left for left, right in zip(y_centers, y_centers[1:]) if right > left]
    rel_product = _relation_fraction(rows, "product")
    summary_match = _summary_column_match(rows, summary_rows)
    total_summary = int(any(_total_label(cell) for row in summary_rows for cell in row if nonempty(cell)))
    score_inputs = {
        "tb_header_fuzzy_frac": sum(score >= .8 for score in header_scores) / max(n_cols, 1),
        "tb_anchor_frac": 0.0, "tb_has_seq_col": 0, "tb_rel_product_frac": rel_product,
        "tb_summary_matches_colsum": summary_match, "tb_has_total_summary": total_summary,
        "tb_frac_numeric_cols": sum(numeric) / max(n_cols, 1), "tb_n_amount_cols_capped": min(amount_cols, 3) / 3,
        "tb_n_rows_capped": min(n_rows, 3) / 3, "tb_row_fill_mean": sum(fills) / max(len(fills), 1),
        "tb_title_kw_items": 0.0, "tb_kv_like_frac": 0.0, "tb_text_only_frac": sum(bool(re.fullmatch(r"[A-Za-zÀ-ÿ /&-]+", value.strip())) for row in rows for value in row if value.strip()) / max(sum(nonempty(value) for row in rows for value in row), 1),
        "tb_title_kw_terms": 0.0, "tb_title_kw_totals": 0.0,
    }
    score = score_inputs["tb_header_fuzzy_frac"] * 2 + score_inputs["tb_rel_product_frac"] * 1.5 + score_inputs["tb_summary_matches_colsum"] + score_inputs["tb_has_total_summary"] * .5 + score_inputs["tb_frac_numeric_cols"] + score_inputs["tb_n_amount_cols_capped"] * .5 + score_inputs["tb_n_rows_capped"] + score_inputs["tb_row_fill_mean"] * .5 - score_inputs["tb_text_only_frac"] - score_inputs["tb_kv_like_frac"] * 1.5
    return {
        "tb_n_rows": n_rows, "tb_n_cols": n_cols, "tb_n_summary_rows": len(summary_rows), "tb_has_summary": int(bool(summary_rows)), "tb_has_header": int(bool(headers)),
        "tb_n_pages": int(table.get("page_end", page.get("page_number", 1)) or page.get("page_number", 1)) - int(page.get("page_number", 1)) + 1, "tb_cut_by_segment": 0,
        "tb_n_visual_rows": len(y_centers), "tb_row_pitch_cv": (statistics.pstdev(row_gaps) / statistics.mean(row_gaps) if len(row_gaps) >= 2 and statistics.mean(row_gaps) else ""),
        "tb_structural_score": float(table.get("confidence", 0) or 0), "tb_page_digital": int((table.get("validation") or {}).get("page_kind") == "DIGITAL"),
        "tb_row_fill_mean": score_inputs["tb_row_fill_mean"], "tb_row_fill_min": min(fills) if fills else 0.0, "tb_col_fill_min": min((sum(nonempty(value) for value in values) / max(len([item for item in values if nonempty(item)]), 1) for values in col_values), default=0.0),
        "tb_header_fuzzy_hits": sum(score >= .8 for score in header_scores), "tb_header_fuzzy_frac": score_inputs["tb_header_fuzzy_frac"], "tb_header_score_mean": sum(header_scores) / max(len(header_scores), 1), "tb_header_numeric_frac": 0.0,
        "tb_frac_numeric_cols": score_inputs["tb_frac_numeric_cols"], "tb_n_amount_cols": amount_cols, "tb_n_int_cols": integer_cols, "tb_n_id_cols": 0, "tb_n_date_cols": 0, "tb_n_city_cols": 0, "tb_n_enum_cols": 0, "tb_n_text_cols": n_cols - sum(numeric), "tb_text_only_frac": score_inputs["tb_text_only_frac"], "tb_kv_like_frac": 0.0, "tb_mean_unique_ratio": 0.0,
        "tb_has_seq_col": 0, "tb_seq_len": "", "tb_anchor_frac": 0.0, "tb_rel_product_frac": rel_product, "tb_rel_sum_frac": "", "tb_summary_matches_colsum": summary_match, "tb_has_total_summary": total_summary, "tb_summary_rate_match": 0,
        "tb_x1": round(table.get("bbox_norm", [0, 0, 1, 1])[0], 4), "tb_y1": round(table.get("bbox_norm", [0, 0, 1, 1])[1], 4), "tb_x2": round(table.get("bbox_norm", [0, 0, 1, 1])[2], 4), "tb_y2": round(table.get("bbox_norm", [0, 0, 1, 1])[3], 4), "tb_width": round(table.get("bbox_norm", [0, 0, 1, 1])[2] - table.get("bbox_norm", [0, 0, 1, 1])[0], 4), "tb_height": round(table.get("bbox_norm", [0, 0, 1, 1])[3] - table.get("bbox_norm", [0, 0, 1, 1])[1], 4), "tb_area_frac": round((table.get("bbox_norm", [0, 0, 1, 1])[2] - table.get("bbox_norm", [0, 0, 1, 1])[0]) * (table.get("bbox_norm", [0, 0, 1, 1])[3] - table.get("bbox_norm", [0, 0, 1, 1])[1]), 4), "tb_y_center": round((table.get("bbox_norm", [0, 0, 1, 1])[1] + table.get("bbox_norm", [0, 0, 1, 1])[3]) / 2, 4), "tb_is_last_block_on_page": 1,
        **score_inputs, "tb_heuristic_score": round(score, 6), "tb_heuristic_rank": "",
        "g_x1": round(table.get("bbox_norm", [0, 0, 1, 1])[0], 4), "g_y1": round(table.get("bbox_norm", [0, 0, 1, 1])[1], 4), "g_x2": round(table.get("bbox_norm", [0, 0, 1, 1])[2], 4), "g_y2": round(table.get("bbox_norm", [0, 0, 1, 1])[3], 4),
    }


def _number(value):
    from .parsers import parse_number
    parsed = parse_number(str(value).replace("Rp", "").strip())
    return parsed.value


def _relation_fraction(rows, kind):
    valid = matches = 0
    for row in rows:
        values = [_number(value) for value in row]
        values = [value for value in values if value is not None]
        if len(values) < 3:
            continue
        valid += 1
        for left in values:
            rest = [value for value in values if value is not left]
            if any(abs(left - (a * b if kind == "product" else a + b)) <= max(Decimal("1"), abs(left) * Decimal("0.000001")) for index, a in enumerate(rest) for b in rest[index + 1:]):
                matches += 1
                break
    return matches / valid if valid else 0.0


def _summary_column_match(rows, summary_rows):
    columns = []
    for index in range(max((len(row) for row in rows), default=0)):
        values = [_number(row[index]) for row in rows if index < len(row) and _number(row[index]) is not None]
        columns.append(sum(values, Decimal(0)) if values else None)
    return int(any(_number(value) is not None and any(abs(_number(value) - total) <= max(Decimal("1"), abs(total) * Decimal("0.000001")) for total in columns if total is not None) for row in summary_rows for value in row))


def _total_label(value):
    return bool(re.search(r"\b(?:subtotal|sub total|total|grand total|jumlah|pajak|ppn|tax|diskon)\b", norm_text(value)))


def _shape_compact(value: str) -> str:
    classes = []
    for char in value:
        current = "9" if char.isdigit() else "A" if char.isupper() else "a" if char.islower() else char
        if not classes or classes[-1] != current:
            classes.append(current)
    return "".join(classes)


def _near_unit(page: Page, token: Token, units: set[str]) -> str:
    if any(token.text.lower().endswith(unit) and token.text.lower() != unit for unit in units):
        return next(unit for unit in units if token.text.lower().endswith(unit) and token.text.lower() != unit)
    for candidate in page.tokens:
        if candidate.line_idx == token.line_idx and candidate.tok_idx == token.tok_idx + 1 and norm_text(candidate.text) in units:
            return norm_text(candidate.text)
    return "none"


def table_evidence(table) -> dict[str, Any]:
    """Serialize additive table diagnostics while retaining the legacy fields."""
    payload = {
        "table_id": table.table_id,
        "bbox_norm": list(table.bbox), "headers": [column.header_text for column in table.columns],
        "rows": [[cell.text for cell in row.cells] for row in table.rows],
        "row_count": len(table.rows), "col_count": len(table.columns), "confidence": table.structural_score or table.conf, "confidence_source": "structural_score" if table.structural_score else "unscored",
        "provider": table.source, "reason": table.reason, "validation": table.validation,
        "columns": [{"idx": column.idx, "bbox_norm": [column.x1, 0.0, column.x2, 1.0], "header": column.header_text} for column in table.columns],
        "cells": [{"row": cell.row_idx, "col": cell.col_idx, "text": cell.text, "bbox_norm": list(cell.bbox)} for row in table.rows for cell in row.cells],
        "summary_rows": [[cell.text for cell in row.cells] for row in table.summary_rows],
        "role": table.role, "role_reason": table.role_reason,
    }
    if table.validation.get("continued_from"):
        payload["continued_from"] = table.validation["continued_from"]
    return payload


def _natural(table_id: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", table_id)]


def stitch_tables(tables):
    """Join tables of consecutive pages that are one logical table; the inputs are NOT modified.

    Two tables join when the first has no summary rows (it did not end), they sit on consecutive pages, and either the
    headers match or the second is a header-less continuation of the first. Ordering is by page, then natural id order
    (t2 before t10).
    """
    stitched: list = []
    for table in sorted((copy.deepcopy(t) for t in tables), key=lambda value: (value.page_start, _natural(value.table_id))):
        previous = next((candidate for candidate in reversed(stitched) if candidate.page_end + 1 == table.page_start), None)
        if previous is not None and _same_table_segment(previous, table):
            offset = len(previous.rows)
            for row in table.rows:
                row.idx += offset
                for cell in row.cells:
                    cell.row_idx = row.idx
            previous.rows.extend(table.rows)
            previous.summary_rows.extend(table.summary_rows)
            previous.page_end = table.page_end
            previous.stitched_pages.extend(page for page in table.stitched_pages if page not in previous.stitched_pages)
            previous.reason = "consecutive_page_continuation" if table.reason == "continuation" else "consecutive_page_header_match"
            previous.validation.setdefault("stitched_table_ids", [previous.table_id]).append(table.table_id)
        else:
            stitched.append(table)
    return stitched


def _same_table_segment(left, right) -> bool:
    if left.summary_rows or left.page_end + 1 != right.page_start or len(left.columns) != len(right.columns):
        return False
    if right.reason == "continuation":
        return right.validation.get("continued_from") in (left.table_id, *left.validation.get("stitched_table_ids", []))
    return [norm_text(column.header_text) for column in left.columns] == [norm_text(column.header_text) for column in right.columns]
