from __future__ import annotations

import hashlib
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .context import build_page, context_for
from .parsers import parse_date, parse_number, parse_time
from .registry import FieldRegistry
from .schema import Candidate, Page, Token
from .text_norm import norm_alnum, norm_text, ocr_fold
from .fuzzy_fields import _window_similarity
from .version import candidate_pipeline_version
from .tables import TableExtractor
from app.core.config import settings


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
        self.table_extractor = TableExtractor.from_config(config_dir, settings.table_provider_order)

    def generate(self, document_id: str, doc_type: str, ocr_result: dict[str, Any], run_id: str = "", vendor_group: str = "UNKNOWN", template_group: str = "UNKNOWN") -> list[dict[str, Any]]:
        started = time.perf_counter()
        self.last_report = {"empty_pages": [], "cap_overflows": [], "timings_ms": {}}
        enabled = {field.cand_type for field in self.registry.for_doc(doc_type)}
        pages = [build_page(page) for page in ocr_result.get("pages", [])]
        if settings.enable_table_extractor:
            self.last_report["tables"] = [
                table_evidence(table)
                for page in pages
                for table in self.table_extractor.extract(page, doc_type, document_id, arithmetic_validation=settings.enable_arithmetic_validation, relative_tolerance=Decimal(str(settings.arith_row_tolerance_relative)), absolute_tolerance=Decimal(str(settings.arith_row_tolerance_absolute_idr)))
            ]
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
        return [self._row(document_id, doc_type, run_id, vendor_group, template_group, candidate) for candidate in candidates]
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
                if re.search(r"%$", token.text) or re.fullmatch(r"\d{4}", text) or (parsed.value < 1000 and not parsed.sym and not any(word in text for word in (".", ","))):
                    continue
                if re.fullmatch(r"(?:\+?62|0)8\d{8,12}", text):
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
            ranked = sorted(group, key=lambda item: (-sum(t.confidence for t in item.tokens) / len(item.tokens), item.bbox[1], item.bbox[0], item.raw_text))
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
            features.update({"pt_n_tokens": len(words), "pt_n_chars": len(raw), "pt_upper_ratio": sum(char.isupper() for char in raw) / max(letters, 1), "pt_title_ratio": sum(bool(word) and word[0].isupper() for word in words) / max(len(words), 1), "pt_digit_ratio": sum(char.isdigit() for char in raw) / max(len(raw.replace(" ", "")), 1), "pt_alpha_ratio": letters / max(len(raw.replace(" ", "")), 1), "pt_legal_marker": legal, "pt_legal_pos": "prefix" if legal != "none" and norm_text(raw).split()[0] == legal else "suffix" if legal != "none" else "none", "pt_has_email": int("@" in raw), "pt_has_url": int(bool(re.search(r"https?://|www\.", raw, re.I))), "pt_has_phone": int(bool(re.search(r"(?:\+?62|0)8\d{8,12}", raw))), "pt_has_npwp": int(bool(re.search(r"\d{2}\.\d{3}", raw))), "pt_is_centered": int(abs((candidate.bbox[0] + candidate.bbox[2]) / 2 - .5) < .12), "pt_is_left_aligned": int(candidate.bbox[0] < .15), "pt_person_like": int(legal == "none" and all(word[:1].isupper() for word in words) and 1 <= len(words) <= 4)})
        elif candidate.cand_type == "enum":
            features.update({"en_norm": candidate.parsed_value, "en_match_type": "exact", "en_in_signature_region": int(candidate.bbox[1] > .7), "en_symbol_vs_code": "symbol" if candidate.raw_text in {"Rp", "Rp.", "$"} else "code" if candidate.raw_text.upper() in {"IDR", "USD", "SGD", "EUR"} else "word"})
        elif candidate.cand_type == "text":
            raw = candidate.raw_text
            features.update({"tx_n_tokens": len(raw.split()), "tx_n_chars": len(raw), "tx_n_lines": len(candidate.tokens), "tx_alpha_ratio": sum(char.isalpha() for char in raw) / max(len(raw.replace(" ", "")), 1), "tx_digit_ratio": sum(char.isdigit() for char in raw) / max(len(raw.replace(" ", "")), 1), "tx_upper_ratio": sum(char.isupper() for char in raw) / max(sum(char.isalpha() for char in raw), 1), "tx_has_amount": int(any(char.isdigit() for char in raw))})
        return features

    def _lexical_features(self, candidate: Candidate) -> dict[str, Any]:
        features = {}
        windows = {"left": candidate.context.get("ctx_left", ""), "above": candidate.context.get("ctx_above", ""), "right": candidate.context.get("ctx_right", ""), "line": candidate.context.get("ctx_line", ""), "below": candidate.context.get("ctx_below", ""), "sent": candidate.context.get("ctx_sent", "")}
        for group, phrases in self.lexicon.items():
            if group.startswith("_") or not isinstance(phrases, list):
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
    return {
        "bbox_norm": list(table.bbox), "headers": [column.header_text for column in table.columns],
        "rows": [[cell.text for cell in row.cells] for row in table.rows],
        "row_count": len(table.rows), "col_count": len(table.columns), "confidence": table.conf,
        "provider": table.source, "reason": table.reason, "validation": table.validation,
        "columns": [{"idx": column.idx, "bbox_norm": [column.x1, 0.0, column.x2, 1.0], "header": column.header_text} for column in table.columns],
        "cells": [{"row": cell.row_idx, "col": cell.col_idx, "text": cell.text, "bbox_norm": list(cell.bbox)} for row in table.rows for cell in row.cells],
    }
