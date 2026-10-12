from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .text_norm import norm_text, ocr_fold

try:
    from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio
except ImportError:  # pragma: no cover - runtime dependency is listed in requirements
    from difflib import SequenceMatcher

    def ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100

    token_set_ratio = ratio
    token_sort_ratio = ratio


@dataclass(frozen=True)
class FieldScore:
    field: str
    score: float
    match_type: str
    alias: str
    alias_tokens: int


@dataclass(frozen=True)
class HeaderMatch:
    header_raw: str
    header_norm: str
    header_norm_orig: str
    unit_hint: str | None
    header_pct: float | None
    candidates: list[FieldScore]
    margin: float

    @property
    def top(self) -> list[tuple[str, float, str]]:
        return [(item.field, item.score, item.match_type) for item in self.candidates]


class FuzzyFieldMatcher:
    def __init__(self, doc_type: str, aliases_yaml: str | Path, abbreviations_yaml: str | Path, *, top_k: int = 3, min_report: float = 0.30):
        self.doc_type = doc_type
        self.aliases_path = Path(aliases_yaml)
        self.abbreviations = yaml.safe_load(Path(abbreviations_yaml).read_text(encoding="utf-8")) or {}
        data: dict[str, Any] = yaml.safe_load(self.aliases_path.read_text(encoding="utf-8")) or {}
        doc = data.get(doc_type, {})
        self.fields: dict[str, dict[str, Any]] = doc.get("fields", {})
        self.surcharge_types = data.get("_surcharge_types", {})
        self.top_k = top_k
        self.min_report = min_report
        self.field_order = {field: index for index, field in enumerate(self.fields)}

    def match(self, header_raw: str) -> HeaderMatch:
        return self._match_cached(header_raw or "")

    def match_multiline(self, lines: list[str]) -> HeaderMatch:
        joined = " ".join(line for line in lines if line and line.strip())
        matches = [self.match(joined), *(self.match(line) for line in lines)]
        best_by_field: dict[str, FieldScore] = {}
        for result in matches:
            for score in result.candidates:
                adjusted = score if result.header_raw == joined else FieldScore(score.field, score.score * 0.9, score.match_type, score.alias, score.alias_tokens)
                if adjusted.field not in best_by_field or adjusted.score > best_by_field[adjusted.field].score:
                    best_by_field[adjusted.field] = adjusted
        ordered = sorted(best_by_field.values(), key=lambda item: (-item.score, -item.alias_tokens, self.field_order[item.field]))[:self.top_k]
        return HeaderMatch(joined, self._normalize(joined)[0], self._normalize(joined)[1], self._normalize(joined)[2], self._normalize(joined)[3], ordered, ordered[0].score - ordered[1].score if len(ordered) > 1 else (ordered[0].score if ordered else 0.0))

    def best_score(self, text: str) -> float:
        result = self.match(text)
        return result.candidates[0].score if result.candidates else 0.0

    def match_label_text(self, text: str, group: str) -> FieldScore | None:
        aliases = self.surcharge_types if group == "_surcharge_types" else {}
        best: FieldScore | None = None
        for field, values in aliases.items():
            for alias in values if isinstance(values, list) else []:
                score = _window_similarity(alias, text)
                if score >= 0.8 and (best is None or score > best.score):
                    best = FieldScore(field, score, "fuzzy" if score < 1 else "exact", norm_text(alias), len(norm_text(alias).split()))
        return best

    def content_compat(self, field: str, cell_texts: list[str]) -> float:
        spec = self.fields.get(field, {})
        dtype = spec.get("dtype", "text")
        values = [text.strip() for text in cell_texts if text.strip()]
        if not values:
            return 0.0
        good = sum(_dtype_compatible(value, dtype) for value in values)
        return good / len(values)

    @lru_cache(maxsize=4096)
    def _match_cached(self, header_raw: str) -> HeaderMatch:
        original, expanded, unit_hint, header_pct = self._normalize(header_raw)
        if not original:
            return HeaderMatch(header_raw, "", "", unit_hint, header_pct, [], 0.0)
        scores: list[FieldScore] = []
        for field, spec in self.fields.items():
            best: FieldScore | None = None
            for alias_raw in spec.get("aliases", []):
                alias = norm_text(str(alias_raw))
                if not alias:
                    continue
                for form, form_name in ((expanded, "expanded"), (original, "original")):
                    candidate = self._score(field, form, alias, spec.get("negative_aliases", []), form_name)
                    if candidate and (best is None or candidate.score > best.score or (candidate.score == best.score and candidate.match_type == "abbrev" and best.match_type == "exact")):
                        best = candidate
            if best and best.score >= self.min_report:
                scores.append(best)
        scores.sort(key=lambda item: (-item.score, -item.alias_tokens, self.field_order[item.field]))
        scores = scores[:self.top_k]
        margin = scores[0].score - scores[1].score if len(scores) > 1 else (scores[0].score if scores else 0.0)
        return HeaderMatch(header_raw, expanded, original, unit_hint, header_pct, scores, round(margin, 6))

    def _score(self, field: str, header: str, alias: str, negative_aliases: list[str], form_name: str) -> FieldScore | None:
        alias_tokens = len(alias.split())
        if header == alias:
            match_type = "abbrev" if alias in self.abbreviations or (form_name == "expanded" and header != alias) else "exact"
            return FieldScore(field, 1.0, match_type, alias, alias_tokens)
        if len(alias) < 4:
            return None
        header_tokens, alias_parts = header.split(), alias.split()
        matched_header = sum(any(_token_match(token, other) for other in alias_parts) for token in header_tokens)
        matched_alias = sum(any(_token_match(token, other) for other in header_tokens) for token in alias_parts)
        coverage = min(matched_header / max(len(header_tokens), 1), matched_alias / max(len(alias_parts), 1))
        if coverage < 0.60:
            return None
        score = (0.5 * token_set_ratio(header, alias) + 0.3 * token_sort_ratio(header, alias) + 0.2 * _ngram_jaccard(header, alias)) / 100 * coverage
        score = min(0.99, score)
        if score <= 0:
            return None
        if any(_subsequence(norm_text(item), header) for item in negative_aliases):
            score *= 0.5
        folded = norm_text(ocr_fold(header))
        match_type = "fuzzy"
        if folded != header and folded == alias:
            score, match_type = 0.95, "ocr_corrected"
        elif folded != header:
            score = max(score, ((0.5 * token_set_ratio(folded, alias) + 0.3 * token_sort_ratio(folded, alias) + 0.2 * _ngram_jaccard(folded, alias)) / 100 * coverage) * 0.95)
            match_type = "ocr_corrected"
        return FieldScore(field, round(score, 6), match_type, alias, alias_tokens)

    def _normalize(self, value: str) -> tuple[str, str, str | None, float | None]:
        text = unicodedata.normalize("NFKD", value or "")
        text = "".join(char for char in text if not unicodedata.combining(char)).lower()
        unit_hint = None
        unit_match = re.search(r"[\(\[]\s*(kg|kgs|rp|idr|%|m3|m³|cbm|hari|days)\s*[\)\]]", text)
        if unit_match:
            unit_hint = {"kgs": "kg", "m³": "m3", "%": "pct"}.get(unit_match.group(1), unit_match.group(1))
            text = text[:unit_match.start()] + text[unit_match.end():]
        header_pct = None
        remaining = []
        for token in text.split():
            if re.fullmatch(r"\d+(?:[.,]\d+)?%?", token):
                if token.endswith("%"):
                    header_pct = float(token[:-1].replace(",", "."))
                continue
            remaining.append(token)
        original = norm_text(" ".join(remaining))
        expanded_tokens = []
        for index, token in enumerate(original.split()):
            replacement = self.abbreviations.get(token, token)
            if token == "no" and index > 0:
                replacement = token
            expanded_tokens.append(replacement)
        expanded = norm_text(" ".join(expanded_tokens))
        return original, expanded, unit_hint, header_pct


def _token_match(left: str, right: str) -> bool:
    return left == right or (len(left) >= 4 and len(right) >= 4 and ratio(left, right) >= 85)


def _subsequence(needle: str, haystack: str) -> bool:
    needle_tokens, haystack_tokens = needle.split(), haystack.split()
    return any(haystack_tokens[index:index + len(needle_tokens)] == needle_tokens for index in range(len(haystack_tokens) - len(needle_tokens) + 1))


def _ngram_jaccard(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        value = f" {value} "
        return {value[index:index + 3] for index in range(max(len(value) - 2, 0))}
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b) if a | b else 0.0


@lru_cache(maxsize=8192)
def _window_similarity(phrase: str, window: str) -> float:
    phrase_norm, window_norm = norm_text(phrase), norm_text(window)
    if not phrase_norm or not window_norm:
        return 0.0
    tokens = window_norm.split()
    phrase_tokens = phrase_norm.split()
    lengths = {len(phrase_tokens)} | ({len(phrase_tokens) - 1, len(phrase_tokens) + 1} if len(phrase_tokens) >= 2 else set())
    best = 0.0
    for length in lengths:
        for index in range(len(tokens) - length + 1):
            best = max(best, ratio(phrase_norm, " ".join(tokens[index:index + length])) / 100)
    folded = norm_text(ocr_fold(window_norm))
    if folded != window_norm:
        best = max(best, 0.95 * _window_similarity(phrase_norm, folded))
    return best if best >= 0.8 else 0.0


def _dtype_compatible(value: str, dtype: str) -> bool:
    if dtype in {"int", "decimal", "amount", "weight", "percent"}:
        return bool(re.search(r"\d", value))
    if dtype == "id":
        return len(value) >= 4 and bool(re.search(r"\d", value))
    return bool(value)