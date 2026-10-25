from __future__ import annotations

import hashlib
import json
import csv
import yaml
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..fuzzy_fields import FuzzyFieldMatcher
from ..schema import Table
from ..text_norm import norm_text


class LineItemBuilder:
    """Convert token-grid tables into label-ready line-item records."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.aliases = self.config_dir / "line_item_aliases.yaml"
        self.abbreviations = self.config_dir / "abbreviations.yaml"
        self.cities = _load_cities(self.config_dir / "gazetteer_cities.csv")
        self.table_resolution = yaml.safe_load((self.config_dir / "table_resolution.yaml").read_text(encoding="utf-8")) if (self.config_dir / "table_resolution.yaml").exists() else {}

    def build(self, document_id: str, doc_type: str, tables: list[Table]) -> dict[str, Any]:
        matcher = FuzzyFieldMatcher(doc_type, self.aliases, self.abbreviations)
        columns, cells, rows, previews = [], [], [], []
        for table in tables:
            column_matches = []
            for column in table.columns:
                match = matcher.match(column.header_text)
                top = match.candidates[0] if match.candidates else None
                field = top.field if top else "EXTRA"
                score = top.score if top else 0.0
                column_matches.append((field, score, match))
            column_matches = self._resolve_ambiguous_columns(table, column_matches)
            for column, (field, score, match) in zip(table.columns, column_matches):
                candidate_id = hashlib.sha1(f"{document_id}|{table.table_id}|{column.idx}|{field}".encode()).hexdigest()[:16]
                columns.append({"m_document_id": document_id, "m_doc_type": doc_type, "m_table_id": table.table_id, "m_page_start": table.page_start, "m_page_end": table.page_end, "m_table_source": table.source, "li_col_idx": column.idx, "li_field": field, "header_raw": column.header_text, "header_norm": norm_text(column.header_text), "m_candidate_id": candidate_id, "li_fuzzy_score": score, "li_fuzzy_rank": 1 if top else "", "li_fuzzy_margin": match.margin, "li_match_type": top.match_type if top else "", "li_n_cols": len(table.columns), "li_n_rows": len(table.rows), "li_table_conf": table.conf})
            preview_rows = []
            for row in table.rows:
                row_values = {}
                nonempty = 0
                for cell, (field, score, _) in zip(row.cells, column_matches):
                    value, value_type, ok, ambiguous, alt = _parse_cell(cell.text, field, self.cities)
                    cells.append({"m_document_id": document_id, "m_doc_type": doc_type, "m_table_id": table.table_id, "m_page": table.page_start, "row_idx": row.idx, "col_idx": cell.col_idx, "field_norm": field, "field_score": score, "header_raw": table.columns[cell.col_idx].header_text, "header_norm": norm_text(table.columns[cell.col_idx].header_text), "cell_text": cell.text, "value_parsed": value, "value_type": value_type, "parse_ok": int(ok), "parse_ambiguous": int(ambiguous), "value_alt": alt, "x1": round(cell.bbox[0], 4), "y1": round(cell.bbox[1], 4), "x2": round(cell.bbox[2], 4), "y2": round(cell.bbox[3], 4), "is_continuation_merged": int(row.continuation_rows > 0), "kind_hint": row.kind_hint, "cell_empty_marker": int(norm_text(cell.text) in {"", "-", "n/a"})})
                    if cell.text:
                        nonempty += 1
                    if field != "EXTRA" and value is not None:
                        row_values[field] = value
                rows.append({"m_document_id": document_id, "m_doc_type": doc_type, "m_table_id": table.table_id, "lr_row_idx": row.idx, "lr_n_cells": len(row.cells), "lr_n_nonempty": nonempty, "lr_fill_ratio": nonempty / max(len(row.cells), 1), "lr_total_kw_sim": int(row.kind_hint == "total_like"), "lr_continuation_rows_merged": row.continuation_rows, "label__is_line_item_row": "", "label__row_type": "", "label_notes": ""})
                preview_rows.append({"row": row.idx, **row_values})
            previews.append({"document_id": document_id, "table_id": table.table_id, "source": table.source, "columns": [{"col": column.idx, "header_raw": column.header_text, "field": match[0], "score": match[1]} for column, match in zip(table.columns, column_matches)], "rows": preview_rows, "unmapped_columns": [column.idx for column, match in zip(table.columns, column_matches) if match[0] == "EXTRA"], "warnings": []})
        return {"columns": columns, "cells": cells, "rows": rows, "previews": previews}

    def _resolve_ambiguous_columns(self, table: Table, matches):
        rules = self.table_resolution.get("ambiguous_headers", {})
        result = list(matches)
        for index, column in enumerate(table.columns):
            rule = rules.get(norm_text(column.header_text))
            if not rule:
                continue
            candidates = {candidate.field for candidate in matches[index][2].candidates}
            quantity, amount = rule.get("quantity_field"), rule.get("amount_field")
            if not {quantity, amount} <= candidates:
                continue
            prior_fields = {field for field, _, _ in result[:index]}
            values = [_number(row.cells[index].text) for row in table.rows if index < len(row.cells)]
            floor = Decimal(str(rule["amount_value_floor"]))
            is_amount = bool(prior_fields & set(rule.get("rate_fields", []))) or any(value is not None and value >= floor for value in values)
            selected = amount if is_amount else quantity
            result[index] = (selected, next((score.score for score in matches[index][2].candidates if score.field == selected), 0.0), matches[index][2])
        return result


def _parse_cell(text: str, field: str, cities: dict[str, str] | None = None) -> tuple[str | None, str, bool, bool, str | None]:
    raw = text.strip()
    if not raw or norm_text(raw) in {"-", "—", "n/a"}:
        return None, "empty", False, False, None
    if field in {"quantity", "line_number", "quantity_delivered", "quantity_received", "weight_min_kg", "weight_max_kg", "chargeable_weight", "weight_kg", "base_rate", "base_rate_idr", "line_total", "fuel_surcharge"}:
        from ..parsers.numbers import parse_number
        parsed = parse_number(raw.replace("kg", "").strip())
        return (str(parsed.value) if parsed.value is not None else None, "number", parsed.value is not None, parsed.ambiguous, str(parsed.value_alt) if parsed.value_alt is not None else None)
    if field in {"origin", "destination", "origin_zone", "destination_zone"}:
        canonical = (cities or {}).get(norm_text(raw).upper())
        if canonical:
            return canonical, "city", True, False, None
        return raw.upper(), "city_unknown", False, False, None
    if field == "condition":
        normalized = norm_text(raw)
        value = "GOOD" if normalized in {"baik", "good", "ok", "lengkap", "complete"} else "DAMAGED" if normalized in {"rusak", "damaged", "pecah", "penyok"} else "SHORTAGE" if normalized in {"kurang", "shortage"} else "OTHER"
        return value, "enum", True, False, None
    return raw, "text", True, False, None


def _load_cities(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["alias"].upper(): row["canonical"].upper() for row in csv.DictReader(handle) if row.get("alias") and row.get("canonical")}


def _number(text: str) -> Decimal | None:
    from ..parsers.numbers import parse_number
    return parse_number(text).value
