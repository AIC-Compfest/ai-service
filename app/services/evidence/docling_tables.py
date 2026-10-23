from __future__ import annotations

from copy import deepcopy
from typing import Any


_REQUIRED = ("bbox_norm", "columns", "cells", "rows", "row_count", "col_count", "confidence", "provider", "reason", "validation")


def normalize_docling_evidence(evidence: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize Docling evidence without detecting or reconstructing tables."""
    result = deepcopy(evidence)
    warnings: list[dict[str, Any]] = []
    for page in result.get("pages", []):
        coord_space = page.get("coord_space")
        if coord_space not in (None, "page_image_normalized", "legacy_1000"):
            raise ValueError(f"unsupported coord_space: {coord_space}")
        divide = coord_space == "legacy_1000" or (coord_space is None and _has_legacy_values(page))
        _normalize_page_boxes(page, divide)
        page.setdefault("coord_space", "page_image_normalized")
        page.setdefault("tables", [])
        page_warnings = []
        for index, table in enumerate(page["tables"], start=1):
            missing = [key for key in _REQUIRED if key not in table]
            if missing:
                raise ValueError(f"Docling table is missing required keys: {', '.join(missing)}")
            table.setdefault("table_id", f"p{page.get('page_number', 1)}-t{index}")
            table.setdefault("headers", [])
            table.setdefault("summary_rows", [])
            table.setdefault("role", None)
            table.setdefault("role_reason", None)
            table.setdefault("validation_warning", [])
            _normalize_table_boxes(table, divide)
            header_by_col = {i: str(value).strip() for i, value in enumerate(table["headers"])}
            for cell in table["cells"]:
                if int(cell.get("row", -1)) == 0 and str(cell.get("text", "")).strip() == header_by_col.get(int(cell.get("col", -1)), ""):
                    message = "cell row 0 duplicates a header; cells are expected to be body-only"
                    table["validation_warning"].append(message)
                    page_warnings.append({"table_id": table["table_id"], "warning": message})
                    break
            max_row = max((int(cell.get("row", -1)) for cell in table["cells"]), default=-1) + 1
            if int(table["row_count"]) != len(table["rows"]) or int(table["row_count"]) != max_row:
                message = "row_count does not match rows/cell row extent"
                table["validation_warning"].append(message)
                page_warnings.append({"table_id": table["table_id"], "warning": message})
        if page_warnings:
            warnings.extend({"page": page.get("page_number", 1), **item} for item in page_warnings)
    return result, warnings


def _has_legacy_values(page: dict[str, Any]) -> bool:
    values: list[float] = []
    for line in page.get("lines", []):
        values.extend(float(value) for value in line.get("bbox_norm", [])[:4])
        for token in line.get("tokens", []):
            values.extend(float(value) for value in token.get("bbox_norm", [])[:4])
    for table in page.get("tables", []):
        values.extend(float(value) for value in table.get("bbox_norm", [])[:4])
    return any(value > 1 for value in values)


def _box(value: list[float], divide: bool) -> list[float]:
    values = [float(item) / 1000 if divide else float(item) for item in value[:4]]
    if len(values) != 4 or any(item < 0 or item > 1 for item in values):
        raise ValueError(f"bbox_norm must normalize to [0,1]: {value}")
    return values


def _normalize_page_boxes(page: dict[str, Any], divide: bool) -> None:
    for line in page.get("lines", []):
        if "bbox_norm" in line:
            line["bbox_norm"] = _box(line["bbox_norm"], divide)
        for token in line.get("tokens", []):
            if "bbox_norm" in token:
                token["bbox_norm"] = _box(token["bbox_norm"], divide)
    for region in page.get("regions", []):
        if "bbox_norm" in region:
            region["bbox_norm"] = _box(region["bbox_norm"], divide)


def _normalize_table_boxes(table: dict[str, Any], divide: bool) -> None:
    table["bbox_norm"] = _box(table["bbox_norm"], divide)
    for column in table["columns"]:
        column["bbox_norm"] = _box(column["bbox_norm"], divide)
    for cell in table["cells"]:
        cell["bbox_norm"] = _box(cell["bbox_norm"], divide)
