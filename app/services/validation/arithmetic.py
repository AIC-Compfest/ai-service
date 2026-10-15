"""Decimal-only arithmetic validation for extracted invoice-like tables."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from app.services.candidates.parsers.numbers import parse_number
from app.services.candidates.schema import Table
from app.services.candidates.text_norm import norm_text

_QUANTITY = {"qty", "quantity", "koli", "jumlah", "volume"}
_RATE = {"rate", "harga", "price", "tarif", "unit price", "base rate"}
_TOTAL = {"total", "amount", "jumlah", "line total", "subtotal"}


def validate_table(table: Table, *, relative_tolerance: Decimal, absolute_tolerance: Decimal) -> dict[str, Any]:
    """Validate qty × rate against each line total without altering source cells."""
    indices = _column_indices(table)
    if not {"quantity", "rate", "line_total"} <= indices.keys():
        return {"available": False, "reason_codes": ["ARITH_COLUMNS_MISSING"], "row_checks": []}
    checks = []
    for row in table.rows:
        qty = _value(row.cells[indices["quantity"]].text)
        rate = _value(row.cells[indices["rate"]].text)
        total = _value(row.cells[indices["line_total"]].text)
        if qty is None or rate is None or total is None:
            checks.append({"row": row.idx, "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"})
            continue
        expected = qty * rate
        delta = abs(expected - total)
        allowed = max(absolute_tolerance, abs(expected) * relative_tolerance)
        checks.append({
            "row": row.idx, "status": "PASS" if delta <= allowed else "FAIL",
            "reason_code": "ARITH_ROW_OK" if delta <= allowed else "ARITH_ROW_FAIL",
            "expected": str(expected), "observed": str(total), "delta": str(delta),
            "suggested_value": None if delta <= allowed else str(expected),
        })
    evaluated = [check for check in checks if check["status"] != "SKIPPED"]
    passed = sum(check["status"] == "PASS" for check in evaluated)
    return {"available": bool(evaluated), "reason_codes": [check["reason_code"] for check in checks], "row_checks": checks, "passed": passed, "evaluated": len(evaluated), "score": Decimal(passed) / Decimal(len(evaluated)) if evaluated else Decimal("0")}


def validate_document_totals(*, line_totals: list[Any], subtotal: Any = None, tax: Any = None, withholding_tax: Any = None, grand_total: Any = None, relative_tolerance: Decimal, absolute_tolerance: Decimal) -> dict[str, Any]:
    """Validate observed document totals. Missing values are never synthesized."""
    observed_lines = [_decimal(value) for value in line_totals]
    line_sum = sum((value for value in observed_lines if value is not None), Decimal("0"))
    checks = []
    if subtotal is not None:
        checks.append(_check("subtotal", line_sum, _decimal(subtotal), relative_tolerance, absolute_tolerance, "ARITH_DOC_SUBTOTAL_FAIL"))
    if grand_total is not None and subtotal is not None:
        expected = _decimal(subtotal)
        for value, sign in ((tax, 1), (withholding_tax, -1)):
            parsed = _decimal(value)
            if parsed is not None:
                expected += sign * parsed
        checks.append(_check("grand_total", expected, _decimal(grand_total), relative_tolerance, absolute_tolerance, "ARITH_DOC_TOTAL_FAIL"))
    return {"available": bool(checks), "line_total_sum": str(line_sum), "checks": checks, "reason_codes": [check["reason_code"] for check in checks]}


def _check(name: str, expected: Decimal, observed: Decimal | None, relative: Decimal, absolute: Decimal, fail_code: str) -> dict[str, Any]:
    if observed is None:
        return {"field": name, "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"}
    delta = abs(expected - observed)
    passed = delta <= max(absolute, abs(expected) * relative)
    return {"field": name, "status": "PASS" if passed else "FAIL", "reason_code": "ARITH_DOC_OK" if passed else fail_code, "expected": str(expected), "observed": str(observed), "delta": str(delta), "suggested_value": None if passed else str(expected)}


def _column_indices(table: Table) -> dict[str, int]:
    result = {}
    for column in table.columns:
        header = norm_text(column.header_text)
        if header in _QUANTITY and "quantity" not in result:
            result["quantity"] = column.idx
        elif header in _RATE and "rate" not in result:
            result["rate"] = column.idx
        elif header in _TOTAL and "line_total" not in result:
            result["line_total"] = column.idx
    return result


def _value(value: str) -> Decimal | None:
    return parse_number(value).value


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
