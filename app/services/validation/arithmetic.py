"""Decimal-only arithmetic validation, driven by ``config/validation/arithmetic.yaml``.

Rules never alter source values. A corrected value (``suggested_value``) is offered only when it differs from the
observed value by one character (typical OCR slip); otherwise the expected value is reported as ``expected`` only.
Tolerances are absolute per currency (K3); a relative tolerance exists only if the YAML or the caller sets one.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import yaml

from app.core.config import settings
from app.services.candidates.parsers.numbers import parse_number
from app.services.candidates.schema import Table
from app.services.candidates.text_norm import norm_text

_PERCENT = {"persen", "percent", "pct"}
_TIE_ORDER = ("line_total", "net", "gross", "discount", "quantity", "weight", "rate")


@lru_cache(maxsize=2)
def _load(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def rules_config() -> dict[str, Any]:
    return _load(str(Path(settings.config_dir) / "validation" / "arithmetic.yaml"))


def tolerances(currency: str | None, relative: Decimal | None = None, absolute: Decimal | None = None) -> tuple[Decimal, Decimal]:
    """(relative, absolute) tolerance; explicit arguments win over the YAML."""
    config = rules_config().get("tolerance", {})
    table = config.get("absolute", {})
    resolved_abs = absolute if absolute is not None else Decimal(str(table.get((currency or "").upper(), table.get("default", "1"))))
    resolved_rel = relative if relative is not None else Decimal(str(config.get("relative", "0")))
    return resolved_rel, resolved_abs


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------

def assign_roles(headers: Sequence[str]) -> dict[str, int]:
    """Map arithmetic roles to column indices from header texts (one column per role)."""
    config = rules_config()
    aliases = {role: [tuple(norm_text(a).split()) for a in values] for role, values in config.get("roles", {}).items()}
    ambiguous = {norm_text(key): value for key, value in config.get("ambiguous", {}).items()}
    found: dict[str, list[int]] = {}
    ambiguous_columns: list[int] = []
    for index, header in enumerate(headers):
        text = norm_text(header or "")
        tokens = set(text.split())
        if not tokens or "%" in (header or "") or tokens & _PERCENT:
            continue
        if text in ambiguous:
            ambiguous_columns.append(index)
            continue
        best_rank: tuple[int, int, int] | None = None
        best_role = ""
        for role, options in aliases.items():
            for alias in options:
                if alias and set(alias) <= tokens:
                    # longest alias first, then an exact header, then the role order (e.g. "Total Harga" is a total, not a rate)
                    rank = (len(alias), 1 if tuple(text.split()) == alias else 0, -_TIE_ORDER.index(role) if role in _TIE_ORDER else -99)
                    if best_rank is None or rank > best_rank:
                        best_rank, best_role = rank, role
        if best_rank is not None:
            found.setdefault(best_role, []).append((best_rank, index))
    roles = {role: _pick(role, candidates) for role, candidates in found.items()}
    if ambiguous_columns:
        ambiguous_columns.sort()
        if len(ambiguous_columns) > 1:
            roles.setdefault("quantity", ambiguous_columns[0])
            roles.setdefault("line_total", ambiguous_columns[-1])
        else:
            column = ambiguous_columns[0]
            if ("rate" in roles or "weight" in roles) and "line_total" not in roles:
                roles["line_total"] = column
            elif "quantity" not in roles:
                roles["quantity"] = column
            elif "line_total" not in roles:
                roles["line_total"] = column
    return roles


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _pick(role: str, candidates: list[tuple[tuple[int, int, int], int]]) -> int:
    """Best-matching column of a role (longest alias, exact header); ties: rightmost for totals, else leftmost."""
    top = max(rank[:2] for rank, _ in candidates)
    columns = sorted(index for rank, index in candidates if rank[:2] == top)
    return columns[-1] if role in {"line_total", "net", "gross"} else columns[0]


def validate_rows(headers: Sequence[str], rows: Sequence[Sequence[str]], *, currency: str | None = None, relative_tolerance: Decimal | None = None, absolute_tolerance: Decimal | None = None) -> dict[str, Any]:
    """Apply every row rule whose roles exist in ``headers`` to ``rows`` (lists of cell texts)."""
    relative, absolute = tolerances(currency, relative_tolerance, absolute_tolerance)
    roles = assign_roles(headers)
    applicable = [rule for rule in rules_config().get("row_rules", []) if rule["target"] in roles and all(term in roles for term in rule["terms"]) and not any(role in roles for role in rule.get("unless", []))]
    if not applicable:
        return {"available": False, "reason_codes": ["ARITH_COLUMNS_MISSING"], "row_checks": [], "roles": roles, "rules": [], "rows_total": len(rows), "rows_evaluated": 0}
    checks: list[dict[str, Any]] = []
    for row_index, cells in enumerate(rows):
        for rule in applicable:
            columns = [roles[term] for term in rule["terms"]]
            target_col = roles[rule["target"]]
            if max(columns + [target_col]) >= len(cells):
                checks.append({"row": row_index, "rule": rule["id"], "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"})
                continue
            terms = [_value(cells[c]) for c in columns]
            observed = _value(cells[target_col])
            if observed is None or any(term is None for term in terms):
                checks.append({"row": row_index, "rule": rule["id"], "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"})
                continue
            expected = terms[0] * terms[1] if rule["op"] == "product" else terms[0] - terms[1]
            checks.append(_row_check(row_index, rule["id"], expected, observed, cells[target_col], relative, absolute))
    evaluated = [c for c in checks if c["status"] != "SKIPPED"]
    passed = sum(c["status"] == "PASS" for c in evaluated)
    evaluated_rows = {c["row"] for c in evaluated}
    return {
        "available": bool(evaluated), "reason_codes": [c["reason_code"] for c in checks], "row_checks": checks, "passed": passed, "evaluated": len(evaluated),
        "score": Decimal(passed) / Decimal(len(evaluated)) if evaluated else Decimal("0"), "roles": roles, "rules": [r["id"] for r in applicable],
        "rows_total": len(rows), "rows_evaluated": len(evaluated_rows),
    }


def validate_table(table: Table, *, relative_tolerance: Decimal | None = None, absolute_tolerance: Decimal | None = None, currency: str | None = None) -> dict[str, Any]:
    """Validate a ``Table`` (cells are read, never modified)."""
    headers = [column.header_text for column in table.columns]
    rows = []
    for row in table.rows:
        by_col = {cell.col_idx: cell.text for cell in row.cells}
        rows.append([by_col.get(index, "") for index in range(len(headers))])
    result = validate_rows(headers, rows, currency=currency, relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance)
    for check in result["row_checks"]:
        check["row"] = table.rows[check["row"]].idx
    return result


def _row_check(row: int, rule: str, expected: Decimal, observed: Decimal, observed_text: str, relative: Decimal, absolute: Decimal) -> dict[str, Any]:
    delta = abs(expected - observed)
    passed = delta <= max(absolute, abs(expected) * relative)
    check = {"row": row, "rule": rule, "status": "PASS" if passed else "FAIL", "reason_code": "ARITH_ROW_OK" if passed else "ARITH_ROW_FAIL", "expected": str(expected), "observed": str(observed), "delta": str(delta)}
    check["suggested_value"] = None if passed else _suggest(expected, observed_text)
    return check


# ---------------------------------------------------------------------------
# document level
# ---------------------------------------------------------------------------

def validate_document_totals(*, line_totals: list[Any], subtotal: Any = None, tax: Any = None, withholding_tax: Any = None, grand_total: Any = None, relative_tolerance: Decimal | None = None, absolute_tolerance: Decimal | None = None, currency: str | None = None) -> dict[str, Any]:
    """Validate observed document totals. Missing values are never synthesized."""
    relative, absolute = tolerances(currency, relative_tolerance, absolute_tolerance)
    observed_lines = [_decimal(value) for value in line_totals]
    present = [value for value in observed_lines if value is not None]
    line_sum = sum(present, Decimal("0"))
    checks = []
    if subtotal is not None:
        if present:
            checks.append(_check("subtotal", line_sum, _decimal(subtotal), relative, absolute, "ARITH_DOC_SUBTOTAL_FAIL"))
        else:
            checks.append({"field": "subtotal", "status": "SKIPPED", "reason_code": "ARITH_NO_LINE_TOTALS"})
    if grand_total is not None and subtotal is not None:
        expected = _decimal(subtotal)
        if expected is None:
            checks.append({"field": "grand_total", "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"})
        else:
            for value, sign in ((tax, 1), (withholding_tax, -1)):
                parsed = _decimal(value)
                if parsed is not None:
                    expected += sign * parsed
            checks.append(_check("grand_total", expected, _decimal(grand_total), relative, absolute, "ARITH_DOC_TOTAL_FAIL"))
    evaluated = [c for c in checks if c["status"] != "SKIPPED"]
    return {"available": bool(evaluated), "line_total_sum": str(line_sum), "line_totals_present": len(present), "line_totals_given": len(observed_lines), "checks": checks, "reason_codes": [c["reason_code"] for c in checks]}


def _check(name: str, expected: Decimal, observed: Decimal | None, relative: Decimal, absolute: Decimal, fail_code: str) -> dict[str, Any]:
    if observed is None:
        return {"field": name, "status": "SKIPPED", "reason_code": "ARITH_VALUE_MISSING"}
    delta = abs(expected - observed)
    passed = delta <= max(absolute, abs(expected) * relative)
    return {"field": name, "status": "PASS" if passed else "FAIL", "reason_code": "ARITH_DOC_OK" if passed else fail_code, "expected": str(expected), "observed": str(observed), "delta": str(delta), "suggested_value": None if passed else _suggest(expected, str(observed))}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _suggest(expected: Decimal, observed_text: str) -> str | None:
    """The expected value, but only when it is one character away from what was read (digits compared)."""
    limit = int(rules_config().get("suggestion", {}).get("max_edit_distance", 1))
    exp_digits = re.sub(r"\D", "", format(expected.normalize(), "f"))
    obs_digits = re.sub(r"\D", "", observed_text or "")
    if not exp_digits or not obs_digits or _edit_distance(exp_digits, obs_digits, limit) > limit:
        return None
    return format(expected.normalize(), "f")


def _edit_distance(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _value(value: str) -> Decimal | None:
    if value is None:
        return None
    return parse_number(str(value)).value


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
