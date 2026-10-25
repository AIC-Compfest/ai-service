"""Document-level arithmetic evidence for the extraction response (additive; never changes an extracted value)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.config import settings
from app.schemas.evidence import ExtractedField
from app.services.candidates.parsers.numbers import parse_number

from .arithmetic import rules_config, validate_document_totals, validate_rows

CAVEAT = (
    "The extractor derives grand_total or tax_amount from the other two when one is missing (OS-008); "
    "a grand_total check on such a document is circular and proves nothing. The subtotal check uses table line totals and is independent. "
    "Document-level discounts and withholding tax are not modelled, so a FAIL on a document that has them can be legitimate."
)
MAX_FAILURES_LISTED = 20


def validate_extraction(fields: list[ExtractedField], pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Row rules on every ``line_items`` table of ``pages`` plus subtotal/grand-total checks against the extracted fields."""
    if not settings.enable_arithmetic_evidence:
        return {"status": "disabled"}
    document = rules_config().get("document", {})
    by_name = {f.field: f for f in fields}

    def value(key: str) -> Any:
        field = by_name.get(document.get(key, ""))
        return None if field is None or field.is_unknown else field.value

    currency = value("currency_field")
    currency = str(currency).upper() if currency else None
    tables, line_totals, failures = [], [], []
    rows_total = rows_evaluated = 0
    for page in pages:
        for table in page.get("tables", []):
            if table.get("role") != "line_items":
                continue
            headers, rows = table.get("headers", []), table.get("rows", [])
            result = validate_rows(headers, rows, currency=currency)
            roles = result["roles"]
            if "line_total" in roles:
                column = roles["line_total"]
                line_totals.extend(_number(row[column]) if column < len(row) else None for row in rows)
            rows_total += result["rows_total"]
            rows_evaluated += result["rows_evaluated"]
            tables.append({
                "table_id": table.get("table_id"), "page": page.get("page_number"), "available": result["available"], "rules": result["rules"], "roles": roles,
                "rows_total": result["rows_total"], "rows_evaluated": result["rows_evaluated"], "passed": result.get("passed", 0), "evaluated": result.get("evaluated", 0),
            })
            failures.extend({**check, "table_id": table.get("table_id"), "page": page.get("page_number")} for check in result["row_checks"] if check["status"] == "FAIL")
    totals = validate_document_totals(line_totals=line_totals, subtotal=value("subtotal_field"), tax=value("tax_field"), grand_total=value("grand_total_field"), currency=currency)
    if not tables and not totals["available"]:
        return {"status": "not_applicable", "currency": currency, "coverage": {"tables": 0, "rows_total": 0, "rows_evaluated": 0}}
    return {
        "status": "ok", "currency": currency, "tables": tables, "totals": totals, "failed_rows": failures[:MAX_FAILURES_LISTED], "failed_rows_total": len(failures),
        "coverage": {"tables": len(tables), "tables_with_rules": sum(t["available"] for t in tables), "rows_total": rows_total, "rows_evaluated": rows_evaluated},
        "caveats": [CAVEAT],
    }


def _number(text: Any) -> Decimal | None:
    return None if text is None else parse_number(str(text)).value
