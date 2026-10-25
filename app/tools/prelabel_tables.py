"""Create review-required table gold drafts from OCR evidence."""

from __future__ import annotations

from typing import Any


def prelabel_tables(document_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Copy extracted tables into a non-reviewed, human-editable gold draft."""
    tables = []
    for page in evidence.get("pages", []):
        for index, table in enumerate(page.get("tables", []), start=1):
            tables.append({
                "table_id": f"{document_id}-p{page.get('page_number', 0)}-t{index}",
                "page_number": page.get("page_number"),
                "headers": table.get("headers", []),
                "rows": table.get("rows", []),
                "source": {"provider": table.get("provider", ""), "reason": table.get("reason", "")},
            })
    return {"document_id": document_id, "reviewed": False, "tables": tables}
