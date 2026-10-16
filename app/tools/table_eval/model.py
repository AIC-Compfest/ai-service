"""Evaluation-side table model, gold schema loading (GOLD_REVIEW) and prediction adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BBox = tuple[float, float, float, float]


class GoldRejected(ValueError):
    """Raised when a gold file violates the review contract."""


@dataclass
class EvalTable:
    table_id: str
    page: int
    bbox: BBox
    role: str | None = None
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    summary_rows: list[list[str]] = field(default_factory=list)
    logical_id: str | None = None
    provider: str | None = None
    header_found: bool = True

    @property
    def n_cols(self) -> int:
        if self.headers:
            return len(self.headers)
        return max((len(row) for row in self.rows), default=0)


@dataclass
class EvalDoc:
    document_id: str
    tables: list[EvalTable] = field(default_factory=list)
    pages_exhaustive: bool = True
    source_label: str = ""
    reviewed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def on_page(self, page: int) -> list[EvalTable]:
        return [table for table in self.tables if table.page == page]

    def pages(self) -> set[int]:
        return {table.page for table in self.tables}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_review(gold: dict, *, allow_unreviewed: bool) -> None:
    """Enforce the GOLD_REVIEW contract: reviewed=true needs two distinct reviewers."""
    reviewed = bool(gold.get("reviewed"))
    if not reviewed:
        if not allow_unreviewed:
            raise GoldRejected("gold is not reviewed (reviewed=false); pass allow_unreviewed=True to evaluate drafts")
        return
    review = gold.get("review") or {}
    prepared, approved = str(review.get("prepared_by", "")).strip(), str(review.get("approved_by", "")).strip()
    if not prepared or not approved:
        raise GoldRejected("reviewed gold requires review.prepared_by and review.approved_by")
    if prepared == approved:
        raise GoldRejected("a reviewer must not approve their own draft (prepared_by == approved_by)")


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, dict):
        return str(cell.get("raw_text", ""))
    return str(cell)


def _row_cells(row: Any) -> list[str]:
    cells = row.get("cells") if isinstance(row, dict) else row
    if isinstance(cells, list) and cells and isinstance(cells[0], dict) and "column_index" in cells[0]:
        cells = sorted(cells, key=lambda item: item["column_index"])
    return [_cell_text(cell) for cell in (cells or [])]


def gold_from_json(gold: dict, *, allow_unreviewed: bool = False) -> EvalDoc:
    """Load a GOLD_REVIEW document (``pages[].tables[].rows[].cells[]``)."""
    validate_review(gold, allow_unreviewed=allow_unreviewed)
    doc = EvalDoc(
        document_id=str(gold.get("document_id", "")),
        pages_exhaustive=bool(gold.get("pages_exhaustive", True)),
        source_label=str(gold.get("source_label", "")),
        reviewed=bool(gold.get("reviewed")),
        meta={"document_type": gold.get("document_type"), "source": gold.get("source", {})},
    )
    for page in gold.get("pages", []):
        for table in page.get("tables", []):
            rows = [_row_cells(row) for row in table.get("rows", []) if row.get("kind", "item") == "item"]
            summary = [_row_cells(row) for row in table.get("summary_rows", [])]
            headers = [_cell_text(header) for header in table.get("headers", [])]
            doc.tables.append(
                EvalTable(
                    table_id=str(table.get("table_id", "")),
                    page=int(page.get("page_number", 1)),
                    bbox=tuple(table.get("bbox_norm", (0.0, 0.0, 1.0, 1.0))),  # type: ignore[arg-type]
                    role=table.get("role", "line_items"),
                    headers=headers,
                    rows=rows,
                    summary_rows=summary,
                    logical_id=table.get("logical_id"),
                    header_found=bool(table.get("quality", {}).get("header_found", True)),
                )
            )
    return doc


def evidence_from_gold(gold: dict) -> dict:
    """Oracle prediction in pipeline-evidence shape (``pages[].tables[]``); used by tests and sanity checks."""
    pages = []
    for page in gold.get("pages", []):
        tables = []
        for table in page.get("tables", []):
            tables.append(
                {
                    "bbox_norm": list(table["bbox_norm"]),
                    "headers": [_cell_text(header) for header in table.get("headers", [])],
                    "rows": [_row_cells(row) for row in table.get("rows", [])],
                    "summary_rows": [_row_cells(row) for row in table.get("summary_rows", [])],
                    "role": table.get("role", "line_items"),
                    "provider": "oracle",
                }
            )
        pages.append({"page_number": page["page_number"], "tables": tables})
    return {"document_id": gold.get("document_id", ""), "pages": pages}


def prediction_from_evidence(evidence: dict, document_id: str | None = None) -> EvalDoc:
    """Adapt pipeline evidence (``pages[].tables[]``) or a flat ``{"tables": [...]}`` draft."""
    doc = EvalDoc(document_id=document_id or str(evidence.get("document_id", "")))
    raw_tables: list[tuple[int, dict]] = []
    for page in evidence.get("pages", []) or []:
        for table in page.get("tables", []) or []:
            raw_tables.append((int(page.get("page_number", 1)), table))
    for table in evidence.get("tables", []) or []:
        raw_tables.append((int(table.get("page_number", 1)), table))
    for index, (page_number, table) in enumerate(raw_tables, start=1):
        doc.tables.append(
            EvalTable(
                table_id=str(table.get("table_id", f"{doc.document_id}-p{page_number}-t{index}")),
                page=page_number,
                bbox=tuple(table.get("bbox_norm", (0.0, 0.0, 1.0, 1.0))),  # type: ignore[arg-type]
                role=table.get("role"),
                headers=[str(value) for value in table.get("headers", [])],
                rows=[[str(value) for value in row] for row in table.get("rows", [])],
                summary_rows=[[str(value) for value in row] for row in table.get("summary_rows", [])],
                logical_id=None,
                provider=table.get("provider"),
            )
        )
    return doc
