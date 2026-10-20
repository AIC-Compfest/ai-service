"""Type-aware table discovery for the document pipeline (K1): after page classification, with the live PDF page.

``discover_page_tables`` runs the shared extractor with the page's document type; ``link_segment_tables`` finds
header-less continuations and stitches tables of one segment that span several pages. Everything is additive
to the evidence contract (``pages[].tables`` keeps its legacy fields).
"""
from __future__ import annotations

from typing import Any

import yaml

from app.core.config import candidates_config_dir, settings

from .context import build_page
from .pipeline import stitch_tables, table_evidence
from .schema import Page, Table
from .tables import TableExtractor
from .tables.structure import continue_table

KNOWN_TYPES = {"INVOICE", "SURAT_JALAN", "POD", "RATE_AGREEMENT"}


def _config() -> dict:
    return yaml.safe_load((candidates_config_dir() / "table_resolution.yaml").read_text(encoding="utf-8")) or {}


def candidate_page(page_dict: dict, source_page: Any = None) -> Page:
    return build_page({**page_dict, "_source_page": source_page})


def discover_page_tables(page: Page, doc_type: str | None, document_id: str) -> list[Table]:
    """Tables of one page for a known (or unknown) document type; empty when the extractor flag is off."""
    if not settings.enable_table_extractor:
        return []
    order = list(settings.table_provider_order)
    if settings.enable_docling and "docling" not in order:
        order.insert(0, "docling")
    extractor = TableExtractor.cached(candidates_config_dir(), order)
    return extractor.extract(page, doc_type if doc_type in KNOWN_TYPES else None, document_id)


def _line_item_tables(tables: list[Table]) -> list[Table]:
    return [t for t in tables if t.role == "line_items"]


def _can_continue(tail: Table, page_tables: list[Table], rules: dict) -> bool:
    """A table may continue on the next page only if it did not end, is a data table and is the lowest table of its page."""
    if tail.summary_rows:
        return False
    if len(tail.rows) < int(rules.get("min_tail_rows", 3)) or len(tail.columns) < int(rules.get("min_tail_columns", 3)):
        return False
    lowest = max((t.bbox[3] for t in page_tables), default=tail.bbox[3])
    return tail.bbox[3] >= lowest - float(rules.get("bottom_tolerance", 0.02))


def link_segment_tables(pages: dict[int, Page], tables: dict[int, list[Table]], segments: list[dict], document_id: str) -> tuple[dict[int, list[Table]], list[dict]]:
    """Add header-less continuation tables and return ``(tables_by_page, table_groups)``.

    Pages are walked in order. A page continues the previous table when the previous table did not end (no summary
    rows), the page has no table of its own at the top and the rows line up with the previous columns. A page whose
    segment is UNKNOWN may continue a table of a known segment, because continuation pages carry no document cues
    and the classifier often rejects them. ``table_groups`` lists logical tables that span more than one page.
    """
    config = _config()
    segment_of = {number: segment for segment in segments for number in range(int(segment["start_page"]), int(segment["end_page"]) + 1)}
    groups: list[dict] = []
    chains: list[tuple[dict, list[Table]]] = []  # (segment of the first page, member tables in page order)
    members: list[Table] = []
    chain_segment: dict | None = None
    for number in sorted(pages):
        segment = segment_of.get(number, {})
        current = _line_item_tables(tables.get(number, []))
        tail = members[-1] if members else None
        same_chain = tail is not None and tail.page_end == number - 1 and chain_segment is not None and (
            segment.get("segment_number") == chain_segment.get("segment_number") or segment.get("document_type") in (None, "UNKNOWN")
        )
        if same_chain and _can_continue(tail, tables.get(tail.page_end, []), config.get("continuation", {})):
            # A header-less table at the top of the page with the same number of columns IS the continuation (header-less
            # discovery finds it before the geometric continuation is tried): give it the columns of the table it continues.
            headerless_top = next((t for t in current if t.bbox[1] < 0.35 and len(t.columns) == len(tail.columns) and not any(c.header_text.strip() for c in t.columns)), None)
            if headerless_top is not None:
                for column, source in zip(headerless_top.columns, tail.columns):
                    column.header_text, column.header_lines = source.header_text, list(source.header_lines)
                headerless_top.reason = "continuation"
                headerless_top.validation["continued_from"] = tail.table_id
                headerless_top.role, headerless_top.role_reason = "line_items", "continuation"
                members.extend(current)
                continue
            has_top_table = any(t.bbox[1] < 0.35 and len(t.columns) == len(tail.columns) for t in current)
            if not has_top_table:
                continued = continue_table(pages[number], tail, config, document_id)
                if continued is not None:
                    continued.role, continued.role_reason = "line_items", "continuation"
                    tables[number] = sorted(tables.get(number, []) + [continued], key=lambda t: (round(t.bbox[1], 3), t.bbox[0]))
                    current = _line_item_tables(tables[number])
        if not same_chain:
            if members:
                chains.append((chain_segment or {}, members))
            members, chain_segment = [], segment
        members.extend(current)
    if members:
        chains.append((chain_segment or {}, members))
    for segment, chain in chains:
        for logical in stitch_tables(chain):
            if logical.page_end > logical.page_start:
                groups.append(
                    {
                        "group_id": f"{document_id}-g{len(groups) + 1}",
                        "segment_number": segment.get("segment_number"),
                        "document_type": segment.get("document_type"),
                        "pages": list(logical.stitched_pages),
                        "table_ids": list(logical.validation.get("stitched_table_ids", [logical.table_id])),
                        "headers": [c.header_text for c in logical.columns],
                        "row_count": len(logical.rows),
                        "rows": [[cell.text for cell in row.cells] for row in logical.rows],
                        "summary_rows": [[cell.text for cell in row.cells] for row in logical.summary_rows],
                        "reason": logical.reason,
                    }
                )
    return tables, groups

def tables_to_evidence(tables: list[Table]) -> list[dict]:
    return [table_evidence(table) for table in tables]
