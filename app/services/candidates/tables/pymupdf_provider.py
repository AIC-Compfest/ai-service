"""PyMuPDF table provider for selectable-text PDF pages."""

from __future__ import annotations

from app.services.candidates.schema import Cell, Column, Page, Table, TableRow


class PyMuPDFTableProvider:
    """Normalize ``Page.find_tables`` results into the internal Table model."""

    name = "pymupdf_find_tables"

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        source = page.source_page
        if source is None or not hasattr(source, "find_tables"):
            return []
        candidates = []
        for strategy in ("lines", "text"):
            try:
                candidates = source.find_tables(strategy=strategy).tables
            except Exception:
                continue
            if candidates:
                break
        return _normalize(candidates, page, document_id)


def _normalize(found, page: Page, document_id: str) -> list[Table]:
    width, height = float(page.source_page.rect.width), float(page.source_page.rect.height)
    output, seen = [], set()
    for found_table in found:
        x1, y1, x2, y2 = found_table.bbox
        key = tuple(round(value, 1) for value in (x1, y1, x2, y2))
        if key in seen:
            continue
        seen.add(key)
        extracted = [[value or "" for value in row] for row in found_table.extract()]
        headers = [value or "" for value in getattr(found_table.header, "names", [])]
        if headers and extracted and extracted[0] == headers:
            extracted = extracted[1:]
        if len(headers) < 2 or not extracted:
            continue
        table_bbox = _norm_bbox(x1, y1, x2, y2, width, height)
        columns = _columns(headers, table_bbox[0], table_bbox[2])
        rows = _rows(extracted, columns, found_table.cells, width, height, table_bbox[1], table_bbox[3])
        if not rows:
            continue
        table_id = f"{document_id}-p{page.page_number}-t{len(output) + 1}"
        output.append(Table(table_id, page.page_number, page.page_number, "grid", "pymupdf_find_tables", columns, rows, table_bbox, conf=0.0, stitched_pages=[page.page_number], reason="pymupdf_find_tables"))
    return output


def _columns(headers, x1, x2):
    width = (x2 - x1) / len(headers)
    return [Column(index, x1 + index * width, x1 + (index + 1) * width, header, [header]) for index, header in enumerate(headers)]


def _rows(extracted, columns, cells, width, height, y1, y2):
    result = []
    for row_idx, values in enumerate(extracted):
        row_cells = []
        for col_idx, column in enumerate(columns):
            bbox = _cell_bbox(cells, row_idx + 1, col_idx, len(columns), width, height, column, y1, y2)
            row_cells.append(Cell(values[col_idx] if col_idx < len(values) else "", bbox, row_idx=row_idx, col_idx=col_idx))
        if any(cell.text.strip() for cell in row_cells):
            result.append(TableRow(len(result), row_cells, (min(cell.bbox[0] for cell in row_cells), min(cell.bbox[1] for cell in row_cells), max(cell.bbox[2] for cell in row_cells), max(cell.bbox[3] for cell in row_cells))))
    return result


def _cell_bbox(cells, row_idx, col_idx, columns, width, height, column, y1, y2):
    index = row_idx * columns + col_idx
    if index < len(cells) and cells[index] is not None:
        x1, y1_px, x2, y2_px = cells[index]
        return _norm_bbox(x1, y1_px, x2, y2_px, width, height)
    return (column.x1, y1, column.x2, y2)


def _norm_bbox(x1, y1, x2, y2, width, height):
    """Clip provider geometry to the established normalized page bounds."""
    left, right = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    top, bottom = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return (left / width, top / height, right / width, bottom / height)
