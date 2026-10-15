from __future__ import annotations

from statistics import median

from ..fuzzy_fields import FuzzyFieldMatcher
from ..schema import Cell, Column, Page, Table, TableRow, Token


class TokenGridTableProvider:
    """Build simple columnar tables directly from OCR token geometry."""

    name = "token_grid"

    def __init__(self, aliases_path, abbreviations_path):
        self.aliases_path = aliases_path
        self.abbreviations_path = abbreviations_path

    def tables(self, page: Page, doc_type: str | None, document_id: str = "document") -> list[Table]:
        matcher = FuzzyFieldMatcher(doc_type, self.aliases_path, self.abbreviations_path) if doc_type else None
        rows = _visual_rows(page.tokens)
        if len(rows) < 2:
            return []
        tables = []
        start = 0
        while start < len(rows) - 1:
            header_index, header_cells = _find_header(rows, matcher, start)
            if header_index is None or len(header_cells) < 2:
                break
            next_header, _ = _find_header(rows, matcher, header_index + 1)
            end = next_header if next_header is not None else len(rows)
            table = _build_table(rows, header_index, end, header_cells, page, document_id)
            if table:
                tables.append(table)
            start = end
        return tables


def _build_table(rows, header_index, end, header_cells, page, document_id):
    columns = _columns(header_cells)
    body = rows[header_index + 1:end]
    logical_rows: list[TableRow] = []
    summary_rows: list[TableRow] = []
    pitch = median([row[0].cy for row in rows[1:]]) - median([row[0].cy for row in rows[:-1]]) if len(rows) > 2 else 0.03
    for visual_index, visual_row in enumerate(body):
            cells = _assign_cells(visual_row, columns, len(logical_rows))
            nonempty = sum(bool(cell.text) for cell in cells)
            if not nonempty:
                continue
            anchor = bool(cells[0].text.strip())
            if logical_rows and not anchor and nonempty <= max(1, len(columns) * 0.4) and visual_index and abs(visual_row[0].cy - body[visual_index - 1][0].cy) <= max(pitch * 1.5, 0.02):
                previous = logical_rows[-1]
                for current, prior in zip(cells, previous.cells):
                    if current.text:
                        prior.text = f"{prior.text} {current.text}".strip()
                        prior.tokens.extend(current.tokens)
                        prior.bbox = _union(prior.bbox, current.bbox)
                previous.continuation_rows += 1
                continue
            row_text = " ".join(cell.text for cell in cells)
            kind = "total_like" if _is_summary_row(row_text) else "item"
            target = summary_rows if kind == "total_like" else logical_rows
            target.append(TableRow(len(target), cells, _row_bbox(cells), kind_hint=kind))
    if not logical_rows:
        return None
    table_bbox = _row_bbox([cell for row in logical_rows for cell in row.cells])
    table_id = f"{document_id}-p{page.page_number}-t{header_index + 1}"
    return Table(table_id, page.page_number, page.page_number, "columnar", "token_grid", columns, logical_rows, table_bbox, conf=0.8, stitched_pages=[page.page_number], reason="token_geometry", summary_rows=summary_rows)


def _visual_rows(tokens: list[Token]) -> list[list[Token]]:
    if not tokens:
        return []
    height = median(token.bbox[3] - token.bbox[1] for token in tokens)
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda item: (item.cy, item.bbox[0])):
        row = next((candidate for candidate in rows if abs(candidate[0].cy - token.cy) <= height * 0.5), None)
        if row is None:
            rows.append([token])
        else:
            row.append(token)
            row.sort(key=lambda item: item.bbox[0])
    return rows


def _find_header(rows: list[list[Token]], matcher: FuzzyFieldMatcher | None, start: int = 0):
    for index in range(start, len(rows) - 1):
        row = rows[index]
        cells = _merge_cells(row)
        if matcher is None and len(cells) >= 2 and len(rows[index + 1]) >= 2:
            return index, cells
        hits = [cell for cell in cells if matcher and matcher.best_score(cell.text) >= 0.8]
        if matcher and len({matcher.match(cell.text).candidates[0].field for cell in hits if matcher.match(cell.text).candidates}) >= 2:
            return index, cells
    return None, []


def _is_summary_row(text: str) -> bool:
    normalized = text.lower().strip()
    return normalized.startswith(("total", "subtotal", "sub total", "ppn", "grand total", "jumlah"))


def _merge_cells(row: list[Token]) -> list[Cell]:
    if not row:
        return []
    height = median(token.bbox[3] - token.bbox[1] for token in row)
    cells: list[Cell] = []
    current: list[Token] = [row[0]]
    for token in row[1:]:
        if token.bbox[0] - current[-1].bbox[2] < height * 1.5:
            current.append(token)
        else:
            cells.append(Cell(" ".join(item.text for item in current), _row_bbox(current), current, is_header=True))
            current = [token]
    cells.append(Cell(" ".join(item.text for item in current), _row_bbox(current), current, is_header=True))
    return cells


def _columns(header_cells: list[Cell]) -> list[Column]:
    ordered = sorted(header_cells, key=lambda cell: cell.bbox[0])
    result = []
    for index, cell in enumerate(ordered):
        left = 0.0 if index == 0 else (ordered[index - 1].bbox[2] + cell.bbox[0]) / 2
        right = 1.0 if index == len(ordered) - 1 else (cell.bbox[2] + ordered[index + 1].bbox[0]) / 2
        result.append(Column(index, left, right, cell.text, [cell.text]))
    return result


def _assign_cells(tokens: list[Token], columns: list[Column], row_idx: int) -> list[Cell]:
    assigned: list[list[Token]] = [[] for _ in columns]
    for token in tokens:
        center = token.cx
        index = next((column.idx for column in columns if column.x1 <= center <= column.x2), min(range(len(columns)), key=lambda item: abs((columns[item].x1 + columns[item].x2) / 2 - center)))
        assigned[index].append(token)
    return [Cell(" ".join(token.text for token in values), _row_bbox(values) if values else (column.x1, tokens[0].bbox[1], column.x2, tokens[0].bbox[3]), values, row_idx, index) for index, (column, values) in enumerate(zip(columns, assigned))]


def _row_bbox(values: list[Token] | list[Cell]) -> tuple[float, float, float, float]:
    boxes = [value.bbox for value in values]
    if not boxes:
        return (0, 0, 0, 0)
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))


def _union(left, right):
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))
