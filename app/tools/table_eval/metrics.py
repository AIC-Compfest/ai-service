"""Structural and content metrics for table extraction.

Levels (reported separately, never merged into one score):

* detection: greedy IoU matching of tables per page (IoU >= threshold);
* structure: header exact, column count exact, row count exact, ordered row F1 (LCS on full rows);
* content: cell P/R/F1 over ``(column, normalized text)`` multisets (duplicates counted), amount recall;
* separation: summary rows leaking into item rows (F9) or item rows swallowed as summary (F10);
* multi-table pages: every gold table matched and structurally exact, with no extra tables.

Failure taxonomy (counted per page unless noted):
F1 header wrong, F2 column count wrong, F3 row count or row order wrong, F4 one predicted table spans
several gold tables, F5 one gold table split into several predicted tables, F6 extra non-gold table,
F7 page-wide table, F8 gold table not detected, F9 summary row leaked into items, F10 item row classified
as summary, F11 cross-page logical table rows wrong.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from .model import BBox, EvalDoc, EvalTable
from .normalize import canonical_decimal, norm_cell, parse_number

FAILURE_CODES = tuple(f"F{i}" for i in range(1, 12))
AMOUNT_MIN = Decimal("1000")
PAGE_WIDE_AREA = 0.6
OVERLAP_PARTIAL = 0.3
OVERLAP_CONTAINED = 0.6


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def lcs_length(left: list, right: list) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _row_key(row: list[str]) -> tuple[str, ...]:
    values = [norm_cell(cell) for cell in row]
    while values and not values[-1]:
        values.pop()
    return tuple(values)


def _nonempty(row: list[str]) -> list[str]:
    return [value for value in (norm_cell(cell) for cell in row) if value]


def _cells(table: EvalTable) -> Counter:
    counter: Counter = Counter()
    for row in table.rows:
        for index, cell in enumerate(row):
            value = norm_cell(cell)
            if value:
                counter[(index, value)] += 1
    return counter


def _amounts(table: EvalTable) -> list[str]:
    result = []
    for row in table.rows:
        for cell in row:
            parsed = parse_number(cell).value
            if parsed is not None and abs(parsed) >= AMOUNT_MIN:
                result.append(canonical_decimal(parsed))
    return result


def _numbers(tables: list[EvalTable]) -> set[str]:
    found = set()
    for table in tables:
        for row in table.rows + table.summary_rows:
            for cell in row:
                parsed = parse_number(cell).value
                if parsed is not None:
                    found.add(canonical_decimal(parsed))
    return found


@dataclass
class DocEval:
    counts: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    pages: list[dict] = field(default_factory=list)


def select_views(gold: EvalDoc, pred: EvalDoc, view: str) -> tuple[list[EvalTable], list[EvalTable]]:
    """``any``: all tables. ``line_items``: gold role line_items vs predicted role line_items (or all if no roles)."""
    if view == "any":
        return list(gold.tables), list(pred.tables)
    gold_tables = [table for table in gold.tables if (table.role or "line_items") == "line_items"]
    has_roles = any(table.role for table in pred.tables)
    pred_tables = [table for table in pred.tables if table.role == "line_items"] if has_roles else list(pred.tables)
    return gold_tables, pred_tables


def match_tables(gold: list[EvalTable], pred: list[EvalTable], threshold: float) -> list[tuple[EvalTable, EvalTable]]:
    pairs = sorted(
        ((iou(g.bbox, p.bbox), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches = []
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gold[gi], pred[pi]))
    return matches


def evaluate_doc(gold: EvalDoc, pred: EvalDoc, *, view: str = "line_items", iou_threshold: float = 0.5) -> DocEval:
    result = DocEval()
    gold_all, pred_all = select_views(gold, pred, view)
    pages = set(table.page for table in gold_all)
    if gold.pages_exhaustive:
        pages |= set(table.page for table in pred_all)
    for page in sorted(pages):
        gold_tables = [table for table in gold_all if table.page == page]
        pred_tables = [table for table in pred_all if table.page == page]
        record = _evaluate_page(gold_tables, pred_tables, iou_threshold, result.counts)
        record.update(document_id=gold.document_id, page=page)
        for code in record["failures"]:
            result.failures[code] += 1
        result.counts["pages"] += 1
        result.pages.append(record)
    _evaluate_logical(gold_all, pred_all, iou_threshold, result)
    return result


def _evaluate_page(gold: list[EvalTable], pred: list[EvalTable], threshold: float, counts: Counter) -> dict:
    failures: set[str] = set()
    matches = match_tables(gold, pred, threshold)
    matched_g = {id(g) for g, _ in matches}
    matched_p = {id(p) for _, p in matches}
    counts["det_tp"] += len(matches)
    counts["det_fn"] += len(gold) - len(matches)
    counts["det_fp"] += len(pred) - len(matches)
    all_exact = True

    for g, p in matches:
        if g.header_found and g.headers:
            counts["hdr_den"] += 1
            if [norm_cell(h) for h in g.headers] == [norm_cell(h) for h in p.headers]:
                counts["hdr_ok"] += 1
            else:
                failures.add("F1")
                all_exact = False
        counts["col_den"] += 1
        if g.n_cols == p.n_cols:
            counts["col_ok"] += 1
        else:
            failures.add("F2")
            all_exact = False
        counts["row_den"] += 1
        g_keys, p_keys = [_row_key(r) for r in g.rows], [_row_key(r) for r in p.rows]
        lcs = lcs_length(g_keys, p_keys)
        counts["rowseq_tp"] += lcs
        counts["rowseq_gold"] += len(g_keys)
        counts["rowseq_pred"] += len(p_keys)
        if len(g.rows) == len(p.rows):
            counts["row_ok"] += 1
        if len(g.rows) != len(p.rows) or lcs != len(g_keys):
            failures.add("F3")
            all_exact = False
        gc, pc = _cells(g), _cells(p)
        counts["cell_tp"] += sum((gc & pc).values())
        counts["cell_gold"] += sum(gc.values())
        counts["cell_pred"] += sum(pc.values())
        pred_numbers = _numbers([p])
        for amount in _amounts(g):
            counts["amt_gold"] += 1
            counts["amt_in_matched"] += amount in pred_numbers

    for g in gold:
        if id(g) in matched_g:
            continue
        counts["cell_gold"] += sum(_cells(g).values())
        counts["rowseq_gold"] += len(g.rows)
        best = max((intersection(g.bbox, p.bbox) / area(g.bbox) if area(g.bbox) else 0.0 for p in pred), default=0.0)
        if best < OVERLAP_PARTIAL:
            failures.add("F8")
        for _ in _amounts(g):
            counts["amt_gold"] += 1
        all_exact = False
    for p in pred:
        if id(p) in matched_p:
            continue
        counts["cell_pred"] += sum(_cells(p).values())
        counts["rowseq_pred"] += len(p.rows)
        all_exact = False

    # F4: a predicted table overlapping two or more gold tables; F5: a gold table covered by several predictions.
    for p in pred:
        covered = [g for g in gold if area(g.bbox) and intersection(g.bbox, p.bbox) / area(g.bbox) >= OVERLAP_PARTIAL]
        if len(covered) >= 2:
            failures.add("F4")
    for g in gold:
        inside = [p for p in pred if area(p.bbox) and intersection(g.bbox, p.bbox) / area(p.bbox) >= OVERLAP_CONTAINED]
        if len(inside) >= 2:
            failures.add("F5")
    for p in pred:
        if id(p) in matched_p:
            continue
        overlaps = [intersection(g.bbox, p.bbox) / area(p.bbox) if area(p.bbox) else 0.0 for g in gold]
        if area(p.bbox) >= PAGE_WIDE_AREA and not any(area(g.bbox) >= PAGE_WIDE_AREA for g in gold):
            failures.add("F7")
        elif not overlaps or max(overlaps) < OVERLAP_PARTIAL:
            failures.add("F6")
    for p in pred:
        if area(p.bbox) >= PAGE_WIDE_AREA and id(p) in matched_p and not any(area(g.bbox) >= PAGE_WIDE_AREA for g in gold):
            failures.add("F7")

    # Separation between summary and item rows.
    pred_item_sets = [frozenset(_nonempty(row)) for p in pred for row in p.rows]
    pred_summary_sets = [frozenset(_nonempty(row)) for p in pred for row in p.summary_rows]
    for g in gold:
        for row in g.summary_rows:
            cells = frozenset(_nonempty(row))
            if not cells:
                continue
            counts["sum_gold"] += 1
            if any(cells <= item for item in pred_item_sets):
                counts["sum_leak"] += 1
                failures.add("F9")
        for row in g.rows:
            cells = frozenset(_nonempty(row))
            if cells and any(cells <= summary for summary in pred_summary_sets):
                counts["item_as_summary"] += 1
                failures.add("F10")

    pred_numbers_page = _numbers(pred)
    for g in gold:
        for amount in _amounts(g):
            counts["amt_on_page"] += amount in pred_numbers_page

    if len(gold) >= 2:
        counts["multi_pages"] += 1
        if all_exact and len(matches) == len(gold) and len(pred) == len(gold):
            counts["multi_ok"] += 1
    counts["gold_tables"] += len(gold)
    counts["pred_tables"] += len(pred)
    return {"failures": sorted(failures), "gold_tables": len(gold), "pred_tables": len(pred), "matched": len(matches)}


def _evaluate_logical(gold: list[EvalTable], pred: list[EvalTable], threshold: float, result: DocEval) -> None:
    groups: dict[str, list[EvalTable]] = {}
    for table in gold:
        if table.logical_id:
            groups.setdefault(table.logical_id, []).append(table)
    for parts in groups.values():
        if len(parts) < 2:
            continue
        result.counts["logical_den"] += 1
        matched_rows = 0
        matched_parts = 0
        for part in parts:
            candidates = [p for p in pred if p.page == part.page]
            matches = match_tables([part], candidates, threshold)
            if matches:
                matched_parts += 1
                matched_rows += len(matches[0][1].rows)
        if matched_parts == len(parts) and matched_rows == sum(len(part.rows) for part in parts):
            result.counts["logical_ok"] += 1
        else:
            result.failures["F11"] += 1


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 6) if den else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision is not None and recall is not None else None
    return round(2 * precision * recall / (precision + recall), 6)


def summarize(counts: Counter, failures: Counter | None = None) -> dict:
    """Turn summed counters into reported metrics with numerators and denominators."""
    c = counts
    det_p = _rate(c["det_tp"], c["det_tp"] + c["det_fp"])
    det_r = _rate(c["det_tp"], c["det_tp"] + c["det_fn"])
    cell_p, cell_r = _rate(c["cell_tp"], c["cell_pred"]), _rate(c["cell_tp"], c["cell_gold"])
    row_p, row_r = _rate(c["rowseq_tp"], c["rowseq_pred"]), _rate(c["rowseq_tp"], c["rowseq_gold"])

    def ratio(num_key: str, den_key: str) -> dict:
        return {"num": c[num_key], "den": c[den_key], "rate": _rate(c[num_key], c[den_key])}

    return {
        "pages": c["pages"],
        "gold_tables": c["gold_tables"],
        "pred_tables": c["pred_tables"],
        "detection": {"tp": c["det_tp"], "fp": c["det_fp"], "fn": c["det_fn"], "precision": det_p, "recall": det_r, "f1": _f1(det_p, det_r)},
        "header_exact": ratio("hdr_ok", "hdr_den"),
        "column_count_exact": ratio("col_ok", "col_den"),
        "row_count_exact": ratio("row_ok", "row_den"),
        "row_sequence": {"tp": c["rowseq_tp"], "pred": c["rowseq_pred"], "gold": c["rowseq_gold"], "precision": row_p, "recall": row_r, "f1": _f1(row_p, row_r)},
        "cell": {"tp": c["cell_tp"], "pred": c["cell_pred"], "gold": c["cell_gold"], "precision": cell_p, "recall": cell_r, "f1": _f1(cell_p, cell_r)},
        "amount_recall_in_matched_table": ratio("amt_in_matched", "amt_gold"),
        "amount_recall_on_page": ratio("amt_on_page", "amt_gold"),
        "summary_leak": {"leaked": c["sum_leak"], "summary_rows": c["sum_gold"], "separation_rate": None if not c["sum_gold"] else round(1 - c["sum_leak"] / c["sum_gold"], 6)},
        "item_rows_as_summary": c["item_as_summary"],
        "multi_table_pages_fully_correct": ratio("multi_ok", "multi_pages"),
        "logical_tables_rows_exact": ratio("logical_ok", "logical_den"),
        "failures_pages": {code: (failures or {}).get(code, 0) for code in FAILURE_CODES},
    }


def evaluate_row_content(gt_tables: list[list[list[str]]], pred: EvalDoc, *, coverage: float = 0.8) -> Counter:
    """Content-only evaluation against GT-derived rows (no structure): which GT rows appear as predicted rows.

    ``gt_tables`` is a list of tables, each a list of rows, each a list of scalar value strings.
    A predicted row covers a GT row when at least ``coverage`` of the GT values are present in it.
    """
    pred_tables = [table for table in pred.tables if table.role == "line_items"] if any(t.role for t in pred.tables) else list(pred.tables)
    pred_rows = [row for table in pred_tables for row in table.rows]
    pred_info = []
    for row in pred_rows:
        norms = [norm_cell(cell) for cell in row if norm_cell(cell)]
        pred_info.append((set(norms), " ".join(norms)))
    gt_rows = [row for table in gt_tables for row in table]
    scores = []
    for gi, values in enumerate(gt_rows):
        keys = [norm_cell(value) for value in values if norm_cell(value)]
        if not keys:
            continue
        for pi, (cell_set, joined) in enumerate(pred_info):
            hits = sum(1 for key in keys if key in cell_set or (len(key) >= 3 and not key.startswith("#") and key in joined))
            fraction = hits / len(keys)
            if fraction >= coverage:
                scores.append((fraction, gi, pi))
    scores.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, gi, pi in scores:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    counts: Counter = Counter()
    counts["rc_gt_rows"] = sum(1 for row in gt_rows if any(norm_cell(v) for v in row))
    counts["rc_pred_rows"] = len(pred_rows)
    counts["rc_covered"] = len(used_g)
    counts["rc_docs"] = 1
    counts["rc_count_exact"] = int(len(pred_rows) == counts["rc_gt_rows"])
    return counts


def summarize_row_content(counts: Counter) -> dict:
    precision = _rate(counts["rc_covered"], counts["rc_pred_rows"])
    recall = _rate(counts["rc_covered"], counts["rc_gt_rows"])
    return {
        "documents": counts["rc_docs"],
        "gt_rows": counts["rc_gt_rows"],
        "pred_rows": counts["rc_pred_rows"],
        "covered_rows": counts["rc_covered"],
        "row_precision": precision,
        "row_recall": recall,
        "row_f1": _f1(precision, recall),
        "docs_row_count_exact": {"num": counts["rc_count_exact"], "den": counts["rc_docs"], "rate": _rate(counts["rc_count_exact"], counts["rc_docs"])},
    }
