"""Unit tests for the table evaluation toolkit (normalization, gold contract, metrics, constructed fixtures)."""
from __future__ import annotations

import copy
from collections import Counter
from decimal import Decimal

import pymupdf
import pytest

from app.tools.table_eval.metrics import evaluate_doc, evaluate_row_content, summarize
from app.tools.table_eval.model import GoldRejected, evidence_from_gold, gold_from_json, prediction_from_evidence
from app.tools.table_eval.normalize import norm_cell, parse_number
from app.tools.table_eval.synthetic import FIXTURE_BUILDERS, MULTI_TABLE_FIXTURES, build_fixture, to_evidence_page


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.000.000", "1000000"),
        ("1,000,000", "1000000"),
        ("1000000", "1000000"),
        ("Rp 1.250.000,50", "1250000.5"),
        ("Rp. 12.500.000", "12500000"),
        ("1.250,50", "1250.5"),
        ("1,250.50", "1250.5"),
        ("1.250", "1250"),
        ("1,250", "1250"),
        ("0.250", "0.25"),
        ("20,5", "20.5"),
        ("12.50", "12.5"),
        ("(1.000)", "-1000"),
        ("$ 3,425.00", "3425"),
        ("US$3.425,00", "3425"),
        ("IDR\u00a02.500.000", "2500000"),
    ],
)
def test_parse_number_locales(text, expected):
    parsed = parse_number(text)
    assert parsed.value == Decimal(expected)


@pytest.mark.parametrize("text", ["", "abc", "12 kg", "1.2.3", "Total", None])
def test_parse_number_rejects_non_numbers(text):
    assert parse_number(text).value is None


def test_parse_number_flags_ambiguous_thousands():
    assert parse_number("1.250").ambiguous is True
    assert parse_number("1.250.000").ambiguous is False


def test_norm_cell_numbers_and_text():
    assert norm_cell("Rp 1.000.000") == norm_cell("1,000,000") == "#1000000"
    assert norm_cell("  Nama\u00a0 Barang ") == "nama barang"
    assert norm_cell("12 kg") == "12 kg"


def _gold(name: str) -> dict:
    return build_fixture(name).gold


def test_all_fixtures_build_deterministically():
    assert len(FIXTURE_BUILDERS) >= 15
    first, second = build_fixture("side_by_side"), build_fixture("side_by_side")
    assert first.gold == second.gold
    assert first.pdf and first.page_count == 1
    for name in MULTI_TABLE_FIXTURES:
        assert any(len(page["tables"]) >= 2 for page in build_fixture(name).gold["pages"]), name


def test_fixture_words_exist_in_drawn_page():
    fixture = build_fixture("single_ruled")
    document = pymupdf.open(stream=fixture.pdf, filetype="pdf")
    text = document[0].get_text()
    for row in fixture.gold["pages"][0]["tables"][0]["rows"]:
        for cell in row["cells"]:
            assert cell["raw_text"].split()[0] in text


def test_phrase_mode_is_deterministic_and_estimated():
    document = pymupdf.open(stream=build_fixture("single_unruled").pdf, filetype="pdf")
    first = to_evidence_page(document[0], 1, "phrase", seed=3)
    second = to_evidence_page(document[0], 1, "phrase", seed=3)
    assert first == second
    assert any(token["_estimated"] for line in first["lines"] for token in line["tokens"])
    exact = to_evidence_page(document[0], 1, "vector")
    assert not any(token["_estimated"] for line in exact["lines"] for token in line["tokens"])


@pytest.mark.parametrize("name", sorted(FIXTURE_BUILDERS))
def test_oracle_prediction_scores_perfect(name):
    gold = _gold(name)
    doc = gold_from_json(gold, allow_unreviewed=True)
    pred = prediction_from_evidence(evidence_from_gold(gold))
    for view in ("any", "line_items"):
        result = evaluate_doc(doc, pred, view=view)
        summary = summarize(result.counts, result.failures)
        assert not result.failures, (name, view, dict(result.failures))
        if summary["gold_tables"]:
            assert summary["detection"]["f1"] == 1.0
            assert summary["cell"]["f1"] == 1.0
            assert summary["row_sequence"]["f1"] == 1.0


def test_unreviewed_gold_is_rejected_by_default():
    with pytest.raises(GoldRejected):
        gold_from_json(_gold("single_ruled"))


def test_reviewed_gold_needs_two_distinct_reviewers():
    gold = _gold("single_ruled")
    gold["reviewed"] = True
    gold["review"] = {"prepared_by": "a", "approved_by": ""}
    with pytest.raises(GoldRejected):
        gold_from_json(gold)
    gold["review"] = {"prepared_by": "a", "approved_by": "a"}
    with pytest.raises(GoldRejected):
        gold_from_json(gold)
    gold["review"] = {"prepared_by": "a", "approved_by": "b"}
    assert gold_from_json(gold).reviewed is True


def test_pipeline_evidence_shape_is_not_silently_zero():
    """Regression for the old evaluator, which read prediction["tables"] and returned 0 without error."""
    gold = _gold("single_ruled")
    evidence = evidence_from_gold(gold)
    assert "tables" not in evidence  # pages[].tables only, as the real pipeline emits
    pred = prediction_from_evidence(evidence)
    assert sum(len(page.get("tables", [])) for page in evidence["pages"]) == len(pred.tables) > 0


def test_duplicate_rows_are_counted():
    gold = _gold("single_ruled")
    gold_dup = copy.deepcopy(gold)
    table = gold_dup["pages"][0]["tables"][0]
    table["rows"] = [table["rows"][0], copy.deepcopy(table["rows"][0])]
    doc = gold_from_json(gold_dup, allow_unreviewed=True)
    evidence = evidence_from_gold(gold_dup)
    evidence["pages"][0]["tables"][0]["rows"] = evidence["pages"][0]["tables"][0]["rows"][:1]
    result = evaluate_doc(doc, prediction_from_evidence(evidence))
    summary = summarize(result.counts, result.failures)
    assert summary["row_sequence"]["recall"] == 0.5
    assert summary["row_count_exact"]["rate"] == 0.0
    assert result.failures["F3"] == 1


def test_missing_table_is_f8_and_extra_table_is_f6():
    gold = _gold("single_ruled")
    doc = gold_from_json(gold, allow_unreviewed=True)
    empty = prediction_from_evidence({"pages": [{"page_number": 1, "tables": []}]})
    assert evaluate_doc(doc, empty).failures["F8"] == 1
    evidence = evidence_from_gold(gold)
    evidence["pages"][0]["tables"].append({"bbox_norm": [0.1, 0.9, 0.3, 0.95], "headers": ["a", "b"], "rows": [["1", "2"]], "role": "line_items"})
    result = evaluate_doc(doc, prediction_from_evidence(evidence))
    assert result.failures["F6"] == 1 and not result.failures["F8"]


def test_merged_prediction_is_f4_and_split_prediction_is_f5():
    gold = _gold("stacked_two_tables")
    doc = gold_from_json(gold, allow_unreviewed=True)
    tables = [t for t in evidence_from_gold(gold)["pages"][0]["tables"]]
    a, b = tables[0]["bbox_norm"], tables[1]["bbox_norm"]
    merged = {"pages": [{"page_number": 1, "tables": [{"bbox_norm": [min(a[0], b[0]), a[1], max(a[2], b[2]), b[3]], "headers": tables[0]["headers"], "rows": tables[0]["rows"] + tables[1]["rows"], "role": "line_items"}]}]}
    assert evaluate_doc(doc, prediction_from_evidence(merged)).failures["F4"] == 1
    top = tables[0]
    mid = (top["bbox_norm"][1] + top["bbox_norm"][3]) / 2
    halves = [dict(top, bbox_norm=[top["bbox_norm"][0], top["bbox_norm"][1], top["bbox_norm"][2], mid], rows=top["rows"][:1]), dict(top, bbox_norm=[top["bbox_norm"][0], mid, top["bbox_norm"][2], top["bbox_norm"][3]], rows=top["rows"][1:])]
    split = {"pages": [{"page_number": 1, "tables": halves + [tables[1]]}]}
    assert evaluate_doc(doc, prediction_from_evidence(split)).failures["F5"] == 1


def test_page_wide_table_is_f7():
    doc = gold_from_json(_gold("paragraph_only"), allow_unreviewed=True)
    wide = {"pages": [{"page_number": 1, "tables": [{"bbox_norm": [0.05, 0.05, 0.95, 0.95], "headers": ["x", "y"], "rows": [["a", "b"]]}]}]}
    result = evaluate_doc(doc, prediction_from_evidence(wide))
    assert result.failures["F7"] == 1


def test_summary_leak_f9_and_item_as_summary_f10():
    gold = _gold("items_total_rows_inside")
    doc = gold_from_json(gold, allow_unreviewed=True)
    evidence = evidence_from_gold(gold)
    table = evidence["pages"][0]["tables"][0]
    leaked = copy.deepcopy(evidence)
    leaked["pages"][0]["tables"][0]["rows"] = table["rows"] + table["summary_rows"]
    leaked["pages"][0]["tables"][0]["summary_rows"] = []
    assert evaluate_doc(doc, prediction_from_evidence(leaked)).failures["F9"] == 1
    swallowed = copy.deepcopy(evidence)
    swallowed["pages"][0]["tables"][0]["summary_rows"] = table["summary_rows"] + [table["rows"][-1]]
    swallowed["pages"][0]["tables"][0]["rows"] = table["rows"][:-1]
    assert evaluate_doc(doc, prediction_from_evidence(swallowed)).failures["F10"] == 1


def test_wrong_header_is_f1_and_wrong_column_count_is_f2():
    gold = _gold("single_ruled")
    doc = gold_from_json(gold, allow_unreviewed=True)
    evidence = evidence_from_gold(gold)
    wrong_header = copy.deepcopy(evidence)
    wrong_header["pages"][0]["tables"][0]["headers"][1] = "Barang"
    assert evaluate_doc(doc, prediction_from_evidence(wrong_header)).failures["F1"] == 1
    fewer = copy.deepcopy(evidence)
    fewer["pages"][0]["tables"][0]["headers"] = fewer["pages"][0]["tables"][0]["headers"][:4]
    assert evaluate_doc(doc, prediction_from_evidence(fewer)).failures["F2"] == 1


def test_multi_table_page_requires_every_table_exact():
    gold = _gold("side_by_side")
    doc = gold_from_json(gold, allow_unreviewed=True)
    evidence = evidence_from_gold(gold)
    ok = summarize(evaluate_doc(doc, prediction_from_evidence(evidence)).counts)
    assert ok["multi_table_pages_fully_correct"]["rate"] == 1.0
    broken = copy.deepcopy(evidence)
    broken["pages"][0]["tables"].pop()
    bad = summarize(evaluate_doc(doc, prediction_from_evidence(broken)).counts)
    assert bad["multi_table_pages_fully_correct"]["rate"] == 0.0


def test_cross_page_logical_table_f11():
    gold = _gold("multipage_repeated_header")
    doc = gold_from_json(gold, allow_unreviewed=True)
    evidence = evidence_from_gold(gold)
    assert not evaluate_doc(doc, prediction_from_evidence(evidence)).failures["F11"]
    broken = copy.deepcopy(evidence)
    broken["pages"][1]["tables"] = []
    assert evaluate_doc(doc, prediction_from_evidence(broken)).failures["F11"] == 1


def test_amount_recall_and_zero_denominator_is_none():
    gold = _gold("single_ruled")
    doc = gold_from_json(gold, allow_unreviewed=True)
    summary = summarize(evaluate_doc(doc, prediction_from_evidence(evidence_from_gold(gold))).counts)
    assert summary["amount_recall_in_matched_table"]["rate"] == 1.0
    assert summarize(Counter())["header_exact"]["rate"] is None


def test_row_content_coverage():
    pred = prediction_from_evidence({"pages": [{"page_number": 1, "tables": [{"bbox_norm": [0, 0, 1, 1], "headers": ["a", "b", "c"], "rows": [["1", "Pupuk Organik Granul", "200 karung"], ["2", "Benih Padi", "120 pak"], ["x", "noise", "row"]]}]}]})
    counts = evaluate_row_content([[["Pupuk Organik", "200 karung"], ["Benih Padi", "120 pak"], ["Garam", "5 kg"]]], pred)
    assert counts["rc_gt_rows"] == 3 and counts["rc_covered"] == 2 and counts["rc_pred_rows"] == 3
