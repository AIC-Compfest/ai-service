"""Guard on the stress set: varied PDF pages with 1 to 4 tables each and exact gold (seeded, deterministic)."""
from __future__ import annotations

from collections import Counter

import pytest

from app.tools.table_eval.metrics import evaluate_doc, summarize
from app.tools.table_eval.model import gold_from_json, prediction_from_evidence
from app.tools.table_eval.runner import predict_fixture
from app.tools.table_eval.stress import build_stress, build_stress_page

SEEDS = 80  # a window that is NOT the one the rules were tuned on (tuning used seeds 0..239; the final check 1000..1239)
FIRST = 3000


@pytest.fixture(scope="module")
def results():
    pages = build_stress(SEEDS, FIRST)
    out: dict[str, dict] = {"router_none": {}, "pymupdf": {}}
    for config in out:
        per_tag: dict[str, list] = {}
        for page in pages:
            gold = gold_from_json(page.fixture.gold, allow_unreviewed=True)
            prediction = predict_fixture(page.fixture.pdf, config, "vector", f"constructed/{page.fixture.name}", 0)
            result = evaluate_doc(gold, prediction_from_evidence(prediction, gold.document_id), view="line_items")
            clean = not any(p["failures"] for p in result.pages)
            for tag in ["all"] + page.tags:
                entry = per_tag.setdefault(tag, [Counter(), Counter(), 0, 0])
                entry[0].update(result.counts)
                entry[1].update(result.failures)
                entry[2] += 1
                entry[3] += int(clean)
        out[config] = {tag: (summarize(c, f), n, ok) for tag, (c, f, n, ok) in per_tag.items()}
    return out


def _rate(item) -> float:
    return item["num"] / item["den"] if item["den"] else 1.0


def test_the_generator_is_deterministic_and_the_gold_matches_the_drawing():
    first, second = build_stress_page(7), build_stress_page(7)
    assert first.fixture.gold["pages"] == second.fixture.gold["pages"] and first.tags == second.tags  # (PDF bytes embed an id, so compare the gold)
    assert any(t["role"] == "line_items" for t in first.fixture.gold["pages"][0]["tables"])


def test_the_router_reads_most_varied_pages_completely(results):
    summary, pages, clean = results["router_none"]["all"]
    assert clean / pages >= 0.75  # measured 0.87 on unseen seeds; the first version of this work was at 0.52
    assert summary["detection"]["f1"] >= 0.90
    assert _rate(summary["header_exact"]) >= 0.95 and _rate(summary["column_count_exact"]) >= 0.97 and _rate(summary["row_count_exact"]) >= 0.95


@pytest.mark.parametrize("tag", ["grouped_header", "headerless", "unruled", "ruled", "summary_rows", "wrapped_header", "wrapped_rows", "tables=3+"])
def test_each_layout_feature_is_covered(results, tag):
    summary, pages, clean = results["router_none"][tag]
    assert pages >= 5 and summary["detection"]["recall"] >= 0.75 and clean / pages >= 0.55


def test_group_labels_reach_every_column_they_span(results):
    summary, _, _ = results["router_none"]["grouped_header"]
    assert _rate(summary["header_exact"]) >= 0.90


def test_pymupdf_alone_misses_unruled_tables_and_that_is_the_documented_limit(results):
    summary, pages, _ = results["pymupdf"]["unruled"]
    assert summary["detection"]["recall"] == 0.0  # find_tables(strategy="lines") needs drawn rules


def test_pymupdf_ruled_tables_keep_summary_rows_out_of_the_items(results):
    summary, _, _ = results["pymupdf"]["ruled"]
    leak = summary["summary_leak"]
    assert leak["leaked"] <= 0.25 * max(leak["summary_rows"], 1)
    assert _rate(summary["row_count_exact"]) >= 0.60  # was 0.40 before the clean-up
