import pytest

from app.tools.prelabel_tables import prelabel_tables
from scripts.eval_tables import evaluate


def test_prelabel_creates_non_reviewed_draft():
    draft = prelabel_tables("doc-1", {"pages": [{"page_number": 1, "tables": [{"headers": ["Qty"], "rows": [["2"]], "provider": "token_grid"}]}]})
    assert draft["reviewed"] is False
    assert draft["tables"][0]["rows"] == [["2"]]


def test_evaluation_requires_reviewed_gold_and_scores_rows():
    with pytest.raises(ValueError):
        evaluate({"reviewed": False}, {"tables": []})
    result = evaluate({"reviewed": True, "tables": [{"rows": [["2"], ["3"]]}]}, {"tables": [{"rows": [["2"], ["4"]]}]})
    assert result["matched_rows"] == 1
    assert result["row_precision"] == result["row_recall"] == 0.5
