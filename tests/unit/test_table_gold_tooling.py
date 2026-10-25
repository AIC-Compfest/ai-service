import pytest

from app.tools.prelabel_tables import prelabel_tables
from app.tools.table_eval.model import GoldRejected
from scripts.eval_tables import evaluate

PAGE_EVIDENCE = {"pages": [{"page_number": 1, "tables": [{"bbox_norm": [0.1, 0.1, 0.9, 0.3], "headers": ["Qty"], "rows": [["2"], ["4"]]}]}]}
FLAT_GOLD = {"reviewed": True, "review": {"prepared_by": "a", "approved_by": "b"}, "tables": [{"page_number": 1, "bbox_norm": [0.1, 0.1, 0.9, 0.3], "headers": ["Qty"], "rows": [["2"], ["3"]]}]}


def test_prelabel_creates_non_reviewed_draft():
    draft = prelabel_tables("doc-1", {"pages": [{"page_number": 1, "tables": [{"headers": ["Qty"], "rows": [["2"]], "provider": "token_grid"}]}]})
    assert draft["reviewed"] is False
    assert draft["tables"][0]["rows"] == [["2"]]


def test_evaluation_requires_reviewed_gold():
    with pytest.raises(GoldRejected):
        evaluate({"reviewed": False, "tables": []}, PAGE_EVIDENCE)


def test_evaluation_reads_pipeline_evidence_pages_not_root_tables():
    """The old evaluator read prediction["tables"] and silently returned zero for real evidence."""
    result = evaluate(FLAT_GOLD, PAGE_EVIDENCE)
    assert result["pred_tables"] == 1
    assert result["row_sequence"]["tp"] == 1
    assert result["row_sequence"]["pred"] == 2 and result["row_sequence"]["gold"] == 2


def test_unreviewed_draft_is_stamped_when_explicitly_allowed():
    draft = prelabel_tables("doc-1", PAGE_EVIDENCE)
    result = evaluate(draft, PAGE_EVIDENCE, allow_unreviewed=True)
    assert result["reviewed"] is False and "bukan gold manusia" in result["stamp"]
