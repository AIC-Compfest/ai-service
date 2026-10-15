from decimal import Decimal

import pytest

from app.services.candidates.schema import Cell, Column, Table, TableRow
from app.services.validation.arithmetic import validate_document_totals, validate_table


def _table(quantity="2", rate="1500", total="3000"):
    columns = [Column(0, 0, .3, "Quantity"), Column(1, .3, .6, "Rate"), Column(2, .6, 1, "Line Total")]
    cells = [Cell(quantity, (0, .1, .3, .2), col_idx=0), Cell(rate, (.3, .1, .6, .2), col_idx=1), Cell(total, (.6, .1, 1, .2), col_idx=2)]
    return Table("t", 1, 1, "columnar", "test", columns, [TableRow(0, cells, (0, .1, 1, .2))], (0, .1, 1, .2))


@pytest.mark.parametrize(("quantity", "rate", "total", "status"), [
    ("2", "1500", "3000", "PASS"), ("2", "1500", "3001", "PASS"),
    ("2", "1500", "3002", "FAIL"), ("1,5", "2000", "3000", "PASS"),
    ("1.000", "2", "2.000", "PASS"), ("2", "1.250,50", "2.501,00", "PASS"),
    ("2", "1500", "2999", "PASS"), ("3", "100", "200", "FAIL"),
])
def test_row_arithmetic_uses_decimal_and_tolerance(quantity, rate, total, status):
    result = validate_table(_table(quantity, rate, total), relative_tolerance=Decimal("0.0001"), absolute_tolerance=Decimal("1"))
    assert result["row_checks"][0]["status"] == status


def test_row_failure_suggests_value_without_mutating_cell():
    table = _table(total="2000")
    result = validate_table(table, relative_tolerance=Decimal("0"), absolute_tolerance=Decimal("0"))
    assert result["row_checks"][0]["suggested_value"] == "3000"
    assert table.rows[0].cells[2].text == "2000"


def test_missing_columns_are_not_reported_as_a_pass():
    table = _table()
    table.columns.pop()
    assert validate_table(table, relative_tolerance=Decimal("0"), absolute_tolerance=Decimal("0"))["reason_codes"] == ["ARITH_COLUMNS_MISSING"]


def test_document_total_supports_tax_and_withholding_tax():
    result = validate_document_totals(line_totals=["1000", "2000"], subtotal="3000", tax="330", withholding_tax="30", grand_total="3300", relative_tolerance=Decimal("0"), absolute_tolerance=Decimal("0"))
    assert [check["status"] for check in result["checks"]] == ["PASS", "PASS"]


def test_document_failure_never_overwrites_observed_total():
    result = validate_document_totals(line_totals=["1000"], subtotal="900", grand_total="1000", relative_tolerance=Decimal("0"), absolute_tolerance=Decimal("0"))
    assert result["checks"][0]["suggested_value"] == "1000"
    assert result["checks"][1]["status"] == "FAIL"
