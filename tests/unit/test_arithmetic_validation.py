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
    assert result["checks"][0]["expected"] == "1000"
    assert result["checks"][0]["suggested_value"] is None  # 900 vs 1000 differ by two characters: no correction is offered
    assert result["checks"][1]["status"] == "FAIL"


def test_suggestion_only_for_a_one_character_slip():
    slip = validate_document_totals(line_totals=["1000"], subtotal="1900", relative_tolerance=Decimal("0"), absolute_tolerance=Decimal("0"))
    assert slip["checks"][0]["suggested_value"] == "1000"  # 1900 -> 1000 is one digit


def test_tolerance_is_absolute_per_currency_and_has_no_relative_default():
    idr = validate_document_totals(line_totals=["1000000"], subtotal="1000002", currency="IDR")
    usd = validate_document_totals(line_totals=["100.00"], subtotal="100.02", currency="USD")
    assert idr["checks"][0]["status"] == "FAIL"  # 2 IDR off; a 1% relative tolerance would have hidden it
    assert usd["checks"][0]["status"] == "FAIL"  # 0.02 USD off; limit is 0.01
    assert validate_document_totals(line_totals=["100.00"], subtotal="100.01", currency="USD")["checks"][0]["status"] == "PASS"


def test_missing_line_totals_skip_the_subtotal_check_instead_of_failing_it():
    result = validate_document_totals(line_totals=[None, "abc"], subtotal="500")
    assert result["checks"][0]["status"] == "SKIPPED" and not result["available"]


@pytest.mark.parametrize(("headers", "row", "rule", "status"), [
    (["No", "Item", "Qty", "Unit Price", "Amount"], ["1", "x", "2", "1.500", "3.000"], "qty_x_rate", "PASS"),
    (["Deskripsi", "Berat", "Tarif", "Total"], ["x", "10", "5.000", "50.000"], "weight_x_rate", "PASS"),
    (["Item", "Gross Amount", "Discount", "Net Amount"], ["x", "1.000", "100", "900"], "gross_minus_discount", "PASS"),
    (["Item", "Gross Amount", "Discount", "Net Amount"], ["x", "1.000", "100", "800"], "gross_minus_discount", "FAIL"),
    (["Nama", "Jumlah", "Harga Satuan", "Jumlah"], ["x", "3", "1.000", "3.000"], "qty_x_rate", "PASS"),
    (["Nama", "Jumlah", "Harga Satuan"], ["x", "3", "1.000"], None, None),
])
def test_rules_come_from_the_registry_and_resolve_headers(headers, row, rule, status):
    from app.services.validation.arithmetic import validate_rows

    result = validate_rows(headers, [row], currency="IDR")
    if rule is None:
        assert not result["available"] and result["reason_codes"] == ["ARITH_COLUMNS_MISSING"]
    else:
        (check,) = [c for c in result["row_checks"] if c["rule"] == rule]
        assert check["status"] == status


def test_short_rows_and_empty_cells_are_skipped_not_crashed():
    from app.services.validation.arithmetic import validate_rows

    result = validate_rows(["Qty", "Rate", "Total"], [["2"], ["", "100", "200"], ["2", "100", "200"]])
    assert [c["status"] for c in result["row_checks"]] == ["SKIPPED", "SKIPPED", "PASS"]
    assert result["rows_total"] == 3 and result["rows_evaluated"] == 1

def test_percent_columns_never_take_a_money_role_and_discounted_rows_skip_the_plain_product():
    from app.services.validation.arithmetic import assign_roles, validate_rows

    headers = ["No.", "Qty", "Unit Price", "Gross Amt.", "Discount Detail %", "Discount Detail Amount", "Net Amount"]
    assert assign_roles(headers)["discount"] == 5  # not the % column
    result = validate_rows(headers, [["1", "2", "1.000", "2.000", "10", "200", "1.800"]])
    assert result["rules"] == ["qty_x_rate_gross", "gross_minus_discount"]
    assert all(check["status"] == "PASS" for check in result["row_checks"])
    # a plain qty * price rule must not fire on a row that has a discount or tax column (the total includes them)
    plain = validate_rows(["Produk", "Qty", "Harga", "Disc", "Pajak", "Jumlah"], [["x", "2", "1.000", "100", "190", "2.090"]])
    assert not plain["available"]
