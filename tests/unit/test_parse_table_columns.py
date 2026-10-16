"""Header-to-field mapping in parse_table_columns (T-15)."""
from __future__ import annotations

from app.services.extraction.invoice import INVOICE_TABLE_COLUMNS
from app.services.extraction.utils import parse_table_columns


def _row(headers: list[str], values: list[str]) -> dict:
    return parse_table_columns([{"headers": headers, "rows": [values]}], INVOICE_TABLE_COLUMNS)[0]


def _filled(row: dict) -> dict:
    return {key: value for key, value in row.items() if value is not None}


def test_empty_header_maps_to_nothing():
    """Before: '' in alias is True, so a blank header column fed ten fields at once."""
    assert _filled(_row(["", "Qty", "Total"], ["x", "2", "10"])) == {"line_total": "10"}


def test_blank_and_whitespace_headers_are_ignored():
    assert _filled(_row(["   ", None, ""], ["a", "b", "c"])) == {}


def test_substring_inside_a_word_does_not_match():
    # 'no' must not match 'Notes'/'Nomor Seri'; 'kg' must not match 'Kgs Ekstra'-like words by substring only.
    assert _filled(_row(["Notes", "Nomor Seri"], ["a", "b"])) == {}


def test_exact_and_phrase_headers_map():
    row = _row(["No", "AWB No.", "Origin Terminal", "Weight (Kg)", "Rate", "Total"], ["1", "AWB-1", "CGK", "12", "1000", "12000"])
    assert row["line_number"] == "1"
    assert row["awb_number"] == "AWB-1"
    assert row["origin"] == "CGK"
    assert row["chargeable_weight"] == "12"
    assert row["base_rate"] == "1000"
    assert row["line_total"] == "12000"


def test_longest_matching_alias_wins_for_a_column():
    row = _row(["No Resi"], ["SHP-001"])
    assert row["shipment_id"] == "SHP-001"
    assert row["line_number"] is None


def test_one_column_feeds_at_most_one_field():
    row = _row(["Total", "Qty"], ["10", "2"])
    assert sum(value is not None for value in row.values()) == 1


def test_first_column_wins_a_tie_instead_of_the_last():
    row = _row(["Jumlah", "Total"], ["first", "second"])
    assert row["line_total"] == "first"


def test_short_alias_requires_whole_header():
    assert _row(["Resi No"], ["SHP-001"])["line_number"] is None
    assert _row(["no."], ["7"])["line_number"] == "7"


def test_rows_shorter_than_headers_do_not_crash():
    row = parse_table_columns([{"headers": ["No", "Total"], "rows": [["1"]]}], INVOICE_TABLE_COLUMNS)[0]
    assert row["line_number"] == "1" and row["line_total"] is None
