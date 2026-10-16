"""Coordinate-unit regressions for invoice/rate-agreement layout zones (T-13): bbox_norm is 0..1."""
from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.extraction.invoice import InvoiceExtractor
from app.services.extraction.rate_agreement import RateAgreementExtractor
from app.services.extraction.utils import to_unit_bbox


def _line(text: str, y: float, x: float = 0.1, scale: float = 1.0) -> dict:
    return {"text": text, "bbox_norm": [x * scale, y * scale, min(x + 0.3, 0.99) * scale, (y + 0.02) * scale], "tokens": []}


def _extract(lines: list[dict], tables: list[dict] | None = None) -> dict:
    evidence = {"pages": [{"page_number": 1, "lines": lines, "tables": tables or []}]}
    return {f.field: f for f in InvoiceExtractor().extract("doc", evidence)}


def test_to_unit_bbox_accepts_both_scales():
    assert to_unit_bbox([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
    assert to_unit_bbox([100, 200, 300, 400]) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert to_unit_bbox(None) == [0.0, 0.0, 0.0, 0.0]
    assert to_unit_bbox([0.1, 0.2]) == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_vendor_comes_from_the_header_zone_not_from_any_line(scale):
    lines = [_line("Invoice", 0.02, scale=scale), _line("PT Payee Lama", 0.70, scale=scale), _line("PT Header Vendor", 0.06, scale=scale)]
    assert _extract(lines)["vendor_name"].value == "PT Header Vendor"


def test_vendor_with_dotted_prefix_is_recognised():
    assert _extract([_line("PT. SAMUDRA KARYA", 0.04)])["vendor_name"].value == "PT. SAMUDRA KARYA"


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_awb_fallback_runs_on_unit_scale_pages(scale):
    """Before the fix the fallback band was 200..700 on 0..1 data, so it never ran and items stayed None."""
    lines = [
        _line("INVOICE", 0.02, scale=scale),
        _line("AWB-12345678", 0.40, x=0.05, scale=scale),
        _line("SHP-0001234 REGULAR 12 kg", 0.401, x=0.45, scale=scale),
        _line("Rp 100.000 Rp 20.000 Rp 120.000", 0.402, x=0.70, scale=scale),
        _line("AWB-87654321", 0.50, x=0.05, scale=scale),
        _line("SHP-0007654 EXPRESS 3 kg", 0.501, x=0.45, scale=scale),
        _line("Rp 300.000 Rp 10.000 Rp 310.000", 0.502, x=0.70, scale=scale),
    ]
    items = _extract(lines)["line_items"].value
    assert [item["awb_number"] for item in items] == ["AWB-12345678", "AWB-87654321"]
    assert items[0]["shipment_id"] == "SHP-0001234" and items[0]["chargeable_weight"] == 12.0
    assert items[0]["line_total"] == 120000.0
    assert items[1]["shipment_id"] == "SHP-0007654"  # not mixed with the first row: rows are 0.1 apart


def test_awb_fallback_ignores_lines_outside_the_table_band():
    lines = [_line("INVOICE", 0.02), _line("AWB-12345678", 0.90)]
    assert _extract(lines)["line_items"].value is None


def test_row_grouping_tolerance_is_a_fraction_of_the_page():
    near = settings.invoice_row_y_tolerance / 2
    far = settings.invoice_row_y_tolerance * 3
    lines = [_line("AWB-12345678", 0.40), _line("SHP-0001111", 0.40 + near, x=0.6), _line("SHP-0002222", 0.40 + far, x=0.6)]
    items = _extract(lines)["line_items"].value
    assert items[0]["shipment_id"] == "SHP-0001111"


def test_summary_zone_prefers_lines_below_the_threshold():
    lines = [
        _line("Total Tagihan 999.999", 0.18),  # table header text above the summary zone
        _line("Subtotal 1.000.000", 0.80),
        _line("Total 1.110.000", 0.85),
    ]
    fields = _extract(lines)
    assert fields["subtotal"].value == 1000000.0
    assert fields["grand_total"].value == 1110000.0


def test_rate_agreement_header_zone_is_a_fraction():
    lines = [_line("Rate Agreement", 0.01), _line("PT Footer Pihak Lain", 0.90), _line("PT Header Vendor", 0.05)]
    evidence = {"pages": [{"page_number": 1, "lines": lines, "tables": []}]}
    fields = {f.field: f for f in RateAgreementExtractor().extract("doc", evidence)}
    assert fields["vendor_name"].value == "PT Header Vendor"
