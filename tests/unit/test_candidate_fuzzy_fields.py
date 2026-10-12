from pathlib import Path

from app.services.candidates.fuzzy_fields import FuzzyFieldMatcher


CONFIG = Path("config/candidates")


def test_required_invoice_header_vectors():
    matcher = FuzzyFieldMatcher("INVOICE", CONFIG / "line_item_aliases.yaml", CONFIG / "abbreviations.yaml")
    assert matcher.match("Qty").candidates[0].field == "quantity"
    assert matcher.match("Jml").candidates[0].match_type == "abbrev"
    assert matcher.match("Jumlah").top[:2] == [("quantity", 1.0, "exact"), ("line_total", 1.0, "exact")]
    assert matcher.match("No.").candidates[0].field == "line_number"
    assert matcher.match("No. Resi").candidates[0].field == "shipment_id"
    assert matcher.match("Berat (Kg)").unit_hint == "kg"
    assert matcher.match("Tarif Dasar").candidates[0].field == "base_rate"
    assert matcher.match("PPN 11%").header_pct == 11.0
    assert matcher.match("xyz123").candidates == []


def test_surcharge_label_matching():
    matcher = FuzzyFieldMatcher("RATE_AGREEMENT", CONFIG / "line_item_aliases.yaml", CONFIG / "abbreviations.yaml")
    assert matcher.match_label_text("Fuel Surcharge sebesar", "_surcharge_types").field == "fuel"
    assert matcher.match_label_text("Pajak", "_surcharge_types") is None