from app.services.extraction.invoice import _is_invoice_item_row


def test_invoice_line_item_filter_is_not_vendor_specific():
    assert not _is_invoice_item_row({"shipment_id": "Logistik", "line_total": None})
    assert not _is_invoice_item_row({"shipment_id": "Any Vendor", "line_total": None})
    assert _is_invoice_item_row({"shipment_id": "SHP-12345", "line_total": None})
    assert _is_invoice_item_row({"shipment_id": "", "line_total": "Rp 125.000"})
