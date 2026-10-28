from app.schemas.evidence import EvidenceSource, ExtractedField, ExtractionMethod
from app.services.extraction.label_value import extract_label_values, merge_review_fields


def _line(text, bbox):
    return {"text": text, "bbox_norm": bbox, "confidence": 0.99}


def test_configured_labels_extract_bounded_values_and_address_continuation():
    evidence = {"pages": [{"page_number": 1, "lines": [
        _line("12/10/25, 14:36 Sales Return DO", [0.0, 0.01, 0.9, 0.03]),
        _line("No Surat Jalan : SDO202508000117", [0.04, 0.2, 0.8, 0.22]),
        _line("No Pesanan : PO-2025-0388", [0.04, 0.23, 0.8, 0.25]),
        _line("Tanggal : 12-10-2025 14:20", [0.04, 0.26, 0.8, 0.28]),
        _line("Dari : PT. GEMILANG TIRTA ABADI", [0.04, 0.30, 0.7, 0.32]),
        _line("JL. RAYA BEKASI KM. 27 KAWASAN INDUSTRI", [0.23, 0.33, 0.8, 0.35]),
        _line("Status : Dikembalikan", [0.04, 0.38, 0.7, 0.40]),
        _line("Dikembalikan ke : Gudang Rungkut", [0.04, 0.42, 0.8, 0.44]),
        _line("(................)", [0.7, 0.8, 0.9, 0.82]),
    ]}]}

    fields = {field.field: field for field in extract_label_values("doc", evidence, "SURAT_JALAN")}

    assert fields["surat_jalan_number"].value == "SDO202508000117"
    assert fields["shipment_id"].value == "PO-2025-0388"
    assert fields["shipment_date"].value == "2025-10-12"
    assert fields["sender_name"].value == "PT. GEMILANG TIRTA ABADI"
    assert "JL. RAYA BEKASI KM. 27" in fields["sender_address"].value
    assert fields["receiver_name"].value == "Gudang Rungkut"
    assert fields["receiver_address"].is_unknown
    assert all(field.source.bbox is None or 0 <= field.source.bbox.x1 <= field.source.bbox.x2 <= 1 for field in fields.values())
    assert all(field.review_status in {"ACCEPTED", "REVIEW", "UNKNOWN"} for field in fields.values())


def test_table_field_uses_selected_table_rows_not_header_text():
    table = {"table_id": "p1-t1", "role": "line_items", "source": "docling", "bbox_norm": [0.1, 0.4, 0.9, 0.7],
             "headers": ["Kode Barang", "Nama Barang", "Jumlah Diretur"],
             "rows": [["0000000003412", "Epson L3210 Printer", "2.00"], ["0000000003587", "Logitech Mouse M170", "5.00"]]}
    fields = {field.field: field for field in extract_label_values("doc", {"pages": [{"page_number": 1, "lines": [], "tables": [table]}]}, "SURAT_JALAN")}
    value = fields["line_items"].value
    assert value["kind"] == "table"
    assert value["headers"] == table["headers"]
    assert [row["cells"] for row in value["rows"]] == table["rows"]
    assert value["rows"][0]["values"]["item_description"] == "Epson L3210 Printer"


def test_table_field_recovers_positioned_text_grid_when_layout_provider_has_no_table():
    lines = [
        _line("Kode Barang", [0.04, 0.40, 0.18, 0.42]),
        _line("Nama Barang", [0.33, 0.40, 0.48, 0.42]),
        _line("Jumlah Diretur", [0.78, 0.40, 0.94, 0.42]),
        _line("0000000003412", [0.02, 0.43, 0.17, 0.45]),
        _line("Epson L3210 Printer", [0.29, 0.43, 0.52, 0.45]),
        _line("2.00", [0.85, 0.43, 0.91, 0.45]),
        _line("0000000003587", [0.02, 0.46, 0.17, 0.48]),
        _line("Logitech Mouse M170", [0.29, 0.46, 0.54, 0.48]),
        _line("5.00", [0.85, 0.46, 0.91, 0.48]),
    ]
    evidence = {"pages": [{"page_number": 1, "lines": lines, "tables": []}]}

    fields = {field.field: field for field in extract_label_values("doc", evidence, "SURAT_JALAN")}
    payload = fields["line_items"].value

    assert payload["headers"] == ["Kode Barang", "Nama Barang", "Jumlah Diretur"]
    assert [row["cells"] for row in payload["rows"]] == [
        ["0000000003412", "Epson L3210 Printer", "2.00"],
        ["0000000003587", "Logitech Mouse M170", "5.00"],
    ]
    assert evidence["pages"][0]["tables"][0]["source"] == "pymupdf_positioned_text"


def test_review_merge_keeps_typed_extractor_value_when_label_policy_is_unknown():
    # The invoice typed extractor can identify a value with geometry/regex even
    # when the label policy finds no matching label. That value must remain
    # reviewable instead of being replaced by a synthetic UNKNOWN mapping.
    evidence = {"pages": [{"page_number": 1, "lines": []}]}
    typed = ExtractedField(
        field="invoice_number", value="INV-2026-001", confidence=0.92,
        source=EvidenceSource(document_id="doc", page=1),
        method=ExtractionMethod.RULE_BASED, is_unknown=False, raw_text="INV-2026-001",
    )
    fields = {item.field: item for item in merge_review_fields("doc", evidence, "INVOICE", [typed], {"fields": []})}
    assert fields["invoice_number"].value == "INV-2026-001"
    assert not fields["invoice_number"].is_unknown
