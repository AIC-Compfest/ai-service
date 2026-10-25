from __future__ import annotations

from pathlib import Path

import fitz

from app.services.extraction.surat_jalan import SuratJalanExtractor
from app.services.extraction.label_value import merge_review_fields
from app.services.ocr.docling_engine import DoclingOCREngine


class UnavailableDoclingRunner:
    def convert_page(self, *, pdf_bytes, image_loader):
        return None


def test_full_dataset_return_note_sender_mapping_uses_nearby_text(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    pdf_path = root / "tests" / "fixtures" / "surat_jalan" / "variation_1.pdf"
    pdf_bytes = pdf_path.read_bytes()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        rect = document[0].rect

    monkeypatch.setattr("app.services.ocr.docling_engine.get_runner", UnavailableDoclingRunner)
    evidence_page = DoclingOCREngine().process(
        pdf_bytes,
        is_pdf=True,
        page_number=1,
        quality={},
        page_width=round(rect.width),
        page_height=round(rect.height),
    )
    evidence = {"pages": [evidence_page]}
    legacy_fields = SuratJalanExtractor().extract("TEST-SJ-RETURN-NOTE", evidence)
    fields = {
        field.field: field
        for field in merge_review_fields("TEST-SJ-RETURN-NOTE", evidence, "SURAT_JALAN", legacy_fields,
                                         {"status": "no_model_for_doc_type", "fields": []})
    }

    assert fields["surat_jalan_number"].value == "SDO202508000117"
    assert fields["sender_name"].value == "PT. GEMILANG TIRTA ABADI"
    assert fields["sender_name"].source.bbox is not None
    assert fields["sender_address"].value.startswith("JL. RAYA BEKASI KM. 27")
    assert fields["sender_address"].value != "Disetujui oleh"
    assert fields["shipment_date"].value == "2025-10-12"
    assert fields["shipment_id"].value == "PO-2025-0388"
    assert fields["shipment_id"].review_status == "REVIEW"
    assert fields["receiver_name"].value == "Gudang Rungkut"
    assert fields["receiver_name"].review_status == "REVIEW"
    assert fields["line_items"].value["kind"] == "table"
    assert len(fields["line_items"].value["rows"]) == 2
