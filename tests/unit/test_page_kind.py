import io

import fitz
from PIL import Image

from app.core.config import settings
from app.services.ocr.engine import OCREngine
from app.services.preprocessing.page_kind import PageKind, detect_page_kind


def _png_bytes(width: int = 200, height: int = 200) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _page(*, image_rect: fitz.Rect | None = None, text: str = "", invisible: bool = False):
    document = fitz.open()
    page = document.new_page(width=200, height=200)
    if image_rect is not None:
        page.insert_image(image_rect, stream=_png_bytes())
    if text:
        page.insert_text((10, 20), text, fontsize=12, render_mode=3 if invisible else 0)
    return document, page


def test_page_kind_deterministic_pdf_cases():
    digital_doc, digital = _page(text="invoice number customer address total quantity rate amount")
    scan_doc, scan = _page(image_rect=fitz.Rect(0, 0, 200, 200))
    hybrid_doc, hybrid = _page(
        image_rect=fitz.Rect(0, 0, 200, 200),
        text="invoice number customer address total quantity rate amount",
        invisible=True,
    )
    mixed_doc, mixed = _page(
        image_rect=fitz.Rect(0, 0, 180, 180),
        text="invoice number customer address total quantity rate amount",
    )
    logo_doc, logo = _page(
        image_rect=fitz.Rect(0, 0, 30, 30),
        text="invoice number customer address total quantity rate amount",
    )
    footer_doc, footer = _page(text="page 1")
    try:
        assert detect_page_kind(digital, word_coverage_min=0.01).kind == PageKind.DIGITAL
        assert detect_page_kind(scan).kind == PageKind.SCAN
        assert detect_page_kind(hybrid, word_coverage_min=0.01).kind == PageKind.HYBRID_OCR_LAYER
        assert detect_page_kind(mixed, word_coverage_min=0.01).kind == PageKind.MIXED
        assert detect_page_kind(logo, word_coverage_min=0.01).kind == PageKind.DIGITAL
        assert detect_page_kind(footer, word_coverage_min=0.01).kind == PageKind.SCAN
    finally:
        for document in (digital_doc, scan_doc, hybrid_doc, mixed_doc, logo_doc, footer_doc):
            document.close()


def test_page_kind_is_added_to_evidence_and_routes_scan(monkeypatch):
    document, page = _page(image_rect=fitz.Rect(0, 0, 200, 200))
    engine = OCREngine()
    monkeypatch.setattr(settings, "page_kind_routing", True)
    calls: list[bytes] = []

    def fake_process_image(image_bytes, page_number, quality, **_options):
        calls.append(image_bytes)
        from app.services.ocr.engine import PageOCREvidence
        return PageOCREvidence(page_number, 200, 200, "fake", True, [], [], [], quality)

    monkeypatch.setattr(engine, "process_image", fake_process_image)
    try:
        evidence = engine.process_pdf_page(page, 1, {})
    finally:
        document.close()

    payload = evidence.to_dict()
    assert calls
    assert payload["page_kind"] == PageKind.SCAN
    assert payload["page_kind_signals"]["has_full_page_image"] is True
    assert "page_kind_reason" in payload
