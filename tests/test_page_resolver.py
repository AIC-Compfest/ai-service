from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField
from app.services.extraction.page_resolver import resolve_field_pages


def _field(name, value, bbox):
    return ExtractedField(
        field=name,
        value=value,
        confidence=0.9,
        source=EvidenceSource(document_id="d", page=1, bbox=bbox),
        method="rule_based",
    )


def _page(no, lines):
    return {"page_number": no, "lines": [{"text": t, "bbox_norm": b} for t, b in lines], "tables": []}


def test_field_is_attributed_to_page_containing_its_bbox_and_text():
    # Same geometry on two pages: the text value disambiguates.
    box = [0.1, 0.1, 0.4, 0.15]
    pages = [_page(1, [("INV-1", box)]), _page(2, [("SJ-9", box)]), _page(3, [("OTHER", [0.5, 0.5, 0.6, 0.6])])]
    f_sj = _field("surat_jalan_number", "SJ-9", BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.15))
    f_other = _field("x", "OTHER", BoundingBox(x1=0.5, y1=0.5, x2=0.6, y2=0.6))
    resolve_field_pages([f_sj, f_other], pages)
    assert f_sj.source.page == 2
    assert f_other.source.page == 3


def test_field_without_bbox_uses_default_page():
    f = _field("a", "v", None)
    resolve_field_pages([f], [_page(1, [])], default_page=4)
    assert f.source.page == 4


def test_unmatched_bbox_falls_back_to_default_page():
    f = _field("a", "zzz", BoundingBox(x1=0.8, y1=0.8, x2=0.9, y2=0.9))
    resolve_field_pages([f], [_page(2, [("t", [0.1, 0.1, 0.2, 0.2])])], default_page=2)
    assert f.source.page == 2
