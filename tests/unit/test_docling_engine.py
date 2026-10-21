"""DoclingOCREngine: conversion of a Docling result into the page-evidence contract. Docling itself is faked."""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace as NS

import pytest

from app.services.docling import runner as runner_module
from app.services.docling.runner import DoclingResult
from app.services.ocr.docling_engine import DoclingOCREngine, _norm_bbox

W, H = 600.0, 800.0


class Origin(Enum):
    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"


def box(l, t, r, b, origin=Origin.TOPLEFT):
    return NS(l=l, t=t, r=r, b=b, coord_origin=origin)


def cell(text, l, t, r, b, confidence=1.0, origin=Origin.TOPLEFT):
    bbox = box(l, t, r, b, origin)
    return NS(text=text, confidence=confidence, rect=NS(to_bounding_box=lambda: bbox))


class FakeRunner:
    def __init__(self, vector=None, ocr=None):
        self.vector, self.ocr, self.calls = vector, ocr, []

    def convert(self, payload, *, is_pdf, ocr):
        self.calls.append((is_pdf, ocr))
        return self.ocr if ocr else self.vector

    def convert_page(self, *, pdf_bytes, image_loader):
        self.calls.append(("page", pdf_bytes is not None))
        if self.vector is not None and self.vector.has_text:
            return self.vector
        return self.ocr


def result(lines, words=(), ocr=False, texts=()):
    return DoclingResult([], list(texts), W, H, ocr, list(lines), list(words))


def run(monkeypatch, fake, **kwargs):
    monkeypatch.setattr("app.services.ocr.docling_engine.get_runner", lambda: fake)
    return DoclingOCREngine().process(b"x", page_number=3, quality={"q": 1}, page_width=1654, page_height=2339, **kwargs)


def test_pdf_page_uses_vector_text_and_keeps_the_contract(monkeypatch):
    lines = [cell("Total 100", 60, 400, 300, 420), cell("INVOICE", 60, 40, 200, 60)]
    words = [cell("INVOICE", 60, 40, 200, 60), cell("Total", 60, 400, 150, 420), cell("100", 160, 400, 300, 420)]
    page = run(monkeypatch, FakeRunner(vector=result(lines, words)), is_pdf=True)
    assert (page["engine_used"], page["fallback_used"], page["page_number"]) == ("docling_vector", False, 3)
    assert (page["width"], page["height"], page["coord_space"], page["tables"]) == (1654, 2339, "page_image_normalized", [])
    assert [l["text"] for l in page["lines"]] == ["INVOICE", "Total 100"]  # reading order
    total = page["lines"][1]
    assert total["bbox_norm"] == [0.1, 0.5, 0.5, 0.525]
    assert [t["text"] for t in total["tokens"]] == ["Total", "100"] and not any(t["_estimated"] for t in total["tokens"])
    assert all(0 <= v <= 1 for l in page["lines"] for v in l["bbox_norm"])
    assert {r["region_type"] for r in page["regions"]} >= {"header", "table"}
    assert page["quality"] == {"q": 1} and page["average_confidence"] == 1.0


def test_image_page_runs_ocr_with_estimated_tokens(monkeypatch):
    fake = FakeRunner(ocr=result([cell("Qty 2 Rate", 60, 100, 360, 120, 0.9)], ocr=True))
    page = run(monkeypatch, fake, is_pdf=False)
    assert fake.calls == [(False, True)]
    assert (page["engine_used"], page["fallback_used"]) == ("docling_rapidocr", False)
    tokens = page["lines"][0]["tokens"]
    assert [t["text"] for t in tokens] == ["Qty", "2", "Rate"] and all(t["_estimated"] for t in tokens)
    assert tokens[0]["bbox_norm"][0] == 0.1 and tokens[-1]["bbox_norm"][2] == 0.6


def test_textless_pdf_page_falls_back_to_ocr(monkeypatch):
    fake = FakeRunner(vector=result([]), ocr=result([cell("Scan", 10, 10, 100, 30)], ocr=True))
    page = run(monkeypatch, fake, is_pdf=True, fallback_image_bytes=b"png")
    assert (page["engine_used"], page["fallback_used"]) == ("docling_rapidocr", True)


def test_tokens_in_the_same_visual_row_share_row_idx(monkeypatch):
    lines = [cell("Item", 60, 300, 150, 320), cell("2 pcs", 300, 301, 400, 321), cell("Next", 60, 340, 150, 360)]
    page = run(monkeypatch, FakeRunner(vector=result(lines)), is_pdf=True)
    rows = [l["tokens"][0]["row_idx"] for l in page["lines"]]
    assert rows[0] == rows[1] != rows[2]


def test_bottom_left_boxes_are_flipped_and_degenerate_boxes_dropped(monkeypatch):
    assert _norm_bbox(box(60, 700, 540, 500, Origin.BOTTOMLEFT), W, H) == (0.1, 0.125, 0.9, 0.375)
    assert _norm_bbox(box(10, 10, 10, 20), W, H) is None and _norm_bbox(None, W, H) is None
    page = run(monkeypatch, FakeRunner(vector=result([cell("ok", 0, 0, 10, 10), cell("flat", 5, 5, 5, 9), cell("  ", 0, 0, 5, 5)])), is_pdf=True)
    assert [l["text"] for l in page["lines"]] == ["ok"]


def test_paragraph_texts_are_the_fallback_when_no_line_cells(monkeypatch):
    item = NS(text="Hello   world", prov=[NS(bbox=box(60, 740, 300, 700, Origin.BOTTOMLEFT))], confidence=None)
    page = run(monkeypatch, FakeRunner(vector=result([], texts=[item])), is_pdf=True)
    assert page["lines"][0]["text"] == "Hello world" and page["lines"][0]["confidence"] == 0.95


def test_failed_conversion_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="Docling conversion failed"):
        run(monkeypatch, FakeRunner(), is_pdf=False)


def test_runner_caches_conversions_and_falls_back_to_ocr(monkeypatch):
    runner = runner_module.DoclingRunner()
    calls = []

    def fake_convert(payload, *, is_pdf, ocr):
        calls.append((is_pdf, ocr))
        return result([] if not ocr else [cell("x", 0, 0, 5, 5)], ocr=ocr)

    monkeypatch.setattr(runner, "convert", fake_convert)
    assert runner.convert_page(pdf_bytes=b"pdf", image_loader=lambda: b"png").used_ocr is True
    assert calls == [(True, False), (False, True)]
    assert runner.convert_page(pdf_bytes=None, image_loader=lambda: None) is None
