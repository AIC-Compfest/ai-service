from app.services.ocr.engine import OCREngine, OCRToken, OCRLine, PageOCREvidence, to_evidence_json


def test_token_to_dict_has_required_candidate_fields():
    token = OCRToken("PT", 0.97, (10, 20, 50, 40), (0.01, 0.02, 0.05, 0.04))

    assert token.to_dict() == {
        "text": "PT",
        "confidence": 0.97,
        "bbox_norm": [0.01, 0.02, 0.05, 0.04],
        "_estimated": False,
        "row_idx": 0,
    }
    assert token.to_full_dict()["bbox_pixel"] == [10, 20, 50, 40]


def test_estimated_token_is_flagged_and_evidence_is_candidate_payload():
    token = OCRToken(
        "Maju", 0.96, (70, 36, 140, 55), (0.07, 0.0365, 0.14, 0.055), estimated=True
    )
    line = OCRLine("Maju", 0.96, (70, 36, 140, 55), (0.07, 0.0365, 0.14, 0.055), [token])
    page = PageOCREvidence(1, 1240, 1754, "paddleocr", False, [line], [], [], {})

    evidence = to_evidence_json([page])

    assert evidence["pages"][0]["lines"][0]["tokens"][0]["_estimated"] is True
    assert "bbox_pixel" not in evidence["pages"][0]["lines"][0]["tokens"][0]
    assert all(0.0 <= value <= 1.0 for value in evidence["pages"][0]["lines"][0]["tokens"][0]["bbox_norm"])


def test_phrase_token_estimation_uses_relative_glyph_widths():
    engine = OCREngine()

    boxes = engine._estimate_token_boxes("WWW i", (0.0, 10.0, 100.0, 20.0))

    assert [word for word, _ in boxes] == ["WWW", "i"]
    assert boxes[0][1][2] - boxes[0][1][0] > boxes[1][1][2] - boxes[1][1][0]
    assert boxes[0][1][0] >= 0.0
    assert boxes[-1][1][2] == 100.0


def test_layout_regions_use_normalized_relative_thresholds():
    engine = OCREngine()
    lines = [
        OCRLine("INVOICE", 0.9, (0, 0, 1, 1), (0.1, 0.10, 0.5, 0.12)),
        OCRLine("Description Quantity", 0.9, (0, 0, 1, 1), (0.1, 0.45, 0.5, 0.47)),
        OCRLine("Grand Total", 0.9, (0, 0, 1, 1), (0.5, 0.82, 0.9, 0.84)),
    ]

    regions = {region.region_type for region in engine._detect_layout_regions(lines, 2000, 400)}

    assert {"header", "table", "totals", "signature_block"} <= regions
