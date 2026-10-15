from __future__ import annotations

import numpy as np

from app.services.ocr.engine import OCREngine, _paddle_payload


class _PaddleResult:
    def __init__(self, payload: dict):
        self.json = payload


class _PaddleBackend:
    def predict(self, _image):
        yield _PaddleResult(
            {
                "res": {
                    "rec_texts": ["Invoice 42"],
                    "rec_scores": [0.91],
                    "rec_boxes": [[10, 20, 110, 40]],
                }
            }
        )


def test_paddle_payload_accepts_v3_result_json():
    result = _PaddleResult({"res": {"rec_texts": ["Invoice"]}})

    assert _paddle_payload(result) == {"rec_texts": ["Invoice"]}


def test_paddle_adapter_preserves_normalized_evidence_contract(monkeypatch):
    engine = OCREngine()
    monkeypatch.setattr(OCREngine, "paddle_ocr", property(lambda _self: _PaddleBackend()))

    lines = engine._read_paddle_lines(np.zeros((100, 200, 3), dtype=np.uint8), 200, 100)

    assert len(lines) == 1
    assert lines[0].text == "Invoice 42"
    assert lines[0].confidence == 0.91
    assert lines[0].bbox_norm == (0.05, 0.2, 0.55, 0.4)
    assert [token.text for token in lines[0].tokens] == ["Invoice", "42"]
    assert all(token.estimated for token in lines[0].tokens)
    assert all(0 <= value <= 1 for token in lines[0].tokens for value in token.bbox_norm)
