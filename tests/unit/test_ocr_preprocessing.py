from __future__ import annotations

import cv2
import numpy as np

from app.core.config import settings
from app.services.ocr.engine import OCREngine, OCRLine, OCRToken
from app.services.preprocessing.ocr_prep import OCRPreprocessor, PreparedImage


def _line(confidence: float, box: tuple[float, float, float, float]) -> OCRLine:
    token = OCRToken("INVOICE", confidence, box, (0, 0, 1, 1))
    return OCRLine("INVOICE", confidence, box, (0, 0, 1, 1), [token])


def test_preprocessor_upscale_tracks_inverse_geometry():
    preprocessor = OCRPreprocessor(min_side_for_upscale=50, max_upscale=2.0)
    prepared = preprocessor.prepare(np.full((40, 80, 3), 255, dtype=np.uint8))

    assert "adaptive_upscale" in prepared.operations
    assert prepared.image.shape[:2] == (50, 100)
    assert np.allclose(prepared.inverse_transform, np.diag([0.8, 0.8, 1.0]))


def test_preprocessing_ab_selects_better_ocr_and_restores_source_coordinates(monkeypatch):
    raw_image = np.full((100, 200, 3), 255, dtype=np.uint8)
    prepared_image = np.full((200, 400, 3), 255, dtype=np.uint8)
    prepared = PreparedImage(
        prepared_image,
        ("adaptive_upscale",),
        np.diag([0.5, 0.5, 1.0]).astype(np.float32),
    )
    encoded_ok, encoded = cv2.imencode(".png", raw_image)
    assert encoded_ok

    engine = OCREngine()
    monkeypatch.setattr(settings, "enable_ocr_preprocess", True)
    monkeypatch.setattr(OCRPreprocessor, "prepare", lambda _self, _image: prepared)

    def fake_read(image, _width, _height):
        return [_line(0.9, (20, 40, 120, 80))] if image.shape[1] == 400 else [_line(0.6, (10, 20, 60, 40))]

    monkeypatch.setattr(engine, "_read_paddle_lines", fake_read)
    evidence = engine.process_image(encoded.tobytes(), page_number=1, quality={})

    assert evidence.quality["ocr_preprocessing"]["selected"] == "prepared"
    assert evidence.quality["ocr_preprocessing"]["operations"] == ["adaptive_upscale"]
    assert evidence.lines[0].bbox_pixel == (10.0, 20.0, 60.0, 40.0)
    assert evidence.lines[0].bbox_norm == (0.05, 0.2, 0.3, 0.4)


def test_preprocessing_disabled_does_not_add_evidence_metadata(monkeypatch):
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    engine = OCREngine()
    monkeypatch.setattr(settings, "enable_ocr_preprocess", False)
    monkeypatch.setattr(engine, "_read_paddle_lines", lambda *_args: [_line(0.6, (10, 20, 60, 40))])

    evidence = engine.process_image(encoded.tobytes(), page_number=1, quality={})

    assert "ocr_preprocessing" not in evidence.quality
