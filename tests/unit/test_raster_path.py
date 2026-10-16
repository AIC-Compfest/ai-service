"""Raster OCR path: configurable Paddle models, thread safety, raster twins. Real Paddle runs only on request."""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.ocr import engine as engine_module
from app.services.ocr.engine import OCREngine
from app.tools.table_eval.runner import raster_twin_png
from app.tools.table_eval.synthetic import build_fixture


@pytest.fixture
def fake_paddle(monkeypatch):
    created = []

    class FakePaddleOCR:
        def __init__(self, **options):
            created.append(options)
            self.active = 0
            self.max_active = 0

        def predict(self, _image):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            self.active -= 1
            return []

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddleOCR))
    monkeypatch.setattr(engine_module, "_GLOBAL_PADDLE_OCR", None)
    return created


def test_default_options_keep_paddle_defaults(fake_paddle):
    OCREngine().paddle_ocr
    (options,) = fake_paddle
    assert "text_detection_model_name" not in options and "text_det_limit_side_len" not in options


def test_models_and_side_limit_are_configurable(fake_paddle, monkeypatch):
    monkeypatch.setattr(settings, "paddle_det_model", "PP-OCRv5_mobile_det")
    monkeypatch.setattr(settings, "paddle_rec_model", "en_PP-OCRv5_mobile_rec")
    monkeypatch.setattr(settings, "paddle_det_limit_side_len", 1280)
    OCREngine().paddle_ocr
    (options,) = fake_paddle
    assert options["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert options["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    assert (options["text_det_limit_type"], options["text_det_limit_side_len"]) == ("max", 1280)


def test_initialisation_and_prediction_are_serialised_across_threads(fake_paddle):
    engine = OCREngine()
    image = np.full((60, 60, 3), 255, np.uint8)
    threads = [threading.Thread(target=lambda: engine._read_paddle_lines(image, 60, 60)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(fake_paddle) == 1  # one predictor despite six concurrent first calls
    assert engine.paddle_ocr.max_active == 1  # never two predictions at once


def test_raster_twin_is_deterministic_and_degradation_changes_pixels(tmp_path):
    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(build_fixture("single_ruled").pdf)
    clean, clean_again = raster_twin_png(pdf, False), raster_twin_png(pdf, False)
    degraded, degraded_again = raster_twin_png(pdf, True), raster_twin_png(pdf, True)
    assert clean == clean_again and degraded == degraded_again and clean != degraded
    decoded = cv2.imdecode(np.frombuffer(clean, np.uint8), 1)
    assert decoded.shape[0] > 1500  # 200 DPI A4


@pytest.mark.skipif(not os.environ.get("RUN_PADDLE_TESTS"), reason="real PaddleOCR is slow (30 to 100 s per page on CPU); set RUN_PADDLE_TESTS=1")
def test_real_paddle_reads_a_rendered_page(monkeypatch):
    import pymupdf

    monkeypatch.setattr(settings, "paddle_det_model", "PP-OCRv5_mobile_det")
    monkeypatch.setattr(settings, "paddle_rec_model", "en_PP-OCRv5_mobile_rec")
    monkeypatch.setattr(settings, "paddle_det_limit_side_len", 1280)
    monkeypatch.setattr(engine_module, "_GLOBAL_PADDLE_OCR", None)
    png = pymupdf.open(stream=build_fixture("single_ruled").pdf, filetype="pdf")[0].get_pixmap(dpi=150).tobytes("png")
    page = OCREngine().process_image(png, 1, {})
    assert page.engine_used == "paddleocr" and any("Deskripsi" in line.text for line in page.lines)
