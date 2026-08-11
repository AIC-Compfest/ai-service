"""
OCR / Document Understanding — PRD §22.

Recommended: PP-OCRv6 + PP-StructureV3 sebagai baseline, fallback ke
PaddleOCR-VL-1.6 jika confidence rendah. Jangan panggil VLM berat untuk
semua dokumen jika jalur OCR/layout biasa sudah cukup.

STUB: model belum diintegrasikan (lihat requirements.txt, paddlepaddle
masih di-comment). Endpoint akan mengembalikan low-confidence result yang
otomatis memicu human review di backend-api, bukan mengarang teks.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)


class OCREngine:
    engine_name = "pp-ocr-v6"
    fallback_engine_name = "paddleocr-vl-1.6"

    def run(self, image_bytes: bytes) -> dict:
        logger.info("ocr_engine.stub_called", size_bytes=len(image_bytes))
        return {
            "text_blocks": [],
            "tables": [],
            "confidence": 0.0,
            "engine_used": self.engine_name,
            "fallback_triggered": False,
        }
