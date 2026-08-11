"""
Image preprocessing — PRD §21. Menggunakan OpenCV.
Potential operations: orientation detection, deskew, perspective correction,
illumination normalization, noise reduction, resolution enhancement, crop.

PENTING: jangan selalu menerapkan preprocessing agresif. Hasilkan beberapa
variant (original, variant_A, variant_B, ...) lalu pilih berdasarkan
OCR/document confidence (PRD §21).

STUB: pipeline nyata (deskew dst) belum diimplementasikan — saat ini hanya
mengembalikan bytes asli sebagai "original" variant.
"""

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def generate_variants(self, image_bytes: bytes) -> dict[str, bytes]:
        """
        Return dict of variant_name -> image_bytes.
        Saat ini hanya "original". Variant_A (deskew), variant_B (illumination
        normalization), variant_C (perspective correction) akan ditambahkan
        di Fase 3 (lihat ROADMAP.md).
        """
        logger.info("preprocessor.stub_called", size_bytes=len(image_bytes))
        _ = np.frombuffer  # placeholder to keep numpy/opencv imports live
        return {"original": image_bytes}
