"""
Base contract untuk semua extractor. PRD §23 secara eksplisit melarang satu
fungsi generik extract_document() — setiap document type punya extractor
sendiri yang tahu field spesifiknya (lihat invoice.py, pod.py, dst).
"""

from abc import ABC, abstractmethod

from app.schemas.evidence import ExtractedField


class BaseExtractor(ABC):
    document_type: str

    @abstractmethod
    def extract(self, document_id: str, ocr_result: dict) -> list[ExtractedField]:
        """
        Ambil field spesifik document type dari hasil OCR/layout.
        Field yang tidak ditemukan HARUS tetap muncul di output dengan
        is_unknown=True, bukan dihilangkan begitu saja (evidence traceability
        harus lengkap — PRD §18).
        """
        raise NotImplementedError
