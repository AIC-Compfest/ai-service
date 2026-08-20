"""
classifier.py

Document Classifier for 3PL Invoice Reconciliation Engine (PRD §20).
Distinguishes between:
- INVOICE
- SURAT_JALAN
- POD
- RATE_AGREEMENT
- UNKNOWN

Multi-signal evidence-based classifier using layout cues, OCR keywords,
and structural markers without hardcoding specific vendor identities.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple
from app.core.logging import get_logger
from app.schemas.classify import ClassifyResponse
from app.schemas.document_type import DocumentType

logger = get_logger(__name__)


class DocumentClassifier:
    """Evidence-based document classifier."""

    CONFIDENCE_THRESHOLD = 0.70

    # Keyword signal weights for each document type with OCR noise resilience
    SIGNALS_INVOICE = {
        "invoice": 4.0, "faktur": 4.0, "tax invoice": 5.0, "faktur pajak": 5.0,
        "tagihan": 3.0, "due date": 3.0, "jatuh tempo": 3.0, "subtotal": 3.5,
        "ppn": 3.0, "tax": 2.0, "grand total": 4.0, "total tagihan": 4.0,
        "bill to": 3.0, "remit to": 3.0, "billed": 2.5, "amount due": 3.5,
        "e-way bill": 3.5, "sac code": 3.0, "bank account": 2.5, "rekening": 2.0,
        "no. invoice": 4.5, "nomor invoice": 4.5, "no invoice": 4.5,
    }

    SIGNALS_SURAT_JALAN = {
        "surat jalan": 5.0, "surat tanda pengiriman": 5.0, "shipment airwaybill": 4.5,
        "airwaybill": 4.0, "delivery note": 4.0, "consignment note": 4.0,
        "pengirim": 2.5, "shipper": 2.5, "penerima": 1.5, "consignee": 2.0,
        "asal": 1.5, "orgn": 2.5, "tujuan": 1.5, "dest": 2.0,
        "koli": 3.0, "dus": 2.0, "berat aktual": 3.0, "scale weight": 3.0,
        "supir": 2.5, "driver": 2.0, "cetak resi": 3.0, "pickup by": 3.0,
        "operasional": 2.0, "pengantar barang": 4.5, "surat pengantar": 4.5,
    }

    SIGNALS_POD = {
        "proof of delivery": 6.0, "pod": 5.0, "delivery receipt": 5.0,
        "bukti pengiriman": 5.0, "bukti serah terima": 6.0, "tanda terima": 4.5,
        "bukti terima": 5.0, "serah terima": 5.0, "sepah terima": 5.0,
        "penyerahan barang": 5.5, "bukti penyerahan": 5.5,
        "received by": 4.0, "diterima oleh": 4.5, "delivered": 3.0,
        "delivered quantity": 4.0, "delivered date": 4.0, "arrival time": 3.5,
        "good condition": 3.5, "kondisi barang": 3.5, "terkirim lengkap": 4.5,
        "consignee copy": 4.0, "transporter copy": 4.0, "signature of delivering": 4.5,
        "tanda tangan & cap": 5.5, "cap penerima": 5.0, "tanda tangan": 3.5,
        "nama penerima": 4.0, "recipient": 4.0, "tanggal terima": 4.5, "waktu tiba": 3.5,
        "received": 3.0,
    }

    SIGNALS_RATE_AGREEMENT = {
        "rate agreement": 6.0, "rate card": 6.0, "tariff sheet": 6.0,
        "perjanjian kerjasama": 6.0, "kontrak pengangkutan": 6.0,
        "perjanjian": 4.5, "kontrak": 4.5, "tariff": 4.0, "tarif": 3.5,
        "master services agreement": 6.0, "effective from": 4.5, "masa berlaku": 4.5,
        "weight tier": 5.0, "tier rate": 5.0, "minimum charge": 5.0,
        "fuel surcharge index": 5.0, "dispute window": 5.0, "pasal": 5.0,
        "base rate matrix": 5.0, "terms of agreement": 4.5, "tabel tarif": 5.0,
        "syarat dan ketentuan tarif": 5.0, "biaya minimum": 4.5,
    }

    def classify_text(self, text: str, document_id: str = "DOC-001") -> ClassifyResponse:
        """Classify document based on extracted text content and structure."""
        # Clean text for robust matching (normalize symbols and multiple spaces)
        text_lower = text.lower()
        cleaned_text = re.sub(r'[^a-z0-9\s/_-]', ' ', text_lower)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        scores = {
            DocumentType.INVOICE: 0.0,
            DocumentType.SURAT_JALAN: 0.0,
            DocumentType.POD: 0.0,
            DocumentType.RATE_AGREEMENT: 0.0,
        }
        detected_signals: list[str] = []

        def match_signal(keyword: str, target_text: str) -> bool:
            if " " in keyword:
                return keyword in target_text
            return bool(re.search(r'(?:\b|[^a-z0-9])' + re.escape(keyword) + r'(?:\b|[^a-z0-9])', target_text))

        # Evaluate Invoice signals
        for kw, wt in self.SIGNALS_INVOICE.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                scores[DocumentType.INVOICE] += wt
                detected_signals.append(f"INVOICE:{kw}")

        # Evaluate Surat Jalan signals
        for kw, wt in self.SIGNALS_SURAT_JALAN.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                scores[DocumentType.SURAT_JALAN] += wt
                detected_signals.append(f"SJ:{kw}")

        # Evaluate POD signals
        for kw, wt in self.SIGNALS_POD.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                scores[DocumentType.POD] += wt
                detected_signals.append(f"POD:{kw}")

        # Evaluate Rate Agreement signals
        for kw, wt in self.SIGNALS_RATE_AGREEMENT.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                scores[DocumentType.RATE_AGREEMENT] += wt
                detected_signals.append(f"RA:{kw}")

        # Disambiguation heuristics
        if scores[DocumentType.POD] >= 5.0 and ("tanda tangan" in text_lower or "penerima" in text_lower or "received" in text_lower or "pod" in text_lower or "bukti" in text_lower):
            # Boost POD when delivery/receiver/signature markers are present
            scores[DocumentType.POD] += 4.0

        best_type = DocumentType.UNKNOWN
        best_score = 0.0
        for dtype, score in scores.items():
            if score > best_score:
                best_score = score
                best_type = dtype

        total_score = sum(scores.values())
        if total_score > 0 and best_score >= 3.5:
            confidence = min(0.99, max(0.50, round(best_score / (best_score + 2.5), 3)))
        else:
            confidence = 0.0
            best_type = DocumentType.UNKNOWN

        requires_review = confidence < self.CONFIDENCE_THRESHOLD or best_type == DocumentType.UNKNOWN

        logger.info(
            "document_classified",
            document_id=document_id,
            best_type=best_type.value,
            confidence=confidence,
            scores={k.value: round(v, 1) for k, v in scores.items()},
        )

        return ClassifyResponse(
            document_id=document_id,
            document_type=best_type,
            confidence=confidence,
            requires_human_review=requires_review,
        )

    def classify(self, document_id: str, file_bytes: bytes, text_hint: str | None = None) -> ClassifyResponse:
        """Entrypoint for classifying document bytes."""
        if text_hint:
            return self.classify_text(text_hint, document_id=document_id)

        # Fallback quick text extraction via PyMuPDF if PDF
        text = ""
        if file_bytes.startswith(b"%PDF"):
            import fitz
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    text += page.get_text() + "\n"
            except Exception as e:
                logger.warning("pdf_text_extraction_failed", error=str(e))

        return self.classify_text(text, document_id=document_id)


def get_classifier() -> DocumentClassifier:
    return DocumentClassifier()
