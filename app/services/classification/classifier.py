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

    # Keyword signal weights for each document type
    SIGNALS_INVOICE = {
        "invoice": 3.0, "faktur": 3.0, "tax invoice": 4.0, "faktur pajak": 4.0,
        "tagihan": 2.5, "due date": 2.0, "jatuh tempo": 2.0, "subtotal": 2.5,
        "ppn": 2.5, "tax": 1.5, "grand total": 3.0, "total tagihan": 3.0,
        "bill to": 2.0, "remit to": 2.5, "billed": 2.0, "amount due": 3.0,
        "e-way bill": 3.5, "sac code": 3.0, "bank account": 2.0, "rekening": 1.5,
    }

    SIGNALS_SURAT_JALAN = {
        "surat jalan": 4.0, "surat tanda pengiriman": 4.5, "shipment airwaybill": 4.5,
        "airwaybill": 3.5, "delivery note": 3.5, "consignment note": 3.5,
        "pengirim": 2.0, "shipper": 2.0, "penerima": 1.5, "consignee": 2.0,
        "asal": 1.5, "orgn": 2.5, "tujuan": 1.5, "dest": 2.0,
        "koli": 2.5, "dus": 2.0, "berat aktual": 2.5, "scale weight": 2.5,
        "supir": 2.0, "driver": 1.5, "cetak resi": 3.0, "pickup by": 3.0,
        "operasional": 2.0, "white : shipper": 4.0,
    }

    SIGNALS_POD = {
        "proof of delivery": 4.5, "pod": 3.5, "delivery receipt": 4.0,
        "bukti pengiriman": 4.0, "bukti serah terima": 4.5, "tanda terima": 3.0,
        "received by": 3.5, "diterima oleh": 3.5, "delivered": 2.5,
        "delivered quantity": 3.5, "delivered date": 3.0, "arrival time": 3.0,
        "good condition": 2.5, "kondisi barang": 2.5, "terkirim lengkap": 4.0,
        "consignee copy": 4.0, "transporter copy": 4.0, "signature of delivering": 4.0,
    }

    SIGNALS_RATE_AGREEMENT = {
        "rate agreement": 6.0, "rate card": 6.0, "tariff sheet": 6.0,
        "perjanjian kerjasama": 6.0, "kontrak pengangkutan": 6.0,
        "perjanjian": 4.0, "kontrak": 4.0, "tariff": 4.0, "tarif": 3.0,
        "master services agreement": 6.0, "effective from": 4.0, "masa berlaku": 4.0,
        "weight tier": 5.0, "tier rate": 5.0, "minimum charge": 5.0,
        "fuel surcharge index": 5.0, "dispute window": 5.0, "pasal": 5.0,
        "base rate matrix": 5.0, "terms of agreement": 4.0, "tabel tarif": 5.0,
        "syarat dan ketentuan tarif": 5.0, "biaya minimum": 4.0,
    }

    def classify_text(self, text: str, document_id: str = "DOC-001") -> ClassifyResponse:
        """Classify document based on extracted text content and structure."""
        text_lower = text.lower()

        scores = {
            DocumentType.INVOICE: 0.0,
            DocumentType.SURAT_JALAN: 0.0,
            DocumentType.POD: 0.0,
            DocumentType.RATE_AGREEMENT: 0.0,
        }
        detected_signals: list[str] = []

        # Evaluate Invoice signals
        for kw, wt in self.SIGNALS_INVOICE.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                scores[DocumentType.INVOICE] += wt
                detected_signals.append(f"INVOICE:{kw}")

        # Evaluate Surat Jalan signals
        for kw, wt in self.SIGNALS_SURAT_JALAN.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                scores[DocumentType.SURAT_JALAN] += wt
                detected_signals.append(f"SJ:{kw}")

        # Evaluate POD signals
        for kw, wt in self.SIGNALS_POD.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                scores[DocumentType.POD] += wt
                detected_signals.append(f"POD:{kw}")

        # Evaluate Rate Agreement signals
        for kw, wt in self.SIGNALS_RATE_AGREEMENT.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                scores[DocumentType.RATE_AGREEMENT] += wt
                detected_signals.append(f"RA:{kw}")

        best_type = DocumentType.UNKNOWN
        best_score = 0.0
        for dtype, score in scores.items():
            if score > best_score:
                best_score = score
                best_type = dtype

        total_score = sum(scores.values())
        if total_score > 0 and best_score >= 4.0:
            confidence = min(0.99, max(0.50, round(best_score / (best_score + 4.0), 3)))
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
