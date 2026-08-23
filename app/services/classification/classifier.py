"""
classifier.py

Document Classifier for 3PL Invoice Reconciliation Engine (PRD §20).
Distinguishes between:
- INVOICE
- SURAT_JALAN
- POD
- RATE_AGREEMENT
- UNKNOWN

Architecture:
- Primary: Supervised ML Model (TF-IDF 1-2 N-grams + Calibrated Logistic Regression)
- Secondary: Structural & Keyword Signal Evidence Verification
- Outputs full class probability distribution + model version metadata.
"""

from __future__ import annotations
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np

from app.core.logging import get_logger
from app.schemas.classify import ClassifyResponse
from app.schemas.document_type import DocumentType

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class DocumentClassifier:
    """Production ML-powered document classifier with structural validation."""

    CONFIDENCE_THRESHOLD = 0.70
    MODEL_VERSION = "doc-classifier-v1"

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

    def __init__(self):
        self._clf = None
        self._vectorizer = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        clf_file = MODELS_DIR / "doc_classifier_v1.joblib"
        vec_file = MODELS_DIR / "tfidf_v1.joblib"

        if clf_file.exists() and vec_file.exists():
            try:
                self._clf = joblib.load(clf_file)
                self._vectorizer = joblib.load(vec_file)
                logger.info("ml_classifier_loaded", model_version=self.MODEL_VERSION)
                return
            except Exception as e:
                logger.warning("failed_to_load_saved_model", error=str(e))

        # Train on-demand if artifact missing
        try:
            from scripts.train_classifier import train_and_save_model
            train_and_save_model()
            self._clf = joblib.load(clf_file)
            self._vectorizer = joblib.load(vec_file)
            logger.info("ml_classifier_trained_and_loaded", model_version=self.MODEL_VERSION)
        except Exception as e:
            logger.warning("ml_classifier_bootstrap_failed", error=str(e))
            self._clf = None
            self._vectorizer = None

    def classify_text(self, text: str, document_id: str = "DOC-001") -> ClassifyResponse:
        """Classify document using ML classifier with calibrated probabilities."""
        text_clean = text.strip()
        if not text_clean:
            return ClassifyResponse(
                document_id=document_id,
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                requires_human_review=True,
                probabilities={d.value: 0.0 for d in DocumentType},
                model_version=self.MODEL_VERSION,
                classification_method="empty_input_fallback",
            )

        probabilities: dict[str, float] = {
            DocumentType.INVOICE.value: 0.0,
            DocumentType.SURAT_JALAN.value: 0.0,
            DocumentType.POD.value: 0.0,
            DocumentType.RATE_AGREEMENT.value: 0.0,
            DocumentType.UNKNOWN.value: 0.0,
        }

        classification_method = "keyword_heuristic_fallback"
        best_type = DocumentType.UNKNOWN
        confidence = 0.0

        # Step 1: ML Model Inference if available
        if self._clf is not None and self._vectorizer is not None:
            try:
                X_vec = self._vectorizer.transform([text_clean])
                probs = self._clf.predict_proba(X_vec)[0]
                classes = self._clf.classes_

                for cls_name, p in zip(classes, probs):
                    if cls_name in probabilities:
                        probabilities[cls_name] = round(float(p), 4)

                ml_best_idx = int(np.argmax(probs))
                ml_best_class = str(classes[ml_best_idx])
                ml_confidence = float(probs[ml_best_idx])

                try:
                    best_type = DocumentType(ml_best_class)
                except ValueError:
                    best_type = DocumentType.UNKNOWN

                confidence = round(ml_confidence, 4)
                classification_method = "ml_tfidf_logistic_regression"
            except Exception as e:
                logger.warning("ml_inference_failed_falling_back_to_rules", error=str(e))

        # Step 2: Structural / Keyword Signal Verification (Ensemble Verification)
        text_lower = text_clean.lower()
        cleaned_text = re.sub(r'[^a-z0-9\s/_-]', ' ', text_lower)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        rule_scores = {
            DocumentType.INVOICE: 0.0,
            DocumentType.SURAT_JALAN: 0.0,
            DocumentType.POD: 0.0,
            DocumentType.RATE_AGREEMENT: 0.0,
        }

        def match_signal(keyword: str, target_text: str) -> bool:
            if " " in keyword:
                return keyword in target_text
            return bool(re.search(r'(?:\b|[^a-z0-9])' + re.escape(keyword) + r'(?:\b|[^a-z0-9])', target_text))

        for kw, wt in self.SIGNALS_INVOICE.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                rule_scores[DocumentType.INVOICE] += wt

        for kw, wt in self.SIGNALS_SURAT_JALAN.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                rule_scores[DocumentType.SURAT_JALAN] += wt

        for kw, wt in self.SIGNALS_POD.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                rule_scores[DocumentType.POD] += wt

        for kw, wt in self.SIGNALS_RATE_AGREEMENT.items():
            if match_signal(kw, text_lower) or match_signal(kw, cleaned_text):
                rule_scores[DocumentType.RATE_AGREEMENT] += wt

        # Disambiguation boost for POD
        if rule_scores[DocumentType.POD] >= 5.0 and any(k in text_lower for k in ["tanda tangan", "penerima", "received", "pod", "bukti"]):
            rule_scores[DocumentType.POD] += 4.0

        rule_best_type = DocumentType.UNKNOWN
        rule_best_score = 0.0
        for dtype, score in rule_scores.items():
            if score > rule_best_score:
                rule_best_score = score
                rule_best_type = dtype

        # Step 3: Ensemble ML prediction with structural signal confirmation
        if best_type != DocumentType.UNKNOWN and rule_best_type == best_type and rule_best_score >= 3.0:
            confidence = round(min(0.99, max(confidence, 0.72 + (rule_best_score / (rule_best_score + 5.0)) * 0.26)), 4)
            classification_method = "ml_structural_ensemble"

        # If ML predicted UNKNOWN but rules strongly suggest known type, or vice-versa
        if best_type == DocumentType.UNKNOWN and rule_best_score >= 8.0:
            best_type = rule_best_type
            confidence = min(0.95, round(rule_best_score / (rule_best_score + 3.0), 3))
            classification_method = "ml_rule_ensemble"

        requires_review = confidence < self.CONFIDENCE_THRESHOLD or best_type == DocumentType.UNKNOWN

        logger.info(
            "document_classified",
            document_id=document_id,
            best_type=best_type.value,
            confidence=confidence,
            method=classification_method,
            probabilities=probabilities,
        )

        return ClassifyResponse(
            document_id=document_id,
            document_type=best_type,
            confidence=confidence,
            requires_human_review=requires_review,
            probabilities=probabilities,
            model_version=self.MODEL_VERSION,
            classification_method=classification_method,
        )

    def classify(self, document_id: str, file_bytes: bytes, text_hint: str | None = None) -> ClassifyResponse:
        """Entrypoint for classifying document bytes."""
        if text_hint:
            return self.classify_text(text_hint, document_id=document_id)

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

