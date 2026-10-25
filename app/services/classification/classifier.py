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
- Supervised ML model only (TF-IDF 1-2 N-grams + Logistic Regression), trained by
  scripts/train_classifier.py on the labelled PDFs in DATASET/Dataset.
  No hardcoded keyword rules.
- Low-confidence predictions are flagged for human review.
- Outputs full class probability distribution + model version metadata.
"""

from __future__ import annotations
from pathlib import Path
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
    # Training data has no negative class, so a document that no known class
    # claims with at least this probability is treated as UNKNOWN.
    UNKNOWN_THRESHOLD = 0.40
    MODEL_VERSION = "doc-classifier-v1"

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

        classification_method = "model_unavailable"
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
                if ml_confidence < self.UNKNOWN_THRESHOLD:
                    best_type = DocumentType.UNKNOWN
                classification_method = "ml_tfidf_logistic_regression"
            except Exception as e:
                logger.warning("ml_inference_failed", error=str(e))

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

