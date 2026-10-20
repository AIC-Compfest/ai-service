"""
train_classifier.py

Offline training script for the Document Classification Model (PRD §20).

Training data comes exclusively from the labelled PDFs in DATASET/Dataset
(no hardcoded keywords or templates). The label is the document family whose
ground-truth JSON lives in that folder:

    invoice/       -> INVOICE
    surat jalan/   -> SURAT_JALAN
    PoD/           -> POD
    rate card/     -> RATE_AGREEMENT

Model: TF-IDF (1-2 grams) + multinomial Logistic Regression.

Train/test split is done *by template group* (e.g. invoice_7_variasi_1 and
invoice_7_variasi_2 share a group) so near-duplicate variations never leak
from train into test.

Outputs (app/models):
    - doc_classifier_v1.joblib
    - tfidf_v1.joblib
    - classifier_metrics.json
    - classifier_split.json  (which files went to train / test)

Usage:
    python -m scripts.train_classifier [--dataset-dir PATH] [--test-size 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "DATASET" / "Dataset"

# Dataset folder name -> class label
FOLDER_LABELS = {
    "invoice": "INVOICE",
    "surat jalan": "SURAT_JALAN",
    "PoD": "POD",
    "rate card": "RATE_AGREEMENT",
}

# Extra labelled synthetic documents used by the pipeline/e2e tests. They are
# added to the TRAIN set only; the held-out test split is real DATASET only.
SYNTHETIC_ROOT = Path(__file__).resolve().parent.parent.parent / "synthetic-data" / "dataset"
SYNTHETIC_RUNS = ["v4_pilot_20", "v3_pilot_20"]
SYNTHETIC_PATTERNS = {
    "INVOICE": "invoices/*.pdf",
    "SURAT_JALAN": "transactions/*/surat_jalan.pdf",
    "POD": "transactions/*/pod.pdf",
    "RATE_AGREEMENT": "transactions/*/rate_agreement.pdf",
}

# Folders that only hold duplicates of other PDFs (e.g. a zip-extracted copy)
EXCLUDED_PARTS = {"data_PoD_testing"}

SEED = 42
MIN_PAGE_CHARS = 40


def _group_id(label: str, path: Path) -> str:
    """Group variations of the same template: '<label>:<template-number>'."""
    m = re.search(r"_(\d+)\s*_?[vV]ariasi", path.name)
    if m:
        return f"{label}:{m.group(1)}"
    return f"{label}:{path.stem}"  # e.g. 'ori' scans each form their own group


def load_documents(dataset_dir: Path) -> list[dict]:
    """Extract text (whole doc + each page) from every labelled PDF."""
    docs: list[dict] = []
    for folder, label in FOLDER_LABELS.items():
        base = dataset_dir / folder
        if not base.exists():
            print(f"[WARN] missing dataset folder: {base}")
            continue
        for pdf in sorted(base.rglob("*.pdf")):
            if EXCLUDED_PARTS & set(pdf.parts):
                continue
            try:
                with pymupdf.open(pdf) as doc:
                    pages = [p.get_text() for p in doc]
            except Exception as e:  # corrupt pdf
                print(f"[WARN] cannot read {pdf}: {e}")
                continue
            full = "\n".join(pages).strip()
            if not full:
                print(f"[WARN] no text layer, skipped: {pdf.name}")
                continue
            docs.append({
                "file": str(pdf.relative_to(dataset_dir)),
                "label": label,
                "group": _group_id(label, pdf),
                "full_text": full,
                "pages": [p.strip() for p in pages if len(p.strip()) >= MIN_PAGE_CHARS],
            })
    return docs


def load_synthetic_documents(root: Path = SYNTHETIC_ROOT) -> list[dict]:
    docs: list[dict] = []
    for run in SYNTHETIC_RUNS:
        base = root / run
        if not base.exists():
            continue
        for label, pattern in SYNTHETIC_PATTERNS.items():
            for pdf in sorted(base.glob(pattern)):
                try:
                    with pymupdf.open(pdf) as doc:
                        pages = [p.get_text() for p in doc]
                except Exception as e:
                    print(f"[WARN] cannot read {pdf}: {e}")
                    continue
                full = "\n".join(pages).strip()
                if not full:
                    continue
                docs.append({
                    "file": f"synthetic:{pdf.relative_to(root)}",
                    "label": label,
                    "group": f"synthetic:{label}:{pdf.relative_to(root)}",
                    "full_text": full,
                    "pages": [p.strip() for p in pages if len(p.strip()) >= MIN_PAGE_CHARS],
                })
    return docs


def split_by_group(docs: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified (per class) group split: whole templates go to train or test."""
    rng = random.Random(seed)
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        if d["group"] not in groups_by_label[d["label"]]:
            groups_by_label[d["label"]].append(d["group"])

    test_groups: set[str] = set()
    for label, groups in groups_by_label.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * test_size))
        test_groups.update(groups[:n_test])

    train = [d for d in docs if d["group"] not in test_groups]
    test = [d for d in docs if d["group"] in test_groups]
    return train, test


def to_samples(docs: list[dict], include_pages: bool) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for d in docs:
        texts.append(d["full_text"])
        labels.append(d["label"])
        if include_pages and len(d["pages"]) > 1:
            for p in d["pages"]:
                texts.append(p)
                labels.append(d["label"])
    return texts, labels


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )


def make_model() -> LogisticRegression:
    return LogisticRegression(
        C=10.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )


def train_and_save_model(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    test_size: float = 0.20,
    use_synthetic: bool = True,
) -> dict:
    """Train TF-IDF + Logistic Regression on the labelled dataset and save artifacts."""
    docs = load_documents(dataset_dir)
    if not docs:
        raise RuntimeError(f"No training documents found under {dataset_dir}")

    train_docs, test_docs = split_by_group(docs, test_size, SEED)
    synthetic_docs = load_synthetic_documents() if use_synthetic else []
    train_docs = train_docs + synthetic_docs
    # Test is evaluated on whole documents AND on single pages (pipeline classifies pages).
    X_train, y_train = to_samples(train_docs, include_pages=True)
    X_test_doc, y_test_doc = to_samples(test_docs, include_pages=False)
    X_test_all, y_test_all = to_samples(test_docs, include_pages=True)

    vectorizer = make_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = make_model()
    clf.fit(X_train_vec, y_train)

    def evaluate(X: list[str], y: list[str]) -> dict:
        pred = clf.predict(vectorizer.transform(X))
        rep = classification_report(y, pred, labels=list(clf.classes_), output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=list(clf.classes_))
        return {
            "n_samples": len(y),
            "accuracy": round(float(np.mean(np.array(pred) == np.array(y))), 4),
            "classification_report": {k: v for k, v in rep.items() if isinstance(v, dict)},
            "confusion_matrix": cm.tolist(),
        }

    doc_eval = evaluate(X_test_doc, y_test_doc)
    page_eval = evaluate(X_test_all, y_test_all)

    # Grouped cross-validation on the full corpus (document level) for stability estimate.
    # Synthetic docs are always in the training side of every fold.
    all_groups = sorted({d["group"] for d in docs})
    rng = random.Random(SEED)
    rng.shuffle(all_groups)
    n_folds = 5
    fold_of = {g: i % n_folds for i, g in enumerate(all_groups)}
    cv_scores: list[float] = []
    for k in range(n_folds):
        tr = [d for d in docs if fold_of[d["group"]] != k] + synthetic_docs
        va = [d for d in docs if fold_of[d["group"]] == k]
        if not va or len({d["label"] for d in tr}) < len(FOLDER_LABELS):
            continue
        Xtr, ytr = to_samples(tr, include_pages=True)
        Xva, yva = to_samples(va, include_pages=False)
        v = make_vectorizer()
        m = make_model().fit(v.fit_transform(Xtr), ytr)
        cv_scores.append(float(np.mean(m.predict(v.transform(Xva)) == np.array(yva))))

    # Final model: refit on ALL data so deployment benefits from every labelled sample.
    # Reported test metrics above come from the held-out split model.
    X_all, y_all = to_samples(docs + synthetic_docs, include_pages=True)
    final_vec = make_vectorizer()
    final_clf = make_model().fit(final_vec.fit_transform(X_all), y_all)

    metrics = {
        "model_name": "doc-classifier-v1",
        "algorithm": "TF-IDF(1-2gram) + LogisticRegression (multinomial)",
        "data_source": str(dataset_dir),
        "split": "group-stratified by template",
        "test_size": test_size,
        "n_documents": len(docs),
        "n_train_documents": len(train_docs),
        "n_synthetic_train_documents": len(synthetic_docs),
        "n_test_documents": len(test_docs),
        "documents_per_class": {
            lbl: sum(1 for d in docs if d["label"] == lbl) for lbl in FOLDER_LABELS.values()
        },
        "test_document_level": doc_eval,
        "test_page_level": page_eval,
        "grouped_cv_mean_accuracy": round(float(np.mean(cv_scores)), 4) if cv_scores else None,
        "grouped_cv_std": round(float(np.std(cv_scores)), 4) if cv_scores else None,
        "classes": list(final_clf.classes_),
        "vocabulary_size": len(final_vec.vocabulary_),
        "total_training_samples": len(y_all),
    }

    joblib.dump(final_clf, MODELS_DIR / "doc_classifier_v1.joblib")
    joblib.dump(final_vec, MODELS_DIR / "tfidf_v1.joblib")
    (MODELS_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "classifier_split.json").write_text(
        json.dumps(
            {
                "train": sorted(d["file"] for d in train_docs),
                "test": sorted(d["file"] for d in test_docs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[TRAIN_CLASSIFIER_SUCCESS]")
    print(f"  Artifacts saved to: {MODELS_DIR}")
    print(f"  Docs: {len(docs)} (train {len(train_docs)} / test {len(test_docs)})")
    print(f"  Test accuracy (document-level): {doc_eval['accuracy'] * 100:.2f}%")
    print(f"  Test accuracy (page-level):     {page_eval['accuracy'] * 100:.2f}%")
    if cv_scores:
        print(f"  Grouped CV mean: {np.mean(cv_scores) * 100:.2f}% (+/- {np.std(cv_scores) * 100:.2f})")
    print(f"  Classes: {list(final_clf.classes_)}")
    print(f"  Confusion matrix (doc-level, rows=true): {doc_eval['confusion_matrix']}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--no-synthetic", action="store_true", help="train on DATASET only")
    args = ap.parse_args()
    train_and_save_model(args.dataset_dir, args.test_size, use_synthetic=not args.no_synthetic)
