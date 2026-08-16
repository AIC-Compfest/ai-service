#!/usr/bin/env python3
"""
benchmark_phase5.py

End-to-End Benchmark & Evaluation Runner for Phase 5 Document Understanding.
Processes 3PL benchmark datasets across diverse layout families and vendors.
Outputs:
- reports/phase5/phase5_benchmark_report.md
- reports/phase5/phase5_metrics.json
- reports/phase5/phase5_failures.json
- reports/phase5/phase5_engine_usage.json
- reports/phase5/samples/ (Visual debug bounding box images and evidence JSONs)
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from app.services.pipeline import Phase5DocumentPipeline
from app.schemas.document_type import DocumentType


def draw_visual_debug(image_bytes: bytes, evidence_page: dict, output_path: Path):
    """Draw bounding boxes and layout regions on sample image for visual inspection."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]

    # 1. Draw layout regions in thick colors
    color_map = {
        "header": (255, 100, 0),        # Blue-ish
        "metadata_block": (0, 165, 255), # Orange
        "table": (0, 200, 0),            # Green
        "totals": (0, 0, 255),           # Red
        "signature_block": (200, 0, 200),# Purple
    }

    for region in evidence_page.get("regions", []):
        rtype = region["region_type"]
        color = color_map.get(rtype, (128, 128, 128))
        bx1, by1, bx2, by2 = region["bbox_norm"]
        px1 = int(bx1 / 1000 * w)
        py1 = int(by1 / 1000 * h)
        px2 = int(bx2 / 1000 * w)
        py2 = int(by2 / 1000 * h)
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
        cv2.putText(img, f"[{rtype.upper()}]", (px1 + 4, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. Draw line bounding boxes in subtle cyan
    for line in evidence_page.get("lines", []):
        bx1, by1, bx2, by2 = line["bbox_norm"]
        px1 = int(bx1 / 1000 * w)
        py1 = int(by1 / 1000 * h)
        px2 = int(bx2 / 1000 * w)
        py2 = int(by2 / 1000 * h)
        cv2.rectangle(img, (px1, py1), (px2, py2), (220, 220, 0), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def run_benchmark(dataset_dir: Path, output_dir: Path, max_docs: int | None = None) -> dict:
    pipeline = Phase5DocumentPipeline()

    # Discover documents from transactions and invoices
    tasks = []
    # 1. Invoices
    for inv_pdf in sorted(dataset_dir.glob("invoices/*.pdf")):
        tasks.append((inv_pdf, DocumentType.INVOICE, "INVOICE"))
    # 2. Transactions (SJ, POD, RA)
    for txn_dir in sorted(dataset_dir.glob("transactions/TXN-*")):
        sj_pdf = txn_dir / "surat_jalan.pdf"
        pod_jpg = txn_dir / "pod.jpg"
        ra_pdf = txn_dir / "rate_agreement.pdf"

        if sj_pdf.exists():
            tasks.append((sj_pdf, DocumentType.SURAT_JALAN, "SURAT_JALAN"))
        if pod_jpg.exists():
            tasks.append((pod_jpg, DocumentType.POD, "POD"))
        if ra_pdf.exists():
            tasks.append((ra_pdf, DocumentType.RATE_AGREEMENT, "RATE_AGREEMENT"))

    if max_docs:
        tasks = tasks[:max_docs]

    print(f"Discovered {len(tasks)} documents to evaluate across dataset {dataset_dir}...")

    correct_classification = 0
    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
    engine_usage = collections.Counter()
    total_processing_time = 0.0
    total_pages = 0
    confidence_scores = []
    table_detections = 0
    region_detections = 0
    failures = []
    sample_evidences = []

    debug_dir = output_dir / "samples"
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved_debug_count = 0

    for idx, (doc_path, expected_type, cat_name) in enumerate(tasks, 1):
        doc_id = f"EVAL-{idx:04d}-{doc_path.stem}"
        try:
            t0 = time.perf_counter()
            evidence = pipeline.process_document(doc_path, filename=doc_path.name, document_id=doc_id)
            duration_ms = (time.perf_counter() - t0) * 1000

            predicted_type = evidence["classification"]["document_type"]
            conf = evidence["classification"]["confidence"]
            confusion_matrix[expected_type.value][predicted_type] += 1
            confidence_scores.append(conf)

            if predicted_type == expected_type.value:
                correct_classification += 1
            else:
                failures.append({
                    "document_id": doc_id,
                    "file_path": str(doc_path),
                    "expected_type": expected_type.value,
                    "predicted_type": predicted_type,
                    "confidence": conf,
                    "reason": "Classification mismatch",
                })

            p_count = evidence["page_count"]
            total_pages += p_count
            total_processing_time += duration_ms

            for page in evidence["pages"]:
                engine_usage[page["engine_used"]] += 1
                if page.get("tables"):
                    table_detections += len(page["tables"])
                if page.get("regions"):
                    region_detections += len(page["regions"])

            # Save visual debug sample for the first 8 documents
            if saved_debug_count < 8:
                ingested_doc = pipeline.ingestion.ingest_file(doc_path)
                if ingested_doc.pages:
                    draw_visual_debug(
                        ingested_doc.pages[0].image_bytes,
                        evidence["pages"][0],
                        debug_dir / f"{saved_debug_count+1:02d}_{expected_type.value}_{doc_path.stem}_annotated.jpg"
                    )
                    (debug_dir / f"{saved_debug_count+1:02d}_{expected_type.value}_{doc_path.stem}_evidence.json").write_text(
                        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    saved_debug_count += 1

            if idx % 25 == 0 or idx == len(tasks):
                print(f"   Processed {idx}/{len(tasks)} docs (Acc: {correct_classification/idx*100:.1f}%, Avg Time: {total_processing_time/idx:.1f}ms/doc)")

        except Exception as e:
            failures.append({
                "document_id": doc_id,
                "file_path": str(doc_path),
                "expected_type": expected_type.value,
                "predicted_type": "ERROR",
                "confidence": 0.0,
                "reason": str(e),
            })

    total_docs = len(tasks)
    accuracy = correct_classification / total_docs if total_docs else 0.0
    avg_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    avg_time_doc = total_processing_time / total_docs if total_docs else 0.0
    avg_time_page = total_processing_time / total_pages if total_pages else 0.0

    # Calculate per-class metrics
    class_metrics = {}
    for cname in [DocumentType.INVOICE.value, DocumentType.SURAT_JALAN.value, DocumentType.POD.value, DocumentType.RATE_AGREEMENT.value]:
        tp = confusion_matrix[cname][cname]
        fn = sum(confusion_matrix[cname][pred] for pred in confusion_matrix[cname] if pred != cname)
        fp = sum(confusion_matrix[actual][cname] for actual in confusion_matrix if actual != cname)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[cname] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    metrics = {
        "dataset": str(dataset_dir),
        "total_documents": total_docs,
        "total_pages": total_pages,
        "overall_accuracy": round(accuracy, 4),
        "average_confidence": round(avg_conf, 4),
        "table_detections_count": table_detections,
        "region_detections_count": region_detections,
        "engine_usage": dict(engine_usage),
        "timing": {
            "avg_ms_per_document": round(avg_time_doc, 2),
            "avg_ms_per_page": round(avg_time_page, 2),
            "total_duration_sec": round(total_processing_time / 1000, 2),
        },
        "per_class_metrics": class_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "failure_count": len(failures),
    }

    # Save output artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase5_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "phase5_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "phase5_engine_usage.json").write_text(json.dumps(dict(engine_usage), indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate Markdown Report
    report_md = f"""# 📑 Phase 5 Document Understanding Benchmark Report

**Dataset Evaluated:** `{dataset_dir}`  
**Evaluated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Documents Processed:** {total_docs} ({total_pages} Pages)  

---

## 1. High-Level Summary Metrics
| Metric | Value |
|---|---|
| **Overall Classification Accuracy** | **{accuracy * 100:.2f}%** ({correct_classification}/{total_docs}) |
| **Average Document Confidence** | **{avg_conf * 100:.2f}%** |
| **Tables Successfully Detected** | **{table_detections}** |
| **Layout Structural Regions Detected** | **{region_detections}** |
| **Average Throughput** | **{avg_time_doc:.1f} ms / document** ({avg_time_page:.1f} ms / page) |
| **Failed Documents** | **{len(failures)}** |

---

## 2. Per-Class Precision, Recall & F1-Score
| Document Type | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **INVOICE** | {class_metrics['INVOICE']['support']} | {class_metrics['INVOICE']['precision']*100:.1f}% | {class_metrics['INVOICE']['recall']*100:.1f}% | **{class_metrics['INVOICE']['f1_score']*100:.1f}%** |
| **SURAT_JALAN** | {class_metrics['SURAT_JALAN']['support']} | {class_metrics['SURAT_JALAN']['precision']*100:.1f}% | {class_metrics['SURAT_JALAN']['recall']*100:.1f}% | **{class_metrics['SURAT_JALAN']['f1_score']*100:.1f}%** |
| **POD** | {class_metrics['POD']['support']} | {class_metrics['POD']['precision']*100:.1f}% | {class_metrics['POD']['recall']*100:.1f}% | **{class_metrics['POD']['f1_score']*100:.1f}%** |
| **RATE_AGREEMENT** | {class_metrics['RATE_AGREEMENT']['support']} | {class_metrics['RATE_AGREEMENT']['precision']*100:.1f}% | {class_metrics['RATE_AGREEMENT']['recall']*100:.1f}% | **{class_metrics['RATE_AGREEMENT']['f1_score']*100:.1f}%** |

---

## 3. OCR Engine Usage & Cascading Telemetry
| OCR Engine | Page Count | Share |
|---|---|---|
"""
    for eng, count in engine_usage.items():
        pct = count / total_pages * 100 if total_pages else 0
        report_md += f"| `{eng}` | {count} | {pct:.1f}% |\n"

    report_md += f"""
---

## 4. Visual Debugging Artifacts
Annotated bounding-box images and evidence JSONs are stored in:
`{output_dir / 'samples'}`
"""
    (output_dir / "phase5_benchmark_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nSaved Benchmark Report to: {output_dir / 'phase5_benchmark_report.md'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Benchmark on 3PL dataset.")
    parser.add_argument("--dataset-dir", default=str(ROOT.parent / "synthetic-data" / "dataset" / "v4_final_200"), help="Dataset directory")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "phase5"), help="Output report directory")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to benchmark")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    if not ds_dir.exists():
        # Fallback to v4_pilot_20 or v3_val_100
        ds_dir = ROOT.parent / "synthetic-data" / "dataset" / "v4_val_50"

    out_dir = Path(args.output_dir)
    run_benchmark(ds_dir, out_dir, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
