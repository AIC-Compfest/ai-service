# 📑 Phase 5 Document Understanding Benchmark Report

**Dataset Evaluated:** `..\synthetic-data\dataset\v4_pilot_20`  
**Evaluated At:** 2026-08-17 01:34:58  
**Total Documents Processed:** 75 (75 Pages)  

---

## 1. High-Level Summary Metrics
| Metric | Value |
|---|---|
| **Overall Classification Accuracy** | **69.33%** (52/75) |
| **Average Document Confidence** | **57.29%** |
| **Tables Successfully Detected** | **51** |
| **Layout Structural Regions Detected** | **168** |
| **Average Throughput** | **267.3 ms / document** (267.3 ms / page) |
| **Failed Documents** | **23** |

---

## 2. Per-Class Precision, Recall & F1-Score
| Document Type | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **INVOICE** | 15 | 83.3% | 100.0% | **90.9%** |
| **SURAT_JALAN** | 20 | 100.0% | 100.0% | **100.0%** |
| **POD** | 20 | 0.0% | 0.0% | **0.0%** |
| **RATE_AGREEMENT** | 20 | 100.0% | 85.0% | **91.9%** |

---

## 3. OCR Engine Usage & Cascading Telemetry
| OCR Engine | Page Count | Share |
|---|---|---|
| `pymupdf_vector` | 55 | 73.3% |
| `raster_ocr` | 20 | 26.7% |

---

## 4. Visual Debugging Artifacts
Annotated bounding-box images and evidence JSONs are stored in:
`reports\phase5_pilot\samples`
