# Table & Line-Item Reliability — Report

## Status

Recon berjalan. Environment Python harus dijalankan dengan izin sandbox
elevated, tetapi tidak perlu direkonstruksi. Tidak ada perubahan source
table/OCR yang dibuat atau di-commit pada sesi ini.

## Verifikasi environment

Perintah yang dicoba:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_benchmark_tables_dataset.py tests/unit/test_candidate_tables.py tests/unit/test_ocr_token_contract.py -q
```

Hasil di sandbox terbatas: launcher ditolak (`Access is denied`). Hasil setelah
izin elevated: Python 3.11.9, PyMuPDF 1.28.2, dan target test table-reliability
`6 passed in 2.19s`. `easyocr` dan `paddleocr` belum terpasang.

## Hasil yang belum dapat diklaim ulang

- Historical baseline: `114 passed, 2 failed, 1 warning in 46.73s`.
- Dua failure pre-existing yang dilaporkan: signature noisy scan nondeterministik
  dan MIME fixture PNG/JPEG yang tidak cocok.
- PaddleOCR/PP-Structure belum terpasang atau diuji. Full-suite summary belum
  tertangkap utuh oleh runner; target table tests lulus 6/6.

## Benchmark canonical

Perintah yang dijalankan:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_tables.py --doc-dir D:\AIC\DATASET --label baseline
```

Artifact tidak di-commit: `out/benchmarks/20261015T010320Z_baseline.json` dan
`out/benchmarks/20261015T010604Z_baseline-easyocr.json`.

Empat file allow-listed ditemukan. Run pertama tanpa EasyOCR menghasilkan nol
output dan dipertahankan hanya sebagai diagnosis availability. Setelah EasyOCR
terpasang, run `baseline-easyocr` menghasilkan table dan line-item presence
4/4, mean OCR confidence `0.748`, dan mean `15,642.4 ms/page`. Surat Jalan
adalah satu-satunya dokumen di bawah confidence 0.6 (`0.5275`). Tidak ada
reviewed gold atau arithmetic validation, sehingga angka ini bukan akurasi.

## Langkah 6 — migrasi PaddleOCR

EasyOCR sudah dikeluarkan dari runtime dan `requirements.txt`; OCR raster kini
selalu menggunakan PaddleOCR. Adapter memakai field hasil publik PaddleOCR v3
dan mengubahnya ke kontrak evidence lama (`lines`, token estimasi, koordinat
ternormalisasi, serta confidence heuristik OCR), tanpa mengubah struktur API
evidence.

Environment yang diverifikasi berisi PaddleOCR `3.7.0` dan PaddlePaddle
`3.3.1`. Smoke inference nyata berhasil dengan CPU MKLDNN dimatikan. Konfigurasi
default tersebut disengaja: MKLDNN pada runtime terpasang gagal dengan error
oneDNN PIR. `pandas` juga dibatasi `<3` karena dependensi PaddleX terpasang
gagal import pada pandas `3.0.6`. Ini adalah kompatibilitas lingkungan, bukan
aturan atau data khusus dokumen.

PP-Structure belum diaktifkan ataupun diklaim tersedia; itu tetap Langkah 9.

## Instalasi OCR historis

Instalasi serial `pip install easyocr` selesai pada percobaan ulang. Verifikasi:
EasyOCR `1.7.2`, Torch `2.14.1+cpu`, CUDA `False`. PaddleOCR/PP-Structure belum
diinstal atau diuji.

## Langkah 1 — Page-kind routing

`PAGE_KIND_ROUTING` tetap `false` secara default. Saat diaktifkan, evidence
per halaman menambahkan `page_kind`, reason, dan raw signals (word coverage,
image coverage, full-page image, invisible text, jumlah kata, dan jumlah
gambar). Routing SCAN menggunakan raster OCR; DIGITAL, MIXED, dan
HYBRID_OCR_LAYER mempertahankan ekstraksi text layer.

Test deterministik menggunakan PyMuPDF 1.28.2 dan mencakup digital, full-page
scan, scan plus invisible render-mode-3 OCR, mixed, logo besar, serta near-empty
footer. Perintah verifikasi:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests/unit/test_page_kind.py tests/unit/test_ocr_token_contract.py tests/unit/test_candidate_tables.py tests/unit/test_benchmark_tables_dataset.py -q
```

Hasil: `8 passed`.

## Langkah 2 — table router

The invoice extractor no longer excludes a row because its shipment id is
`Logistik`. It now requires an identifier-like value or a parseable line total.
The focused unit test passed (`1 passed`).

`TableExtractor` is feature-flagged by `ENABLE_TABLE_EXTRACTOR=false` and
returns the unified internal `Table` model. It falls back after empty provider
results, selects non-empty results by a structural completeness heuristic (not
accuracy), and preserves all tables from the selected provider. The token-grid
provider is the only registered provider in this step. OCR-time discovery uses
no fabricated document type; CandidatePipeline performs alias-aware discovery
after type is known. Evidence fields `provider`, `reason`, `validation`,
`columns`, and `cells` are additive to the established table fields.

## Langkah 3 — arithmetic validation

`app/services/validation/arithmetic.py` validates `quantity × rate = line total`
and document subtotal / tax / withholding-tax / grand-total relationships with
`Decimal`. It is disabled by default and does not claim accuracy or mutate OCR
values. Failure metadata includes reason codes and a review-only suggestion.

## Langkah 4 — PyMuPDF provider

The digital-PDF provider uses PyMuPDF 1.28.2 `find_tables(**kwargs)` with
`lines` and then `text` strategies. It normalizes coordinates to the established
0–1 evidence space, deduplicates strategy results, and preserves multiple
tables on one page. It has no dependency on PaddleOCR or PP-Structure.

## Langkah 5 — token-grid segmentation

Repeated headers now split token-grid output into separate tables. Total-like
rows are retained as `summary_rows`, not passed as candidate line items.
City line-item values now require a configured gazetteer match; unknown values
are explicitly marked rather than treated as valid cities.
The ambiguous `jumlah` header is resolved by versioned candidate configuration,
not vendor-specific or code-hardcoded data.
Multi-line table headers are joined only under the configured limit and only
when the first header row does not already match enough configured fields.
Candidate runtime now has opt-in cross-page stitching based exclusively on
provider, consecutive pages, and normalized header structure.

## Reproduksi setelah Python 3.11.9 dipulihkan

```powershell
cd D:\AIC\ai-service
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\benchmark_tables.py --doc-dir D:\AIC\DATASET --label baseline
```

Benchmark hanya boleh menggunakan empat path allow-listed di
`scripts/benchmark_tables.py`; artifact `out/` tidak akan di-commit.
