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

## Instalasi OCR

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

## Reproduksi setelah Python 3.11.9 dipulihkan

```powershell
cd D:\AIC\ai-service
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\benchmark_tables.py --doc-dir D:\AIC\DATASET --label baseline
```

Benchmark hanya boleh menggunakan empat path allow-listed di
`scripts/benchmark_tables.py`; artifact `out/` tidak akan di-commit.
