# Table & Line-Item Reliability — PLAN

**Branch:** `feat/table-reliability`  
**Dibuat:** 2026-10-14  
**Status:** 🟡 Fase 0 — baseline verification in progress

---

## 0. Ringkasan Masalah

Ekstraksi tabel dan line item tidak andal karena:

1. `_detect_tables()` di `OCREngine` memakai heuristik y-band (0.32–0.76) dan confidence hardcoded 0.94.
2. `TokenGridTableProvider` hanya dipakai tool offline (`build_candidate_dataset.py`), TIDAK dipanggil dari `CandidatePipeline` runtime.
3. Confidence hardcoded di mana-mana — tidak ada sinyal nyata kebenaran.
4. `li_rows_*.csv` dan `li_cells_*.csv` kosong (hanya header).
5. Deteksi "digital vs scan" terlalu kasar: hanya `len(words) < 5`.
6. Alias ambigu: `jumlah` tercatat sebagai `quantity` sekaligus `line_total`.
7. EasyOCR bbox per frasa, word-level diestimasi (`_estimated: true`).

---

## 1. Fase dan Langkah Implementasi

### Fase 0 — Recon & Baseline (SEKARANG)
- [x] Baca semua file kunci yang disebutkan
- [x] Buat branch `feat/table-reliability`
- [x] Jalankan test suite dan catat kegagalan pre-existing
- [x] Buat harness `scripts/benchmark_tables.py`
- [x] Jalankan baseline benchmark (OCR unavailable; result is a capability baseline, not accuracy)
- [x] Buat PLAN.md, DECISIONS.md

**Done criteria:** PLAN.md, hasil test awal, baseline.json, daftar file dibaca.

### Langkah 1 — Deteksi Digital vs Scan Per Halaman
**File baru:** `app/services/preprocessing/page_kind.py`
- Sinyal: rasio luas union bbox kata / luas halaman, rasio luas gambar, full-page image flag, invisible text layer (render mode 3)
- Output: `page_kind ∈ {DIGITAL, SCAN, HYBRID_OCR_LAYER, MIXED}`
- Threshold di `config.py` sebagai feature flag
- Pertahankan `len(words) < 5` sebagai safety net
- **Risk:** PyMuPDF `get_texttrace()` API bisa berbeda antar versi — harus cek versi terpasang
- **Test:** PDF sintetis deterministik (5 kasus)
- [x] Implemented and regression-tested (DIGITAL, SCAN, HYBRID_OCR_LAYER,
  MIXED, logo, and near-empty footer).

### Langkah 2 — Model Tabel Terpadu dan Router Provider
**File baru:** `app/services/candidates/tables/router.py`
- Interface `TableProvider.tables(page, doc_type, document_id) -> list[Table]`
- `TableExtractor` mencoba provider sesuai `TABLE_PROVIDER_ORDER` dan `page_kind`
- Ganti `OCREngine._detect_tables()` (y-band hardcoded) dengan pemanggilan router
- Pertahankan kompatibilitas field evidence lama + tambah field baru
- **Risk:** Harus memastikan `extraction/invoice.py` dan test lama tidak pecah
- [x] Router and additive evidence serialization implemented behind
  `ENABLE_TABLE_EXTRACTOR=false`; the first registered provider is token-grid.
  PyMuPDF and PP-Structure providers remain separate later steps.

### Langkah 3 — Validasi Aritmetika sebagai Confidence Sungguhan
**File baru:** `app/services/validation/arithmetic.py`
- Cek per baris: `qty × rate ≈ line_total`
- Cek per dokumen: `Σ line_total ≈ subtotal`, `subtotal + PPN - PPh ≈ grand_total`
- Koreksi OCR satu-karakter dengan `suggested_value` (tidak timpa raw)
- Ganti semua confidence hardcoded
- Reason codes: `ARITH_ROW_FAIL`, `ARITH_DOC_FAIL`, dll.
- **Risk:** Format angka Indonesia vs Inggris bisa muncul bersama dalam satu dokumen
- **Test:** 10+ kasus termasuk ambiguitas locale, koreksi OCR, diskon, PPh
- [x] Decimal-only row and document validation implemented behind
  `ENABLE_ARITHMETIC_VALIDATION=false`; it emits suggestions without replacing
  observed OCR values.

### Langkah 4 — Provider `pymupdf_find_tables` (PDF Digital)
**File baru:** `app/services/candidates/tables/pymupdf_provider.py`
- Gunakan `Page.find_tables()` — **harus cek API versi terpasang dulu**
- Strategi `lines` dan `text`, pilih yang terbaik
- Normalisasi koordinat konsisten dengan `process_pdf_page`
- Dukung banyak tabel per halaman
- **Risk:** API `find_tables` hanya tersedia mulai PyMuPDF ≥ 1.23.x
- [x] Implemented with `lines` then `text` strategies, normalized coordinates,
  deduplication, and multi-table support. It is reachable only through the
  existing table-extractor feature flag.

### Langkah 5 — Perbaiki `TokenGridTableProvider` dan Sambungkan ke Runtime
**Perbaikan besar di:** `app/services/candidates/tables/token_grid.py`
**Koneksi ke:** `app/services/candidates/pipeline.py`
- Header multi-baris, kolom dari klaster tepi kanan, atom sel = frasa OCR
- Pisahkan baris total/subtotal/PPN ke `summary_rows`
- Banyak tabel per halaman, stitching antar halaman per segmen
- Hilangkan ambiguitas `jumlah` dengan konteks posisi
- `_parse_cell` kota: validasi dengan gazetteer, tidak boleh `ok=True` bila unknown
- **Risk:** Perubahan ini akan mempengaruhi test kandidat yang sudah ada
- [x] Repeated headers now segment multiple token-grid tables, and total-like
  rows are emitted as `summary_rows` rather than line-item rows.
- [x] City line-item cells are accepted only when their alias appears in the
  configured gazetteer; unknown values remain explicit review candidates.
- [x] Ambiguous `jumlah` is resolved from the versioned table-resolution
  configuration, observed values, and the presence of a preceding rate field.
- [x] Token-grid combines multi-line headers up to the configured row limit,
  only when a single row is not already a valid header.
- [x] Candidate runtime can stitch consecutive-page tables with the same
  provider and normalized headers behind `ENABLE_TABLE_STITCHING=false`.

### Langkah 6 — Migrasi Raster OCR ke PaddleOCR
**Refactor:** `app/services/ocr/engine.py`
- [x] PaddleOCR menjadi satu-satunya backend OCR raster; EasyOCR dihapus dari
  runtime dan dependency manifest.
- [x] Kontrak `PageOCREvidence.to_dict()` tetap sama; `engine_used` kini
  melaporkan `paddleocr` untuk halaman raster.
- [x] Adapter membaca hasil publik PaddleOCR v3 tanpa mengakses kelas privat.
- [x] CPU oneDNN/MKLDNN dapat dikonfigurasi dan default aman dimatikan setelah
  smoke test menemukan kegagalan runtime pada build yang terpasang.
- **Risk:** Model PaddleOCR perlu tersedia di cache/diunduh saat inisialisasi;
  kompatibilitas dependency tetap diverifikasi lewat smoke test nyata.

### Langkah 7 — Preprocessing untuk Scan/Foto
**File baru:** `app/services/preprocessing/ocr_prep.py`
- [x] Deskew, contrast normalization, dan adaptive upscale menjaga transformasi
  balik koordinat ke ruang gambar sumber.
- [x] Guard A/B menjalankan versi mentah dan prepared, lalu memilih confidence
  rata-rata dan jumlah baris secara deterministik; seri memilih raw.
- [x] Metadata transformasi serta skor pemilihan ditambahkan secara additive di
  `quality.ocr_preprocessing` hanya saat `ENABLE_OCR_PREPROCESS=true`.
- [ ] Perspective correction dan stamp color filtering ditunda: tidak ada
  detektor geometri/stamp yang tervalidasi, sehingga tidak aman diaktifkan.
- **Risk:** Inference pada gambar DATASET nyata belum memberikan hasil proses
  yang dapat ditangkap runner Windows; jangan membuat klaim benchmark sebelum
  kegagalan runtime tersebut terdiagnosis.

### Langkah 8 — Gold Set Tabel dan Evaluasi
**File baru:** `app/tools/prelabel_tables.py`, `scripts/eval_tables.py`
- [x] Draft gold berasal dari evidence tabel dan selalu `reviewed=false`.
- [x] Evaluasi exact normalized-row hanya menerima gold `reviewed=true`.
- [ ] Pre-label LLM dan gold nyata ditunda sampai inference DATASET stabil.
- **Risk:** Belum ada dokumen nyata berformat gold yang sudah direview manusia;
  metrik evaluasi tidak boleh diklaim sebagai accuracy.
- [x] Human-review, accuracy, and PaddleOCR Windows diagnostic protocol is
  documented in `GOLD_REVIEW_AND_PADDLE_DIAGNOSTICS.md`.

### Langkah 9 — PP-Structure untuk Tabel pada Scan
**File baru:** `app/services/candidates/tables/ppstructure_provider.py`
- Cek API versi PaddleOCR yang terpasang untuk PP-Structure
- Fallback elegan jika model tidak tersedia
- **Risk:** API PP-Structure bisa berubah antar versi PaddleOCR

### Langkah 10 — Model Awal Baris dan Klasifikasi Kolom (LightGBM)
- Row-start classifier dan column classifier
- Hanya diaktifkan bila mengalahkan baseline aturan
- Train manual, bukan otomatis saat inisialisasi
- **Risk:** Belum ada cukup data berlabel

---

## 2. Dependensi Antar Langkah

```
Fase 0 → 1 → 2 → {3, 4, 5} → 6 → 7
                              ↘ 8 → 9 → 10
```

Langkah 3, 4, 5 bisa dikerjakan paralel setelah Langkah 2 selesai.  
Langkah 8 bisa dimulai paralel dari awal sebagai tooling.

---

## 3. Risiko Utama

| Risiko | Langkah | Mitigasi |
|--------|---------|---------|
| PyMuPDF `find_tables` API tidak tersedia | 4 | Cek versi, fallback ke token_grid |
| PaddleOCR tidak bisa diunduh | 6, 9 | Skip test dengan alasan, mock di test |
| Kompatibilitas numpy/opencv dengan Paddle | 6 | requirements-paddle.txt terpisah |
| Preprocessing memperburuk scan bersih | 7 | Guard A/B, hanya aktif bila kualitas rendah |
| Perubahan confidence memutus test lama | 2, 3 | Feature flag default aman |
| Tidak ada data berlabel untuk ML | 10 | Bangun pipeline dulu, tandai "belum dilatih" |

---

## 4. Asumsi

1. Dataset gambar sampel ada di `D:\AIC\DATASET` (dari audit script argumen default).
2. PyMuPDF versi yang terpasang mendukung `get_images()` dan `get_text("words")`.
3. Lingkungan Python 3.10.11 (ditemukan dari test run).
4. `StrEnum` tidak tersedia (Python 3.10) — ini bug pre-existing, bukan dari perubahan kita.

---

## 5. Perubahan yang Dilarang

Sesuai scope, TIDAK mengubah:
- Classifier (`classification/classifier.py`)
- Segmenter (`segmentation/segmenter.py`)
- LightGBM matching
- API publik di luar yang disebutkan

Temuan di luar scope dicatat di `OUT_OF_SCOPE_NOTES.md`.

---

## 6. Log Kemajuan

| Tanggal | Fase/Langkah | Status | Catatan |
|---------|-------------|--------|---------|
| 2026-10-14 | Fase 0: Recon | 🟡 In Progress | Membaca semua file kunci |
| 2026-10-14 | Branch dibuat | ✅ Done | `feat/table-reliability` |
| 2026-10-14 | PLAN.md dibuat | ✅ Done | — |
| 2026-10-15 | Dataset boundary | ⚠️ Code changed, unverified | Benchmark now has an explicit allow-list of four valid `DATASET` documents; no directory-wide discovery is permitted. Test execution remains blocked by the missing Python interpreter. |
| 2026-10-15 | Initial test run | ⚠️ Historical result: 114 passed, 2 pre-existing failures | The reported baseline remains unverified in this workspace: `.venv` points to a missing Python 3.11 executable. The two reported failures are `test_noisy_scan_signature_low_confidence` (unseeded random noise) and `test_document_ingestion_jpg` (PNG fixture versus JPEG assertion). |
| 2026-10-15 | Environment verification | ✅ Done with sandbox elevation | Python 3.11.9 virtualenv and PyMuPDF 1.28.2 run outside the restricted sandbox. `easyocr` and `paddleocr` are not installed. Target table tests: 6 passed. |
| 2026-10-15 | Canonical capability baseline | ✅ Done, not accuracy | `out/benchmarks/20261015T010604Z_baseline-easyocr.json`: 4 allow-listed documents, table/line-item presence 4/4, mean OCR confidence 0.748, mean 15,642.4 ms/page. No reviewed gold or arithmetic validation exists, so no accuracy is claimed. |
| 2026-10-15 | Step 1: page-kind routing | ✅ Done | Feature-flagged per-page routing/evidence added. Deterministic PDF tests cover digital, scan, invisible OCR layer, mixed page, large-logo page, and near-empty footer; 8 related tests passed. |
| 2026-10-15 | Step 2: invoice row filter | ✅ Partial | Removed vendor-specific `shipment_id != "Logistik"` exception; structural identifier/total validation is unit-tested. Unified table router remains in progress. |
| 2026-10-15 | Step 2: table router | ✅ Done | Added scored, feature-flagged table router, CandidatePipeline diagnostics, and backward-compatible evidence metadata. |
