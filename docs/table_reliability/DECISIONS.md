# DECISIONS.md — Keputusan Teknis dan Asumsi

Dokumen ini mencatat semua keputusan teknis yang diambil untuk mengatasi
ambiguitas selama implementasi `feat/table-reliability`.

---

## D-001: StrEnum Pre-existing Bug (Python 3.10)

**Tanggal:** 2026-10-14  
**Konteks:** `tests/unit/test_labeling_phase2.py` gagal import karena `StrEnum`
hanya tersedia di Python 3.11+, sedangkan environment memakai Python 3.10.11.  
**Keputusan:** Catat sebagai pre-existing bug. Jangan perbaiki dalam scope ini
karena mengubah `labeling/schema.py` di luar lingkup tugas.  
**Dampak:** Test tersebut akan tetap di-skip / diabaikan dengan flag
`--ignore=tests/unit/test_labeling_phase2.py` di semua test run kita.  
**Lokasi:** `app/services/labeling/schema.py:4`

---

## D-002: Lokasi Dataset Gambar Sampel

**Tanggal:** 2026-10-14  
**Konteks:** `scripts/audit_ocr_candidate_dataset.py` argumen default
`--dataset-dir D:\AIC\DATASET`. Perlu konfirmasi keberadaan folder.  
**Keputusan:** Jika folder tidak ada, buat fixture sintetis dari PDF generator
(`fitz`) untuk menguji pipeline. Tidak menghentikan pekerjaan.  
**Dampak:** Benchmark baseline akan memakai dokumen sintetis jika dataset
nyata tidak tersedia; akan ditandai jelas di laporan.

---

## D-003: Representasi Uang — Decimal bukan float

**Tanggal:** 2026-10-14  
**Konteks:** PRD mengharuskan tidak pakai `float` untuk uang. Kode lama
(`parse_amount` di `extraction/utils.py`) mengembalikan `float`.  
**Keputusan:** Modul baru (validasi aritmetika, Langkah 3) WAJIB memakai
`Decimal`. Kode lama (`parse_amount`) tidak diubah dalam scope ini karena
itu perubahan API yang lebih luas; catat sebagai limitation. Jika terpaksa
menyentuh kode lama di Langkah 3, konversi ke `Decimal` secara lokal.  
**Lokasi:** `app/services/extraction/utils.py` (kode lama)

---

## D-004: Confidence Hardcoded vs Heuristik Berlabel

**Tanggal:** 2026-10-14  
**Konteks:** Banyak confidence hardcoded (0.90, 0.94, 0.95, dll.). Untuk
mengganti ini dengan sinyal nyata, perlu data yang sudah divalidasi.  
**Keputusan:** 
- Semua confidence hardcoded akan diganti dengan formula yang
  didokumentasikan di `CONFIDENCE.md`.
- Bila belum ada data gold, formula tetap menghasilkan nilai heuristik
  TAPI diberi label eksplisit `confidence_source: "structural_heuristic"`,
  bukan berpura-pura lebih akurat dari yang sebenarnya.
- Validasi aritmetika (Langkah 3) menjadi sinyal confidence utama.

---

## D-005: Feature Flag Default

**Tanggal:** 2026-10-14  
**Konteks:** Semua perilaku baru harus di balik feature flag.  
**Keputusan:** Default semua flag aman (perilaku lama), kecuali:
- `TABLE_PROVIDER_ORDER` default ke `["token_grid"]` (tidak mengubah behavior
  saat ini, karena PyMuPDF provider belum ada sebelumnya).
- `PAGE_KIND_ROUTING` default ke `false` (pakai logika lama `len(words) < 5`).
- `OCR_BACKEND` tetap `"easyocr"` (tidak mengaktifkan Paddle otomatis).
- `ENABLE_OCR_PREPROCESS` default ke `false`.

---

## D-006: Stitching Antar Halaman

**Tanggal:** 2026-10-14  
**Konteks:** Stitching tabel lintas halaman harus terjadi di tahap
candidate/segment, bukan di tahap OCR per halaman.  
**Keputusan:** Implementasikan stitching di `CandidatePipeline.generate()`
setelah segmentasi, bukan di `OCREngine`. Field `stitched_pages` di `Table`
schema akan diisi di sana.

---

## D-007: Alias Ambigu "jumlah"

**Tanggal:** 2026-10-14  
**Konteks:** `jumlah` terdaftar sebagai alias untuk `quantity` DAN `line_total`
di `line_item_aliases.yaml`. Ini bisa menyebabkan salah assign kolom.  
**Keputusan:** Resolusi berbasis konteks posisi: jika kolom `jumlah` berada
**setelah** kolom harga/rate dan berisi nilai besar (> threshold dari config),
anggap sebagai `line_total`. Jika berada **sebelum** kolom harga dan berisi
nilai kecil, anggap `quantity`. Tambahkan ke `ambiguous_ok` list dengan
resolusi eksplisit.

---

## D-008: Kota dengan `ok=True` di `_parse_cell`

**Tanggal:** 2026-10-14  
**Konteks:** `_parse_cell` untuk kota selalu mengembalikan `ok=True`, bahkan
untuk teks yang tidak cocok di gazetteer.  
**Keputusan:** Ubah menjadi: return `ok=True` hanya jika teks cocok dengan
`gazetteer_cities.csv`, sinon return `ok=False` dengan `value_type="city_unknown"`.
Gazetteer perlu diperluas bila ada kota yang valid tapi belum terdaftar.

---

## D-009: Canonical Benchmark Dataset Boundary

**Date:** 2026-10-15  
**Context:** `DATASET` contains four approved document-type collections, but
also nested duplicate/transformed files, archives, generated candidate CSVs,
and a separate `Dataset` directory.  Including all descendants would make the
benchmark data provenance invalid.
**Decision:** `benchmark_tables.py` accepts only immediate PDF/image files in
`DATASET/invoice/{pdf,image}`, `DATASET/PoD/{pdf,image}`,
`DATASET/rate card/{pdf,image}`, and `DATASET/surat jalan/{pdf,image}`.
Synthetic fixtures require `--synthetic`; the harness never silently falls
back to them.
**Impact:** Every real-data metric can be traced to one of the four approved
document-type collections.  A human still needs to identify a reviewed gold
subset before accuracy claims are made.
