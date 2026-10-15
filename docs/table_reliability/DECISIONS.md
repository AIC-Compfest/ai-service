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
- Raster OCR sekarang memakai PaddleOCR secara eksklusif (lihat D-023).
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
**Context:** The user clarified that the only valid real dataset population is
four documents in `D:\\AIC\\DATASET`, one representative each for Invoice,
POD, Rate Agreement, and Surat Jalan. The directory also contains variants,
archives, exports, and unreviewed examples.
**Decision:** `benchmark_tables.py` uses an explicit four-file allow-list in
`CANONICAL_DATASET_DOCUMENTS`. It never discovers every file in the four
directories. Synthetic fixtures require `--synthetic`; the harness never
silently falls back to them.
**Impact:** A benchmark is incomplete when an allow-listed file is absent;
it must not substitute another file. Any manifest change requires a documented
human decision before metrics are compared.

---

## D-010: Baseline Test Failures Outside Table Reliability

**Date:** 2026-10-15
**Context:** The first complete baseline run after rebuilding Python reported
`114 passed, 2 failed`.
**Decision:** Do not change either failure in this feature branch because both
are outside the requested scope:

1. `test_noisy_scan_signature_low_confidence` creates unseeded random noise,
   so the signature score varies between runs.
2. `test_document_ingestion_jpg` selects a PNG POD fixture but asserts a JPEG
   MIME type.

**Impact:** Subsequent table-reliability tests must pass. Full-suite reporting
will preserve these two baseline failures until their owning components are
fixed separately.

---

## D-011: Project Python Requires Sandbox Elevation

**Date:** 2026-10-15
**Context:** The project virtual environment's launcher references Python 3.11.9
under the user's profile. It is available, but the restricted sandbox cannot
execute it (`Access is denied`). It runs successfully when the command is
approved outside the sandbox.
**Decision:** Use the documented `.venv` with explicit elevation only for
verification and installation commands. Do not recreate or retarget it.
**Impact:** PyMuPDF 1.28.2 and the target table tests are now verified. Package
availability and benchmark claims still require their respective commands.

---

## D-012: Canonical Baseline Is a Capability Measurement, Not Accuracy

**Date:** 2026-10-15
**Context:** The first run without EasyOCR produced a zero-output availability
artifact. EasyOCR 1.7.2 (Torch 2.14.1+cpu) was then installed and imported,
and the required four-file benchmark completed.
**Decision:** Preserve both JSON files as diagnostic artifacts, but report only
the EasyOCR run as the real-data capability baseline. It measures emitted
tables/line items, OCR confidence, and latency; it is not accuracy because no
reviewed gold or arithmetic validation is present.
**Impact:** `out/benchmarks/20261015T010604Z_baseline-easyocr.json` is not
committed. Its 4/4 table and line-item presence must not be interpreted as
correct extraction.

---

## D-013: Page-Kind Safety-Net Precedes DIGITAL Classification

**Date:** 2026-10-15
**Context:** A deterministic near-empty footer PDF had only two words but its
word boxes covered more than a low synthetic threshold. Classifying on coverage
first incorrectly returned `DIGITAL`.
**Decision:** Evaluate full-page image and invisible OCR-layer signals first,
then apply the configurable legacy word-count safety net before the DIGITAL
coverage rule. The default remains five words.
**Impact:** Pages containing only a footer cannot bypass scan routing due to a
misleading bbox-area ratio. This applies only when `PAGE_KIND_ROUTING=true`.

---

## D-014: Invoice Line Items Must Not Depend on a Vendor Name

**Date:** 2026-10-15
**Decision:** Replace the literal `shipment_id != "Logistik"` exception with a
structural row predicate: a row needs an identifier containing a digit or a
parseable line total. This keeps header/vendor text out without encoding a
specific company name.

---

## D-015: Table Discovery Does Not Invent a Document Type

**Date:** 2026-10-15
**Context:** OCR engine runs before document classification, whereas header
aliases are document-type specific.
**Decision:** The table router accepts `doc_type: str | None`. At OCR time it
uses geometry-only token-grid discovery; CandidatePipeline supplies the known
document type for alias-aware discovery. The router selects a non-empty
provider result by structural completeness score, with configured order used
only as a deterministic tie-breaker.
**Impact:** `ENABLE_TABLE_EXTRACTOR` defaults to false. Existing evidence
fields remain unchanged; `provider`, `reason`, `validation`, `columns`, and
`cells` are additive only when router output is present. Arithmetic validation
is deliberately deferred to Step 3.

---

## D-016: Arithmetic Validation Is an Additive Signal

**Date:** 2026-10-15
**Decision:** Arithmetic checks use `Decimal` end-to-end and only append
validation evidence. A failed check exposes `suggested_value` for review but
never changes an OCR cell or a public extracted field. Router scoring includes
the arithmetic signal only when `ENABLE_ARITHMETIC_VALIDATION=true`.

---

## D-017: PyMuPDF Provider Receives the Live PDF Page Only at OCR Time

**Date:** 2026-10-15
**Decision:** The internal candidate `Page` may carry a non-serializable live
PyMuPDF page solely during OCR-time discovery. Persisted OCR evidence remains
JSON-only. CandidatePipeline has no live PDF object and therefore naturally
falls back to token-grid.

---

## D-018: Token-Grid Summary Rows Are Not Line Items

**Date:** 2026-10-15
**Decision:** Rows beginning with total/subtotal/PPN-like labels are stored as
`summary_rows` and excluded from the normal table rows consumed by line-item
building. Repeated detected headers delimit independent tables on the page.

---

## D-019: City Cells Require Gazetteer Evidence

**Date:** 2026-10-15
**Decision:** Line-item city values are canonicalized only through
`gazetteer_cities.csv`. Unrecognized text retains its raw uppercase value with
`value_type=city_unknown` and `parse_ok=false`.

---

## D-020: Ambiguous Headers Are Resolved by Configuration

**Date:** 2026-10-15
**Decision:** `jumlah` mapping is controlled by
`config/candidates/table_resolution.yaml`, not code constants. Its configured
value floor and rate-field list determine whether it represents a quantity or
a line total.

---

## D-021: Multi-Line Header Limit Is Configured

**Date:** 2026-10-15
**Decision:** `token_grid.header_max_rows` in table-resolution configuration
controls header joining. A valid single header takes precedence, preventing a
first line item from being swallowed as a second header line.

---

## D-022: Cross-Page Stitching Requires Structural Continuity

**Date:** 2026-10-15
**Decision:** Candidate runtime stitches only adjacent-page tables from the
same provider with identical normalized header sequences. The behavior is
feature-flagged off by default; it does not infer continuity from vendor data.

---

## D-023: PaddleOCR Is the Sole Raster OCR Backend

**Date:** 2026-10-15
**Context:** The requested migration explicitly replaces EasyOCR rather than
adding an alternative runtime. PaddleOCR 3.7.0 and PaddlePaddle 3.3.1 are
installed in the project environment.
**Decision:** `OCREngine` initializes only PaddleOCR and adapts its public v3
result JSON (`rec_texts`, `rec_scores`, and `rec_boxes`) into the existing OCR
line/token evidence contract. EasyOCR is removed from `requirements.txt` and
there is no backend-selection fallback.
**Operational compatibility:** `pandas` is constrained below major version 3
because the installed PaddleX dependency fails to import with pandas 3.0. CPU
MKLDNN defaults to disabled because the installed Paddle runtime raised a
oneDNN PIR attribute conversion error during real inference when it was
enabled. Both language and CPU settings remain application configuration, not
document-specific constants.
**Impact:** A Paddle initialization or model-availability failure is explicit;
the service does not silently switch OCR engines. Existing vector PDF
extraction remains PyMuPDF-based.

---

## D-024: OCR Preprocessing Is Geometry-Aware and Conservative

**Date:** 2026-10-15
**Decision:** The opt-in preprocessing path evaluates raw and prepared images,
then keeps the result with the greater `(mean OCR confidence, line count)`
tuple. Ties retain raw OCR. Deskew and adaptive upscale carry an inverse
homography so all selected OCR boxes are restored to the source-image
coordinate system before evidence or table discovery.
**Scope boundary:** Perspective correction and stamp-color filtering are not
enabled without a validated detector. Applying either broadly would risk
removing document content or corrupting coordinates. This is explicitly not a
document/vendor exception.
