# OUT_OF_SCOPE_NOTES.md — Temuan di Luar Scope

Dokumen ini mencatat temuan penting yang ditemukan saat audit tetapi
di luar scope tugas `feat/table-reliability`.

---

## OS-001: `StrEnum` Incompatibility — Python 3.10

**File:** `app/services/labeling/schema.py:4`  
**Temuan:** `from enum import StrEnum` gagal di Python 3.10.11 (baru ada di 3.11).  
**Impact:** `tests/unit/test_labeling_phase2.py` selalu gagal import.  
**Rekomendasi:** Update ke Python 3.11+ atau gunakan `str, Enum` workaround:
```python
class FieldLabel(str, Enum):  # Python 3.10 compatible
    ...
```

---

## OS-002: `parse_amount` Mengembalikan float, Bukan Decimal

**File:** `app/services/extraction/utils.py:367`  
**Temuan:** `parse_amount()` mengembalikan `float`, yang melanggar aturan
"no float for money". Semua extractor (invoice, pod, surat_jalan, rate_agreement)
memakai helper ini.  
**Rekomendasi:** Refactor `parse_amount` agar mengembalikan `Decimal | None`
dalam scope pekerjaan berikutnya.

---

## OS-003: Anti-Pattern Pelatihan Otomatis di `DocumentClassifier`

**File:** `app/services/classification/classifier.py`  
**Temuan:** Disebutkan dalam spesifikasi bahwa ada anti-pattern "melatih
otomatis saat inisialisasi service" di `DocumentClassifier`. Ini berpotensi
menyebabkan startup time yang tidak terduga di produksi.  
**Rekomendasi:** Pisahkan pelatihan dari inisialisasi service; pelatihan
hanya dilakukan oleh script eksplisit (bukan `__init__`).

---

## OS-004: Hardcoded bbox_norm Tolerance di extraction/utils.py

**File:** `app/services/extraction/utils.py:106-107`  
**Temuan:** `y_tolerance=20.0` dan `anchor_x2 - 10.0` hardcoded di
`find_key_value_horizontal`. Ini bisa gagal pada dokumen dengan DPI berbeda.  
**Rekomendasi:** Buat toleransi relatif terhadap font size atau page size.

---

## OS-005: `BBoxCropVerifier._crop_and_ocr` — Stub "HAS_CONTENT"

**File:** `app/services/extraction/verification.py:247`  
**Temuan:** Verifier selalu mengembalikan `"HAS_CONTENT"` sebagai placeholder,
bukan hasil OCR sesungguhnya. Verification report tidak dapat dipercaya.  
**TODO comment:** `# TODO: integrate lightweight Tesseract call here for production use.`  
**Rekomendasi:** Integrasikan Tesseract atau PaddleOCR ringan untuk crop
verification yang sesungguhnya.

---

## OS-006: Magic Number y=250 dan y=700 di invoice.py

**File:** `app/services/extraction/invoice.py:178, 239, 435`  
**Temuan:** Koordinat piksel hardcoded seperti `y < 150`, `y > 250`,
`200 <= y <= 700` dipakai untuk filter baris. Ini bisa rusak jika resolusi
atau ukuran halaman berbeda.  
**Rekomendasi:** Gunakan fraksi halaman (0.0–1.0) konsisten dengan
`bbox_norm` yang sudah ada di evidence.

---

## OS-007: `_detect_layout_regions` — Confidence Hardcoded

**File:** `app/services/ocr/engine.py:473, 481, 489, 497, 505`  
**Temuan:** Layout region confidence: header=0.95, metadata=0.92, table=0.94,
totals=0.95, signature=0.90 — semuanya hardcoded tanpa sinyal nyata.  
**Catatan:** Ini terkait langsung dengan tugas kita (tabel) dan AKAN
diperbaiki di Langkah 2/3. Hanya region bukan-tabel yang dicatat di sini.

---

## OS-008: `grand_total` Dihitung Ulang dari Subtotal + Tax

**File:** `app/services/extraction/invoice.py:262-264`  
**Temuan:** Jika `grand_total` tidak ditemukan, dihitung ulang sebagai
`subtotal + tax_amount`. Ini bisa menyembunyikan kesalahan parsing dan
melanggar "No Hallucination Rule" (PRD §60) — nilai yang dihitung ulang
seharusnya diberi tanda berbeda dari nilai yang ditemukan secara langsung.  
**Rekomendasi:** Tandai field sebagai `method=ExtractionMethod.DERIVED`
atau `is_calculated=True`.

---
## OS-009: Satuan di `utils.py`, `pod.py`, `surat_jalan.py` (K11, Fase 9 ditunda)
**Temuan:** Penanganan satuan (kg/koli/dus) di extractor POD dan Surat Jalan belum diseragamkan dengan registry kandidat. Tidak diubah karena Q1=A (hanya tabel).
**Rekomendasi:** Satukan parser satuan dengan `candidates/parsers.py`, lalu tambah test per satuan.

## OS-010: Extractor POD dan Surat Jalan belum memakai tabel (K12, Fase 9 ditunda)
**Temuan:** Hanya `invoice.py` dan `rate_agreement.py` membaca `pages[].tables` (difilter `role`). POD dan Surat Jalan tetap berbasis baris teks. Tabel sudah tersedia di evidence untuk keduanya (surat jalan F1 deteksi 0.724).
**Rekomendasi:** Setelah gold direview manusia, konsumsi tabel `role=line_items` untuk item barang.

## OS-011: Halaman lanjutan tabel yang diklasifikasi UNKNOWN masuk segmen sendiri
**Temuan:** Halaman lanjutan tanpa header sering berkeyakinan < 0.45 (mis. 0.345) sehingga `segmenter.py` memisahkannya sebagai segmen UNKNOWN. Tabelnya kini tersambung di `table_groups` (D-036), tetapi extractor invoice hanya membaca halaman segmennya sendiri sehingga baris halaman itu tidak ikut `line_items`.
**Rekomendasi:** Implementasikan aturan "UNKNOWN berkeyakinan rendah digabung ke segmen tetangga" yang sudah tertulis di docstring `segmenter.py` tetapi belum dikodekan; ukur dampaknya pada metrik klasifikasi/segmentasi dulu.

## OS-012: Model LightGBM belum berguna (D-037)
**Temuan:** Hanya satu bundle (`POD.pihak_pertama_nama`, accepted 0). Label hanya untuk POD (34 dokumen, draf LLM). Nama field ML tidak sama dengan nama field extractor aturan, sehingga promosi nilai ML tidak pernah terpicu.
**Rekomendasi:** Label manusia untuk INVOICE/SURAT_JALAN/RATE_AGREEMENT, latih ulang dengan split per-template, tambah pemetaan nama field ML -> extractor.
