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

---
## OS-013: Tabel PyMuPDF dengan Sel Tergabung Masih Bisa Menang
**Temuan:** `find_tables` strategi `lines` kadang mengembalikan tabel dengan satu sel berisi seluruh teks halaman dan baris yang menempel (mis. invoice_13 sebelum D-042). Penalti sel tergabung tidak membantu (D-043) karena penyebabnya tidak adanya kandidat lain, bukan skor. Setelah D-042 kandidat token-geometri ada dan menang pada dokumen itu; pada tata letak lain yang tidak dikenali token-geometri, tabel tergabung tetap bisa terpilih.
**Rekomendasi:** pecah sel tergabung memakai posisi kata (kata PyMuPDF sudah punya bbox) alih-alih memakai teks sel, atau tolak tabel yang selnya memuat banyak kata dari kolom lain.

## OS-014: Skrip Pengukuran Lama Perlu Dibaca dengan Hati-hati
**Temuan:** `measure_amount_recall.py` membandingkan digit nominal dengan digit sel; nominal bertitik desimal ("5,400,000.00" menjadi 540000000) tidak pernah cocok dengan "5400000". Itu menjelaskan 6 miss yang konstan pada `invoice_4_variasi_1/2` sebelum dan sesudah perbaikan (dugaan, tidak diuji satu per satu). Skrip ini tidak boleh dipakai sebagai bukti akurasi.
**Rekomendasi:** pakai `scripts/eval_tables.py`; perbaiki normalisasi desimal bila skrip lama masih dipakai.

---
## OS-015: Tabel Berdampingan Tanpa Garis dengan Header Berbeda Digabung
**Temuan (stres, seed 1000-1239):** recall 0.47 pada halaman side_by_side, 14 halaman gagal dengan F4 (satu tabel prediksi menutup dua tabel gold). Celah di antara dua tabel sering hanya 3% lebar halaman; tabel tunggal dengan kolom pendek menghasilkan celah serupa, dan percobaan memisah berdasarkan "header di kedua sisi" merusak tabel tunggal (D-048). Tabel bergaris tidak terpengaruh karena PyMuPDF memisahnya.
**Rekomendasi:** gunakan struktur baris (jumlah dan tinggi baris kiri dan kanan berbeda) atau garis tepi tabel; ukur pada PDF nyata yang punya tabel berdampingan (gold nyata saat ini tidak punya).
## OS-016: PyMuPDF `find_tables` Tidak Menemukan Tabel Tanpa Garis
**Temuan:** pada stres, recall 0.00 untuk halaman tanpa garis (96 halaman); pada tabel bergaris deteksi 0.99. Strategi `text` dihapus karena 1 sampai 5 detik per halaman (D-034). Tabel tanpa garis dibaca oleh provider token-geometri (recall 0.88).
## OS-017: Header Dua Tingkat dengan Sub-label di Baris Pertama (invoice_14)
**Temuan:** pada `invoice_14` sub-label "ONDOL"/"PASIR" berada di baris yang sama dengan nilai pertama sehingga sel pertama menjadi "ONDOL 480.000". Menyumbang -3 nominal di tabel (155 -> 152) dan -2 baris pada E2E setelah tabel PyMuPDF yang lama (berisi baris campur) tidak lagi menang.
## OS-018: Label Grup Menempel pada Blok Key-Value
**Temuan:** label di atas blok key-value ("Agreement") ikut ditempel ke header dua kolom pada rate card. Tidak berdampak pada gold (bukan tabel item) tetapi mengotori `headers` tabel `info`.
