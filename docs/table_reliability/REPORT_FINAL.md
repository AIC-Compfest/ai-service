# REPORT_FINAL: keandalan ekstraksi tabel `ai-service`

Branch `fix/table-reliability-p0` (lokal dan `origin`), Fase 0 sampai 8. Fase 9 sengaja ditunda (OS-009, OS-010).
**Stempel:** gold tabel adalah draf sintetis (`synthetic_grounded` untuk data nyata, `constructed_exact` untuk fixture), belum direview manusia. Semua angka mengukur kecocokan dengan gold itu, bukan kebenaran mutlak. Holdout hanya 13 tabel.

## 1. Ringkasan
Sebelum pekerjaan ini, pada dokumen nyata API tidak mengembalikan tabel sama sekali bila flag mati, dan bila hidup banyak tabel yang salah header, kolom, atau baris, atau menyatu dengan blok lain. Sekarang tabel ditemukan sadar-tipe setelah klasifikasi halaman, header/kolom/baris/ringkasan ditentukan dari geometri teks, beberapa tabel per halaman dipisah, tabel lanjutan antar-halaman disambung, extractor invoice membaca hanya tabel `line_items`, dan hasilnya disertai bukti aritmetika. Perbaikan nyata dan terukur, tetapi **target K4 pada data nyata belum tercapai** (bagian 3).

## 2. Metrik sebelum dan sesudah
Data nyata, dev+val (60 + 44 dokumen, 70 tabel gold, gold v2, IoU >= 0.5):

| | Awal (router, tanpa tipe) | Awal (router bertipe, setara flag-on lama) | **Akhir (jalur API)** |
|---|---|---|---|
| Tabel prediksi | 207 | 88 | 109 |
| Deteksi P / R / F1 | 0.150 / 0.443 / 0.224 | 0.409 / 0.514 / 0.456 | **0.550 / 0.857 / 0.670** |
| Header exact (tabel yang cocok) | 26/29 | 27/34 | **42/56** |
| Jumlah kolom exact | 28/31 | 33/36 | **55/60** |
| Jumlah baris exact | 23/31 | 23/36 | **51/60** |
| Cell F1 | 0.449 | 0.332 | **0.673** |
| Nominal di dalam tabel cocok | 85/181 | 93/181 | **155/181** |
| Nominal di halaman | 90/181 | 167/181 | **163/181** |
| Ringkasan bocor ke item | 2/70 | 12/70 | **6/70** |

Per jenis dokumen (F1 deteksi, akhir): surat jalan 0.690, POD 0.857, invoice 0.678, rate card 0.583.

Fixture konstruksi (23 tabel, gold eksak): vector F1 0.553 -> **0.978** (halaman multi-tabel benar 1/4 -> **4/4**, baris exact 8/13 -> 22/22, nominal 114/200 -> 176/200); mode phrase (simulasi OCR) 0.000 -> **0.978** (multi-tabel 0/4 -> 3/4).

Holdout (13 tabel gold): baseline run 1 router F1 0.233; akhir (run 4, jalur API) P/R/F1 **0.750 / 0.692 / 0.720**, header 6/8, kolom 6/9, baris 7/9, cell F1 0.567, nominal di tabel 27/27. Run 4 identik dengan run 3.

End to end invoice (28 PDF dengan GT, 77 baris GT): flag mati 0 baris; flag hidup sebelum 50/77 baris dan 6/28 dokumen tepat; **akhir 63/77 baris dan 16/28 dokumen tepat**.

Lain-lain: `pytest` 150 lulus 1 gagal -> **337 lulus 1 dilewati**; bbox sel PyMuPDF salah 96.7% -> **0%**; PaddleOCR menjalankan **52/52** PNG dataset tanpa crash (model mobile); validasi aritmetika berlaku pada 7/28 invoice (21/132 baris, semua lulus).

Catatan baca angka: "header/kolom/baris exact" dihitung hanya pada tabel yang terdeteksi, sehingga rasionya bisa turun saat tabel terdeteksi bertambah. Ringkasan bocor naik dari 2/70 ke 6/70 karena jauh lebih banyak tabel yang kini ada untuk bocor; ini belum diperbaiki.

## 3. Target K4 (digital PDF, nyata, dev+val)
| Target | Nilai akhir | Status |
|---|---|---|
| Deteksi F1 >= 0.90 | 0.670 | tidak tercapai |
| Header exact >= 0.90 | 0.750 | tidak tercapai |
| Jumlah kolom >= 0.90 | 0.917 | tercapai (pada tabel yang cocok) |
| Jumlah baris exact >= 0.85 | 0.850 | tercapai tepat di batas |
| Cell F1 >= 0.85 | 0.673 | tidak tercapai |
| Nominal di tabel >= 0.95 | 0.856 | tidak tercapai |
| Pemisahan ringkasan >= 0.98 | 0.914 | tidak tercapai |
| Halaman multi-tabel: 0.80 nyata / 100% fixture | nyata tidak terukur (tidak ada halaman multi-tabel di gold nyata); fixture vector 4/4, phrase 3/4 | fixture vector tercapai |
Ambang tidak diturunkan dan tidak disetel pada holdout.

## 4. Apa yang dikerjakan per fase
- **0**: branch, tag `baseline-before-audit-fix`, ENV_FACTS, baseline (pytest, bbox, recall, E2E), verifikasi klaim audit awal (`AUDIT_VERIFICATION.md`).
- **1**: toolkit evaluasi (`app/tools/table_eval`, `scripts/eval_tables.py`), 18 fixture konstruksi, manifest dengan split per template, gold draf, baseline.
- **2**: perbaikan P0: grid baris PyMuPDF, tolak tabel selebar halaman, zona invoice/rate lewat config, `config_dir` dan cache, pencocokan header per token.
- **3**: penulisan ulang penemuan tabel (geometri, header berskor, kolom dari header dan data, baris/ringkasan, peran, arbitrase per tabel), strategi `text` PyMuPDF dihapus dari produksi.
- **4**: tabel sadar-tipe di pipeline, lanjutan tanpa header, stitching, extractor memfilter peran, flag default `true`, kandidat + LightGBM di jalur ekstraksi.
- **5**: aritmetika dari YAML (toleransi mutlak per mata uang), `arithmetic_validation`, `confidence_source`.
- **6**: PaddleOCR: runner terisolasi, setelan model, kunci thread, raster-twin.
- **7**: dependensi dipin, Dockerfile, pipeline tunggal, batas upload, retensi PNG, bersih dari path absolut.
- **8**: evaluasi akhir, perbaikan D-042, rekonsiliasi dokumen.

## 5. Perubahan perilaku API (semua aditif kecuali yang ditandai)
- `ENABLE_TABLE_EXTRACTOR` default `true` (perilaku berubah: tabel dan line item muncul).
- `pages[].tables[]`: `table_id`, `role`, `role_reason`, `summary_rows`, `continued_from`, `confidence_source`; **`confidence` kini skor struktural (dulu selalu 0.0)**.
- `/v1/extract/pipeline/full`: `ml_extraction`, `arithmetic_validation`, `table_groups`. `ExtractedField.confidence_source`.
- **Nilai `confidence` `line_items` berubah** dari 0.90 tetap menjadi skor struktural terlemah (0.57 sampai 0.95) atau 0.50 untuk pemindaian baris. Ini mengubah nilai yang mungkin dipakai konsumen.
- Endpoint unggah: HTTP 413 di atas `MAX_UPLOAD_SIZE_MB` (25 MB); PNG halaman lebih tua dari 24 jam dihapus.

## 6. Temuan penting
- Audit awal benar bahwa tabel salah, tetapi sebagian klaimnya keliru atau sebagian benar: lihat `AUDIT_VERIFICATION.md` (dua versi scikit-learn tidak terkonfirmasi; konfigurasi y-band sudah dihapus lebih dulu; flag mati memberi tabel kosong).
- Jalur produksi tidak memanggil candidate generation maupun LightGBM; kini dipanggil (D-037), tetapi tidak ada model yang berguna (OS-012).
- Skrip recall lama menangkap kemunduran nyata (invoice_13) yang tidak terlihat jelas pada metrik agregat; diagnosis awal saya salah dan dikoreksi (D-042).
- Default PaddleOCR adalah PP-OCRv6 medium (133 MB, sekitar 70 sampai 120 detik per halaman scan di CPU 6 inti); PP-OCRv5 mobile (12 MB) sekitar 3 kali lebih cepat. oneDNN gagal pada Paddle 3.3.1.
- PyMuPDF `find_tables` sendirian: F1 0.480, recall 0.514 tidak berubah oleh perbaikan; kini menang pada sekitar 30% tabel akhir.

## 7. Blocked, tidak terverifikasi, risiko
- **Build image Docker**: tidak ada daemon. Dockerfile belum pernah dibangun.
- **`uvicorn` tidak terpasang** di venv; server sesungguhnya belum dijalankan. Pengganti: `TestClient` dan tes subprocess dari direktori lain.
- **scikit-learn**: model klasifikasi di-pickle dengan 1.7.2 dan 1.9.0; venv dan manifest 1.5.2 (`InconsistentVersionWarning`). Latih ulang pada versi yang dipin.
- **Akurasi OCR** tidak diukur (tidak ada teks gold); kesepakatan mobile vs default hanya antar model.
- **Raster nyata**: raster-twin hanya 12 tabel invoice; tidak ada gold tabel untuk PNG dataset.
- **Paddle 3.0.x dengan oneDNN** (kemungkinan jauh lebih cepat) belum dicoba; `.venv-exp` tidak dibuat.
- **Konkurensi nyata** dan klien Go tidak diuji.
- **Model LightGBM**: hanya inferensinya diuji, bukan kualitasnya.
- **Sel tergabung PyMuPDF** masih bisa menang bila tidak ada kandidat lain (OS-013).
- **Halaman lanjutan berklasifikasi UNKNOWN** tidak ikut `line_items` (OS-011).
- **Grand total turunan** membuat cek aritmetika sirkular (OS-008); diskon dan PPh tingkat dokumen tidak dimodelkan.
- Pengecekan paired per tabel antar versi tidak dibuat; perbandingan sebelum/sesudah memakai agregat.
- Skrip `test_*.py` di `scripts/` dengan path yang diperbarui tidak dijalankan ulang.

## 8. Keputusan yang menunggu pemilik
1. Apakah model mobile PaddleOCR menjadi default (3 kali lebih cepat, kesepakatan angka 0.99, akurasi belum terukur)?
2. Apakah backend memakai ambang pada `confidence` `line_items`? (Nilai berubah dari 0.90 tetap.)
3. Review manusia atas gold; paket cek ada di `out/review/gold_check/` (CHECKLIST belum diisi), dan label untuk melatih LightGBM selain POD (OS-012).
4. Membangun image Docker dan menjalankan `/healthz` di mesin yang punya Docker; melatih ulang klasifikasi pada scikit-learn yang dipin.
5. Apakah Paddle 3.0.x di `.venv-exp` dicoba untuk mempercepat OCR scan.
6. Apakah Fase 9 (satuan di utils/pod/surat_jalan, POD dan surat jalan membaca tabel) dikerjakan.

## 9. Reproduksi
```powershell
cd ai-service
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_table_predictions.py --config api_flag_on --splits dev,val --out out\pred\x
.\.venv\Scripts\python.exe scripts\eval_tables.py --gold-dir gold\regression --pred-dir out\pred\x --manifest gold\manifest.json --splits dev,val --row-content-dir gold\row_content --allow-unreviewed --report out\reports\x.json
.\.venv\Scripts\python.exe scripts\compare_reports.py --group all --label "x=out\reports\x.json"
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe scripts\baseline\e2e_line_items.py on
.\.venv\Scripts\python.exe scripts\paddle_isolated_run.py --out out\paddle     # 52 PNG, satu proses per gambar
```
`--splits holdout` hanya untuk run akhir. Dokumen terkait: `DECISIONS.md` (D-027 sampai D-043), `OUT_OF_SCOPE_NOTES.md` (OS-001 sampai OS-014), `CONFIDENCE.md`, `REPORT_FASE_0.md` sampai `REPORT_FASE_8.md`.
