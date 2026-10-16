# AUDIT_VERIFICATION (Fase 0)

Status: `terkonfirmasi` / `sebagian` / `tidak` / `belum` (diverifikasi di fase lain). Tag bukti: `[diuji]` perintah dijalankan, `[kode]` dibaca dari kode.
GT dan angka dataset bersifat sintetis (bukan gold manusia).

| ID | Status | Bukti |
|---|---|---|
| T-01 flag mati = `tables=[]` | terkonfirmasi | [kode] `engine.py` `_detect_tables` return `[]`; [diuji] `e2e_line_items.py off`: 0/28 dokumen bertabel, 0 line item. Koreksi: y-band lama dihapus di commit `3e4326c`, bukan `fc41cca` (`git log -S"0.32"`) |
| T-02 token_grid `doc_type=None` degenerate | terkonfirmasi | [diuji] overlay `out/review/phase0/inv10v1.png`: tabel item asli (Description/AWB/Commodity) TIDAK terdeteksi; token_grid menghasilkan 4 "tabel" dari kop dan syarat. Sensus: router(None) rata-rata ≥3 tabel/halaman (invoice 126 tabel/40 halaman) |
| T-03 `_visual_rows` page-global | terkonfirmasi (sebagian) | [kode] + overlay: blok kop kiri dan logo kanan menyatu satu baris visual. Tabel berdampingan murni belum ada contoh nyata, diuji dengan fixture konstruksi (Fase 1) |
| T-04 kolom dari tengah antar header, tepi 0.0/1.0 | terkonfirmasi | [kode] `_columns`; overlay: garis biru kolom menjangkau tepi halaman |
| T-05 tabel tanpa akhir | terkonfirmasi | [diuji] overlay `rc13v1.png`: seluruh paragraf Notes menjadi baris item (T4, 22 "baris") |
| T-06 pitch dan urutan continuation vs ringkasan | terkonfirmasi [kode] | akan diuji dengan fixture konstruksi (Fase 3) |
| T-07 header dua baris rapuh | terkonfirmasi [kode] | uji perilaku di Fase 3 |
| T-08 router satu provider per halaman | terkonfirmasi | [diuji] sensus: invoice, PyMuPDF(lines) punya ≥1 tabel di 36/40 halaman, tetapi router memilih token_grid junk (skor fill-ratio) pada `inv10v1` |
| T-09 tanpa peran tabel | terkonfirmasi [kode] | `Table` tidak punya field role |
| T-10 strategi PyMuPDF break pertama, tanpa penolakan lebar | terkonfirmasi | [diuji] `measure_amount_recall.py`: 5 dari 36 tabel PyMuPDF menutupi >60% halaman; sensus: strategi `text` selalu 1 tabel per halaman (invoice 40/40, PoD 34/35, rate 52/52, SJ 32/32) yaitu halaman penuh |
| T-11 bbox sel PyMuPDF salah | terkonfirmasi | [diuji] `check_pymupdf_bbox.py`: 52 tabel, 1.627 sel, 1.574 salah (96,7%). Identik dengan dokumen lama |
| T-12 `_is_summary_row` startswith | terkonfirmasi [kode] | PyMuPDF tidak punya deteksi ringkasan |
| T-13 satuan di `invoice.py` | terkonfirmasi | [diuji] introspeksi sumber: `<150`, `>250`, `200..700`, `25.0` masih ada pada `bbox_norm` skala 0..1 |
| T-14 `table_id`, stitching | terkonfirmasi [kode] | uji perilaku di Fase 4 |
| T-15 `parse_table_columns` header kosong | terkonfirmasi, lebih parah | [diuji] header `["", "Qty", "Total"]` membuat nilai `"x"` dipetakan ke 10 field kanonik sekaligus |
| T-16 token palsu untuk OCR tingkat baris | terkonfirmasi [kode] | `context.build_page` |
| T-17 satuan di utils/pod/surat_jalan | terkonfirmasi sebagian [kode] | DITUNDA (OS-009) |
| T-18 POD/SJ tidak memakai tabel | terkonfirmasi [kode] | DITUNDA (OS-010) |
| T-19 alias bermasalah | sebagian | `item`→`line_number`, `jumlah` ambigu terbaca di YAML; dampak diukur Fase 3 |
| T-20 aritmetika | terkonfirmasi [kode] | kosakata `_QUANTITY/_RATE/_TOTAL` hardcoded; hanya qty×rate; `validate_document_totals` tidak dipakai produksi |
| T-21 confidence hardcoded | terkonfirmasi [kode] | token_grid 0.8, PyMuPDF 0.0, invoice 0.90 |
| T-22 `eval_tables.py` | terkonfirmasi [kode] | set baris, `casefold`, `prediction["tables"]` di root |
| T-23 `benchmark_tables.py:290` `engine.reader` | terkonfirmasi [kode] | |
| T-24 Dockerfile | terkonfirmasi [kode] | hanya `COPY app` |
| T-25 from_config per halaman | terkonfirmasi [kode] | |
| T-26 Settings dobel | terkonfirmasi [kode] | |
| T-27 paket ganda venv | terkonfirmasi sebagian | [diuji] opencv-contrib + headless, easyocr, torch, torchvision terpasang. Dua scikit-learn TIDAK terkonfirmasi (hanya 1.5.2) |
| T-28 Paddle crash PNG nyata | belum | Fase 6 |
| T-29 test bergantung path luar | terkonfirmasi | [diuji] `pytest`: 150 passed, 1 failed (`test_document_ingestion_jpg`). `test_noisy_scan_signature_low_confidence` LULUS pada run ini (flaky, tidak deterministik). Fixture eksternal ada di disk |
| T-30 path absolut | terkonfirmasi [kode] | |
| T-31 kontradiksi dokumen | terkonfirmasi | diperbaiki di Fase 8 |

## Verifikasi tambahan

- Pemetaan GT `invoice_variasi_N.pdf` → `invoice_7_variasi_N.pdf` [diuji]: teks halaman 1 memuat `0409015` (variasi 1) dan `0509027` (variasi 2). Dengan pemetaan: 28 dokumen GT cocok, 77 line item GT (dokumen lama: 26 dokumen, 74).
- 12 PDF invoice (template 5,6,8,9,10,11) tidak punya GT [diuji].
- `DATASET\{invoice,PoD,rate card,surat jalan}` vs `DATASET\Dataset\*` [diuji, SHA-256]: invoice, rate card, surat jalan identik; PoD punya 2 perbedaan nama/isi. `Dataset\*` ditetapkan kanonik.
- `PoD\pdf\data_PoD_testing`: berisi `bersih(variasi + transformasi)` (PDF, ~3 KB, kemungkinan duplikat variasi) dan `kotor`; tidak dipakai.
- `GOLD_REVIEW` menyebut 52 gambar: terkonfirmasi (20 + 11 + 7 + 14).
- Hasil Fase 0 vs dokumen lama: bbox identik (96,7%); recall amount (28 dokumen, 55 amount): token_grid 52,7% / pymupdf 72,7% / router 89,1% (lama 50,0/71,2/88,5 pada 52); E2E flag aktif 28/28 bertabel, 50 dari 77 baris, 6/28 jumlah persis (lama 26 dokumen: 50/74, 6/26). Selisih berasal dari 2 dokumen tambahan hasil pemetaan template 7.
