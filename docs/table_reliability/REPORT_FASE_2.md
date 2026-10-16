## Fase 2: ringkasan

Stempel: GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact` belum direview manusia.

- Dikerjakan (satu commit per bug, masing-masing dengan test yang gagal dulu pada kode lama):
  1. `fix(tables): derive PyMuPDF cell bbox/columns/header from table.rows grid (T-11, F2)`: bbox dari `table.rows[r].cells[c]`, sel `None` dan sel gabungan tidak menggeser indeks, header `external` tidak membuang baris data, header internal diambil dari baris pertama `extract()` (bukan `header.names` yang diganti nama oleh PyMuPDF), kolom dari geometri sel nyata.
  2. `fix(tables): reject page-wide PyMuPDF tables and make strategy options configurable (T-10, F7)`: opsi di `table_resolution.yaml` (`max_page_area_ratio: 0.5`, asal: tabel gold terbesar 0,298 pada dev dan 0,249 pada val, p90 0,22), strategi `lines` lalu `text` dengan penerimaan, bukan berhenti di hasil non-kosong pertama.
  3. `fix(extraction): use 0..1 layout zones from settings in invoice/rate_agreement (T-13)`: lima konstanta piksel diganti pengaturan pecahan halaman; `to_unit_bbox` menerima fixture lama 0..1000; regex vendor `PT.`.
  4. `fix(tables): resolve config from settings, cache extractor and matchers (T-24, T-25)`: `settings.config_dir` relatif lokasi paket (override `CONFIG_DIR`), `TableExtractor.cached`, matcher per tipe dokumen di-cache.
  5. `fix(extraction): token-based header matching, empty headers never match (T-15)`.
  6. D-031 mengoreksi D-005 (status flag nyata); D-005 diberi tanda koreksi.
- Bukti:
  - [diuji] bbox: 1.574 dari 1.627 sel salah (96,7%) -> 0 dari 1.625 (`check_pymupdf_bbox.py`, `test_pymupdf_cell_geometry.py`).
  - [diuji] test gagal pada kode lama lalu lulus: geometri 6 gagal; layout unit 5 gagal; config/cache 3 gagal, termasuk `FileNotFoundError: config\candidates\table_resolution.yaml` dari CWD lain; parse_table_columns 7 gagal. Perintah: `git stash push` file produksi, `pytest`, `git stash pop`.
  - [diuji] `parse_table_columns` dengan header kosong memetakan satu nilai ke 10 field; sekarang tidak ada field terisi selain `line_total`.
- Metrik sebelum -> sesudah: lihat `METRICS_AFTER_P0.md`. Ringkas (dev+val, 70 tabel gold):

  | konfigurasi | det F1 | cell F1 | summary leak | F7 | amount@table |
  |---|---|---|---|---|---|
  | pymupdf | 0.432 -> 0.458 | 0.350 -> 0.452 | 14 -> 10 | 7 -> 0 | 89/181 -> 89/181 |
  | router tanpa tipe | 0.217 -> 0.235 | 0.441 -> 0.444 | 2 -> 2 | 0 -> 0 | 81/181 -> 81/181 |
  | router bertipe | 0.443 -> 0.464 | 0.327 -> 0.403 | 12 -> 10 | 5 -> 0 | 89/181 -> 89/181 |

  Recall deteksi tetap 0,500 dan jumlah baris benar 23/35: perbaikan plumbing tidak mengubah akar masalah. `amount@page` turun (151 -> 131 dari 181) karena tabel selebar halaman yang dibuang memuat semua angka halaman; penjelasan lengkap di METRICS_AFTER_P0.md. Field header invoice: vendor_name 6 -> 10 dari 28, lainnya tetap.
- Temuan baru / koreksi:
  - Strategi `text` PyMuPDF pada halaman paragraf mengembalikan "tabel" dengan luas hanya 0,21 sehingga aturan luas tidak menolaknya. Penolakan semacam itu butuh klasifikasi peran tabel (Fase 3, langkah 5).
  - `check_pymupdf_bbox.py` memasangkan baris tanpa memperhitungkan baris kosong yang kini dibuang provider; skrip diperbaiki.
  - Router masih memilih satu provider per halaman, dan E2E line item tidak berubah (50/77) sampai Fase 3 dan 4.
  - Perubahan jalur default `config/` di `app/tools/labeling.py`, `training.py`, `build_candidate_dataset.py`, `training/run.py` dan `extraction/ml/predictor.py` hanya mengganti default CWD-relatif menjadi `settings.config_dir`; tidak ada logika labeling/training yang diubah. 30 test terkait lulus.
- Tidak terverifikasi / risiko:
  - Raster belum terukur.
  - Gold `synthetic_grounded` adalah draf; selisih kecil (satu sampai tiga tabel) pada 70 tabel belum signifikan secara statistik.
  - Holdout tidak dijalankan di fase ini (jumlah run = 1).
- Keputusan yang dibutuhkan dari pemilik: tidak ada.
- Langkah berikutnya: Fase 3 (algoritma tabel: segmentasi wilayah, header berskor, kolom dari data, baris dan akhir tabel, peran tabel, arbitrase per tabel, raster).
