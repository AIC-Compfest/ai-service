# METRICS_AFTER_P0 (Fase 2)

Stempel: **GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact`; bukan gold manusia.**
Perbandingan sebelum (`METRICS_BASELINE.md`) dan sesudah lima perbaikan P0, split dev + val, view `line_items`, IoU >= 0.5. Holdout tidak dijalankan di fase ini (run holdout tetap 1).

## Data nyata (70 tabel gold pada dev+val)

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before:pymupdf | 70/92 | 0.380 / 0.500 / 0.432 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.268 | 0.350 | 89/181 (0.49) | 151/181 (0.83) | 14/70 | 0/0 | 11 3 18 0 0 21 7 2 10 0 0 |
| after:pymupdf | 70/83 | 0.422 / 0.500 / 0.458 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.366 | 0.452 | 89/181 (0.49) | 131/181 (0.72) | 10/70 | 0/0 | 11 3 18 0 0 20 0 11 6 0 0 |
| before:router_none | 70/207 | 0.145 / 0.429 / 0.217 | 23/30 (0.77) | 27/30 (0.90) | 23/30 (0.77) | 0.253 | 0.441 | 81/181 (0.45) | 90/181 (0.50) | 2/70 | 0/0 | 7 3 13 0 5 35 0 28 2 0 0 |
| after:router_none | 70/185 | 0.162 / 0.429 / 0.235 | 23/30 (0.77) | 27/30 (0.90) | 23/30 (0.77) | 0.263 | 0.444 | 81/181 (0.45) | 90/181 (0.50) | 2/70 | 0/0 | 7 3 13 0 3 35 0 28 2 0 0 |
| before:router_typed | 70/88 | 0.398 / 0.500 / 0.443 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.244 | 0.327 | 89/181 (0.49) | 167/181 (0.92) | 12/70 | 0/0 | 11 3 18 0 0 23 5 4 8 0 0 |
| after:router_typed | 70/81 | 0.432 / 0.500 / 0.464 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.314 | 0.403 | 89/181 (0.49) | 159/181 (0.88) | 10/70 | 0/0 | 11 3 18 0 0 22 0 11 6 0 0 |
| token_grid_none | 70/269 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/181 (0.00) | 12/181 (0.07) | 0/70 | 0/0 | 0 0 0 0 7 70 0 62 0 0 0 |
| token_grid_typed | 70/30 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/181 (0.00) | 72/181 (0.40) | 2/70 | 0/0 | 0 0 0 0 0 22 0 42 2 0 0 |

## Fixture konstruksi, mode vector

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before:pymupdf | 23/23 | 0.783 / 0.783 / 0.783 | 6/17 (0.35) | 11/18 (0.61) | 10/18 (0.56) | 0.322 | 0.391 | 174/200 (0.87) | 200/200 (1.00) | 7/7 | 1/4 (0.25) | 11 7 12 2 0 3 0 0 5 0 1 |
| after:pymupdf | 23/23 | 0.783 / 0.783 / 0.783 | 6/17 (0.35) | 11/18 (0.61) | 10/18 (0.56) | 0.322 | 0.391 | 174/200 (0.87) | 200/200 (1.00) | 7/7 | 1/4 (0.25) | 11 7 12 2 0 3 0 0 5 0 1 |
| before:router_none | 23/24 | 0.542 / 0.565 / 0.553 | 6/12 (0.50) | 9/13 (0.69) | 8/13 (0.62) | 0.306 | 0.358 | 114/200 (0.57) | 162/200 (0.81) | 6/7 | 1/4 (0.25) | 6 4 7 1 1 3 0 7 4 0 2 |
| after:router_none | 23/24 | 0.542 / 0.565 / 0.553 | 6/12 (0.50) | 9/13 (0.69) | 8/13 (0.62) | 0.306 | 0.358 | 114/200 (0.57) | 162/200 (0.81) | 6/7 | 1/4 (0.25) | 6 4 7 1 1 3 0 7 4 0 2 |

Mode `phrase` tidak berubah (PyMuPDF tidak aktif tanpa halaman PDF hidup; token_grid belum disentuh).

## Skrip baseline

| ukuran | sebelum | sesudah |
|---|---|---|
| bbox sel PyMuPDF salah (`check_pymupdf_bbox.py`) | 1.574 / 1.627 (96,7%) | 0 / 1.625 (0,0%) |
| E2E invoice flag aktif (`e2e_line_items.py on`): dokumen bertabel / baris terekstrak / jumlah persis | 28 / 50 dari 77 / 6 dari 28 | 28 / 50 dari 77 / 6 dari 28 (tidak berubah) |
| E2E flag mati | 0 / 0 / 0 | 0 / 0 / 0 |
| field header invoice (`invoice_fields_check.py`): invoice_number / vendor_name / subtotal / grand_total (dari 28) | 18 / 6 / 12 / 13 | 18 / 10 / 12 / 13 |

## Penjelasan perubahan metrik

- **PyMuPDF detection F1 0.432 -> 0.458, cell F1 0.350 -> 0.452, summary leak 14 -> 10**: tabel selebar halaman dari strategi `text` dibuang (F7 7 -> 0) dan bbox/kolom/header sel kini berasal dari grid PyMuPDF.
- **amount@page 151/181 -> 131/181 (turun)**: bukan kemunduran kualitas. Tabel selebar halaman yang dibuang memuat hampir semua angka halaman, sehingga "angka ada di halaman" terlalu mudah terpenuhi. Metrik yang dipakai K4 adalah `amount@table` (tabel yang cocok), yang tidak berubah (89/181).
- **F8 2 -> 11** (tabel gold tanpa kandidat): sebelumnya tabel selebar halaman tumpang tindih dengan tabel gold sehingga tidak dihitung "tidak terdeteksi", padahal tidak ada yang cocok (IoU < 0.5). Recall deteksi tetap 0.500. Ini kegagalan yang sama yang kini terlihat jelas, bukan yang baru.
- **Recall deteksi tetap 0.500 dan baris benar 23/35**: perbaikan plumbing tidak menyentuh akar masalah (header/kolom/baris/multi-tabel pada tabel tanpa garis). Itu dikerjakan di Fase 3.
- **E2E line item tidak berubah (50/77)**: jalur ekstraktor invoice mengonsumsi tabel router (token_grid tanpa tipe) yang belum diperbaiki; hasil baru terlihat setelah Fase 3 dan 4.
- **vendor_name 6 -> 10**: zona header sekarang benar-benar zona header, ditambah regex `PT.` (dengan titik).
- `measure_amount_recall.py` dan `e2e_line_items.py on` tidak bergerak karena skrip itu membuat `PyMuPDFTableProvider()` tanpa opsi YAML (recall) atau router token_grid yang dominan (E2E).

## Catatan
- Raster tidak terukur. Bagian PyMuPDF yang baru hanya memengaruhi PDF digital.
- Holdout tidak dijalankan di fase ini.
