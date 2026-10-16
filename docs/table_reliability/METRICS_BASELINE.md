# METRICS_BASELINE (Fase 1)

Stempel: **GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact`; bukan gold manusia.**
Evaluator: `scripts/eval_tables.py` (view `line_items`, IoU >= 0.5, perbandingan berurutan, duplikat dihitung, normalisasi angka Indonesia/Inggris).
Prediksi: `scripts/run_table_predictions.py` pada kode di commit sebelum perbaikan (flag tabel = komponen dipanggil langsung, bukan lewat default API).
Semua pada PDF digital. Raster/Paddle belum diukur (Fase 6).

## 1. Data nyata, split dev + val (gold `synthetic_grounded`)

Dokumen bergold: 70 (dev 42, val 28), 70 tabel gold. Gold tidak punya halaman dengan 2 tabel atau lebih, sehingga kolom `multi-table` bernilai 0/0 di data nyata (sensus Fase 0: 12 halaman invoice punya >=2 tabel menurut PyMuPDF, tanpa gold). Kasus multi-tabel hanya terukur di fixture konstruksi (bagian 3).

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token_grid_none | 70/269 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/181 (0.00) | 12/181 (0.07) | 0/70 | 0/0 | 0 0 0 0 7 70 0 62 0 0 0 |
| token_grid_typed | 70/30 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/181 (0.00) | 72/181 (0.40) | 2/70 | 0/0 | 0 0 0 0 0 22 0 42 2 0 0 |
| pymupdf | 70/92 | 0.380 / 0.500 / 0.432 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.268 | 0.350 | 89/181 (0.49) | 151/181 (0.83) | 14/70 | 0/0 | 11 3 18 0 0 21 7 2 10 0 0 |
| router_none | 70/207 | 0.145 / 0.429 / 0.217 | 23/30 (0.77) | 27/30 (0.90) | 23/30 (0.77) | 0.253 | 0.441 | 81/181 (0.45) | 90/181 (0.50) | 2/70 | 0/0 | 7 3 13 0 5 35 0 28 2 0 0 |
| router_typed | 70/88 | 0.398 / 0.500 / 0.443 | 24/35 (0.69) | 32/35 (0.91) | 23/35 (0.66) | 0.244 | 0.327 | 89/181 (0.49) | 167/181 (0.92) | 12/70 | 0/0 | 11 3 18 0 0 23 5 4 8 0 0 |

Per jenis dokumen:

**type=INVOICE**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pymupdf | 22/30 | 0.333 / 0.455 / 0.385 | 3/10 (0.30) | 7/10 (0.70) | 6/10 (0.60) | 0.235 | 0.342 | 77/116 (0.66) | 100/116 (0.86) | 12/62 | 0/0 | 7 3 6 0 0 11 3 0 8 0 0 |
| router_none | 22/84 | 0.107 / 0.409 / 0.170 | 2/9 (0.22) | 6/9 (0.67) | 6/9 (0.67) | 0.205 | 0.437 | 71/116 (0.61) | 75/116 (0.65) | 2/62 | 0/0 | 7 3 5 0 4 14 0 10 2 0 0 |
| router_typed | 22/26 | 0.385 / 0.455 / 0.417 | 3/10 (0.30) | 7/10 (0.70) | 6/10 (0.60) | 0.175 | 0.284 | 77/116 (0.66) | 116/116 (1.00) | 10/62 | 0/0 | 7 3 6 0 0 12 1 2 6 0 0 |

**type=POD**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pymupdf | 8/8 | 0.750 / 0.750 / 0.750 | 4/6 (0.67) | 6/6 (1.00) | 3/6 (0.50) | 0.294 | 0.256 | 0/4 (0.00) | 0/4 (0.00) | 0/0 | 0/0 | 2 0 3 0 0 2 0 0 0 0 0 |
| router_none | 8/22 | 0.182 / 0.500 / 0.267 | 4/4 (1.00) | 4/4 (1.00) | 3/4 (0.75) | 0.373 | 0.526 | 0/4 (0.00) | 0/4 (0.00) | 0/0 | 0/0 | 0 0 1 0 0 4 0 4 0 0 0 |
| router_typed | 8/8 | 0.750 / 0.750 / 0.750 | 4/6 (0.67) | 6/6 (1.00) | 3/6 (0.50) | 0.294 | 0.256 | 0/4 (0.00) | 0/4 (0.00) | 0/0 | 0/0 | 2 0 3 0 0 2 0 0 0 0 0 |

**type=RATE_AGREEMENT**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pymupdf | 16/28 | 0.286 / 0.500 / 0.364 | 6/8 (0.75) | 8/8 (1.00) | 6/8 (0.75) | 0.207 | 0.317 | 12/21 (0.57) | 18/21 (0.86) | 2/6 | 0/0 | 2 0 5 0 0 6 4 2 2 0 0 |
| router_none | 16/57 | 0.105 / 0.375 / 0.164 | 6/6 (1.00) | 6/6 (1.00) | 6/6 (1.00) | 0.168 | 0.358 | 10/21 (0.48) | 10/21 (0.48) | 0/6 | 0/0 | 0 0 3 0 0 10 0 7 0 0 0 |
| router_typed | 16/28 | 0.286 / 0.500 / 0.364 | 6/8 (0.75) | 8/8 (1.00) | 6/8 (0.75) | 0.207 | 0.317 | 12/21 (0.57) | 18/21 (0.86) | 2/6 | 0/0 | 2 0 5 0 0 6 4 2 2 0 0 |

**type=SURAT_JALAN**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pymupdf | 24/26 | 0.423 / 0.458 / 0.440 | 11/11 (1.00) | 11/11 (1.00) | 8/11 (0.73) | 0.350 | 0.424 | 0/40 (0.00) | 33/40 (0.82) | 0/2 | 0/0 | 0 0 4 0 0 2 0 0 0 0 0 |
| router_none | 24/44 | 0.250 / 0.458 / 0.324 | 11/11 (1.00) | 11/11 (1.00) | 8/11 (0.73) | 0.361 | 0.492 | 0/40 (0.00) | 5/40 (0.12) | 0/2 | 0/0 | 0 0 4 0 1 7 0 7 0 0 0 |
| router_typed | 24/26 | 0.423 / 0.458 / 0.440 | 11/11 (1.00) | 11/11 (1.00) | 8/11 (0.73) | 0.338 | 0.412 | 0/40 (0.00) | 33/40 (0.82) | 0/2 | 0/0 | 0 0 4 0 0 3 0 0 0 0 0 |

Bacaan: `token_grid` tanpa tipe dokumen (jalur API saat flag aktif) tidak menemukan SATU pun tabel gold dari 70 (0 TP) dengan 269 tabel prediksi. `token_grid` bertipe pun 0 TP karena bbox tabel melebar (36 dari 38 tabel token_grid bertipe selebar >90% halaman [diuji]; IoU < 0.5), walau amount ada di halaman (40%). Router tanpa tipe: precision 0.145 (207 tabel untuk 70 gold). PyMuPDF paling dekat, tetapi baris benar hanya 23 dari 35 tabel yang cocok dan 14 dari 70 ringkasan bocor ke item.

## 2. Content-only (baris GT sebagai nilai skalar, tanpa struktur), dev + val

| config | GT rows | pred rows | covered | row precision | row recall | docs row-count exact |
|---|---|---|---|---|---|---|
| token_grid_none | 248 | 919 | 10 | 0.010881 | 0.040323 | 5/70 |
| token_grid_typed | 248 | 387 | 53 | 0.136951 | 0.21371 | 0/70 |
| pymupdf | 248 | 1036 | 164 | 0.158301 | 0.66129 | 23/70 |
| router_none | 248 | 993 | 118 | 0.118832 | 0.475806 | 24/70 |
| router_typed | 248 | 1123 | 165 | 0.146928 | 0.665323 | 23/70 |

## 3. Fixture konstruksi (gold eksak, 18 dokumen / 20 halaman / 23 tabel, 4 halaman multi-tabel)

Mode `vector` (kata PDF eksak + halaman PDF hidup):

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token_grid_none | 23/18 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/200 (0.00) | 43/200 (0.21) | 0/7 | 0/4 (0.00) | 0 0 0 0 2 3 0 19 0 0 2 |
| pymupdf | 23/23 | 0.783 / 0.783 / 0.783 | 6/17 (0.35) | 11/18 (0.61) | 10/18 (0.56) | 0.322 | 0.391 | 174/200 (0.87) | 200/200 (1.00) | 7/7 | 1/4 (0.25) | 11 7 12 2 0 3 0 0 5 0 1 |
| router_none | 23/24 | 0.542 / 0.565 / 0.553 | 6/12 (0.50) | 9/13 (0.69) | 8/13 (0.62) | 0.306 | 0.358 | 114/200 (0.57) | 162/200 (0.81) | 6/7 | 1/4 (0.25) | 6 4 7 1 1 3 0 7 4 0 2 |

Mode `phrase` (box frasa, token estimasi, jitter, skew 0.4 derajat; tanpa halaman PDF hidup):

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token_grid_none | 23/17 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/200 (0.00) | 43/200 (0.21) | 0/7 | 0/4 (0.00) | 0 0 0 0 1 3 0 19 0 0 2 |
| pymupdf | 23/0 | n/a / 0.000 / n/a | 0/0 | 0/0 | 0/0 | n/a | n/a | 0/200 (0.00) | 0/200 (0.00) | 0/7 | 0/4 (0.00) | 0 0 0 0 0 0 0 19 0 0 2 |
| router_none | 23/17 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/200 (0.00) | 43/200 (0.21) | 0/7 | 0/4 (0.00) | 0 0 0 0 1 3 0 19 0 0 2 |

Bacaan: pada fixture bersih saja router hanya menjawab 1 dari 4 halaman multi-tabel. Pada mode raster-simulasi, tidak ada tabel yang cocok (0/23).

## 4. Holdout, run #1 (agregat saja; tidak dipakai untuk desain)

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pymupdf | 13/13 | 0.462 / 0.462 / 0.462 | 6/6 (1.00) | 6/6 (1.00) | 3/6 (0.50) | 0.177 | 0.246 | 24/27 (0.89) | 27/27 (1.00) | 2/8 | 0/0 | 0 0 5 0 0 3 2 2 2 0 0 |
| router_none | 13/30 | 0.167 / 0.385 / 0.233 | 5/5 (1.00) | 5/5 (1.00) | 3/5 (0.60) | 0.209 | 0.380 | 24/27 (0.89) | 24/27 (0.89) | 2/8 | 0/0 | 0 0 4 0 2 7 0 5 2 0 0 |
| router_typed | 13/13 | 0.462 / 0.462 / 0.462 | 6/6 (1.00) | 6/6 (1.00) | 3/6 (0.50) | 0.215 | 0.330 | 24/27 (0.89) | 27/27 (1.00) | 2/8 | 0/0 | 0 0 5 0 0 3 2 4 2 0 0 |

Jumlah run holdout sejauh ini: 1 (baseline, Fase 1).

## 5. Perbandingan dengan baseline kasar Fase 0

Recall amount kasar (dokumen lama/skrip lama: token_grid 52.7, pymupdf 72.7, router 89.1 persen) mengukur "angka ada di sel mana pun" dan menutupi kegagalan struktur. Pada evaluator baru, amount ditemukan DI TABEL YANG COCOK hanya 45 sampai 49 persen (router/pymupdf) dan amount ada di halaman 50 sampai 92 persen. Presence tinggi, struktur rendah.

## 6. Keterbatasan
- Structural gold dibangun dari nilai GT dan kata PDF (`app/tools/table_eval/grounding.py`). 83 tabel pada 83 dari 124 dokumen bergold berhasil di-grounding (9 tabel gagal: 5 POD baris tak terjangkar, 2 invoice dan 1 surat jalan cakupan lemah, 1 POD satu kolom; sisa dokumen bergold, 41 secara total tanpa gold tabel, tidak punya list-of-dict tabel di GT). Hasilnya draf; spot-check manusia ada di `out/review/gold_check/CHECKLIST.md` (belum diisi).
- Sebagian kolom gold bergantung pada urutan kunci GT dan ruling PDF; tabel tanpa ruling dengan kolom tanpa kunci GT bisa kekurangan kolom.
- Raster belum tercakup.
