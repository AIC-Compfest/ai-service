# METRICS_AFTER_ALGO (Fase 3)

Stempel: **GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact`; bukan gold manusia.**
Gold nyata di sini adalah **gold v2** (D-033). Seluruh set prediksi lama (baseline, P0) dievaluasi ulang terhadap gold v2 supaya sebelum dan sesudah sebanding. Split dev + val, view `line_items`, IoU >= 0.5. Run holdout: 2 (baseline Fase 1 dan akhir Fase 3).

## 1. Data nyata (70 tabel gold, dev+val)

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:pymupdf | 70/92 | 0.391 / 0.514 / 0.444 | 27/34 (0.79) | 33/36 (0.92) | 23/36 (0.64) | 0.273 | 0.356 | 93/181 (0.51) | 151/181 (0.83) | 14/70 | 0/0 | 7 3 19 0 0 24 7 2 10 0 0 |
| base:router_none | 70/207 | 0.150 / 0.443 / 0.224 | 26/29 (0.90) | 28/31 (0.90) | 23/31 (0.74) | 0.258 | 0.449 | 85/181 (0.47) | 90/181 (0.50) | 2/70 | 0/0 | 3 3 14 0 4 36 0 29 2 0 0 |
| base:router_typed | 70/88 | 0.409 / 0.514 / 0.456 | 27/34 (0.79) | 33/36 (0.92) | 23/36 (0.64) | 0.248 | 0.332 | 93/181 (0.51) | 167/181 (0.92) | 12/70 | 0/0 | 7 3 19 0 0 25 5 4 8 0 0 |
| p0:router_typed | 70/81 | 0.444 / 0.514 / 0.477 | 27/34 (0.79) | 33/36 (0.92) | 23/36 (0.64) | 0.319 | 0.409 | 93/181 (0.51) | 159/181 (0.88) | 10/70 | 0/0 | 7 3 19 0 0 23 0 11 6 0 0 |
| p3:router_none | 70/110 | 0.536 / 0.843 / 0.656 | 42/57 (0.74) | 52/59 (0.88) | 50/59 (0.85) | 0.581 | 0.664 | 141/181 (0.78) | 148/181 (0.82) | 6/70 | 0/0 | 15 7 21 0 0 31 0 5 6 0 0 |
| p3:router_typed | 70/110 | 0.536 / 0.843 / 0.656 | 42/57 (0.74) | 52/59 (0.88) | 50/59 (0.85) | 0.581 | 0.664 | 141/181 (0.78) | 148/181 (0.82) | 6/70 | 0/0 | 15 7 21 0 0 31 0 5 6 0 0 |

Per jenis dokumen (router tanpa tipe):

**type=INVOICE**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 22/84 | 0.119 / 0.455 / 0.189 | 5/8 (0.62) | 7/10 (0.70) | 6/10 (0.60) | 0.222 | 0.463 | 75/116 (0.65) | 75/116 (0.65) | 2/62 | 0/0 | 3 3 6 0 4 14 0 9 2 0 0 |
| p3:router_none | 22/38 | 0.474 / 0.818 / 0.600 | 9/16 (0.56) | 13/18 (0.72) | 15/18 (0.83) | 0.411 | 0.538 | 90/116 (0.78) | 90/116 (0.78) | 2/62 | 0/0 | 7 5 7 0 0 13 0 2 2 0 0 |

**type=POD**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 8/22 | 0.182 / 0.500 / 0.267 | 4/4 (1.00) | 4/4 (1.00) | 3/4 (0.75) | 0.373 | 0.526 | 0/4 (0.00) | 0/4 (0.00) | 0/0 | 0/0 | 0 0 1 0 0 4 0 4 0 0 0 |
| p3:router_none | 8/6 | 1.000 / 0.750 / 0.857 | 4/6 (0.67) | 4/6 (0.67) | 5/6 (0.83) | 0.564 | 0.581 | 0/4 (0.00) | 0/4 (0.00) | 0/0 | 0/0 | 2 2 3 0 0 0 0 2 0 0 0 |

**type=RATE_AGREEMENT**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 16/57 | 0.105 / 0.375 / 0.164 | 6/6 (1.00) | 6/6 (1.00) | 6/6 (1.00) | 0.168 | 0.358 | 10/21 (0.48) | 10/21 (0.48) | 0/6 | 0/0 | 0 0 3 0 0 10 0 9 0 0 0 |
| p3:router_none | 16/32 | 0.438 / 0.875 / 0.583 | 11/14 (0.79) | 14/14 (1.00) | 10/14 (0.71) | 0.506 | 0.576 | 19/21 (0.90) | 19/21 (0.90) | 4/6 | 0/0 | 3 0 7 0 0 8 0 1 4 0 0 |

**type=SURAT_JALAN**

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 24/44 | 0.250 / 0.458 / 0.324 | 11/11 (1.00) | 11/11 (1.00) | 8/11 (0.73) | 0.361 | 0.492 | 0/40 (0.00) | 5/40 (0.12) | 0/2 | 0/0 | 0 0 4 0 0 8 0 7 0 0 0 |
| p3:router_none | 24/34 | 0.618 / 0.875 / 0.724 | 18/21 (0.86) | 21/21 (1.00) | 20/21 (0.95) | 0.755 | 0.818 | 32/40 (0.80) | 39/40 (0.97) | 0/2 | 0/0 | 3 0 4 0 0 10 0 0 0 0 0 |

## 2. Fixture konstruksi (23 tabel, 4 halaman multi-tabel)

Mode vector:

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 23/24 | 0.542 / 0.565 / 0.553 | 6/12 (0.50) | 9/13 (0.69) | 8/13 (0.62) | 0.306 | 0.358 | 114/200 (0.57) | 162/200 (0.81) | 6/7 | 1/4 (0.25) | 6 4 7 1 1 3 0 7 4 0 2 |
| p3:router_none | 23/22 | 1.000 / 0.957 / 0.978 | 21/22 (0.95) | 21/22 (0.95) | 22/22 (1.00) | 0.936 | 0.942 | 176/200 (0.88) | 176/200 (0.88) | 0/7 | 4/4 (1.00) | 1 1 1 0 0 0 0 1 0 0 1 |

Mode phrase (raster-simulasi):

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:router_none | 23/17 | 0.000 / 0.000 / 0.000 | 0/0 | 0/0 | 0/0 | 0.000 | 0.000 | 0/200 (0.00) | 43/200 (0.21) | 0/7 | 0/4 (0.00) | 0 0 0 0 1 3 0 19 0 0 2 |
| p3:router_none | 23/22 | 1.000 / 0.957 / 0.978 | 18/22 (0.82) | 18/22 (0.82) | 22/22 (1.00) | 0.862 | 0.909 | 176/200 (0.88) | 176/200 (0.88) | 0/7 | 3/4 (0.75) | 4 4 4 0 0 0 0 1 0 0 1 |

## 3. Holdout (13 tabel gold; jumlah kecil, jangan dibaca sebagai estimasi stabil)

| config | gold/pred tables | det P / R / F1 | header exact | cols exact | rows exact | row F1 | cell F1 | amount@table | amount@page | summary leak | multi-table | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run1_baseline:router_none | 13/30 | 0.167 / 0.385 / 0.233 | 5/5 (1.00) | 5/5 (1.00) | 3/5 (0.60) | 0.209 | 0.380 | 24/27 (0.89) | 24/27 (0.89) | 2/8 | 0/0 | 0 0 4 0 2 7 0 5 2 0 0 |
| run1_baseline:router_typed | 13/13 | 0.462 / 0.462 / 0.462 | 6/6 (1.00) | 6/6 (1.00) | 3/6 (0.50) | 0.215 | 0.330 | 24/27 (0.89) | 27/27 (1.00) | 2/8 | 0/0 | 0 0 5 0 0 4 2 2 2 0 0 |
| run2_phase3:router_none | 13/11 | 0.727 / 0.615 / 0.667 | 6/8 (0.75) | 6/8 (0.75) | 6/8 (0.75) | 0.415 | 0.544 | 27/27 (1.00) | 27/27 (1.00) | 2/8 | 0/0 | 2 2 4 0 0 2 0 4 2 0 0 |
| run2_phase3:router_typed | 13/12 | 0.750 / 0.692 / 0.720 | 6/8 (0.75) | 7/9 (0.78) | 7/9 (0.78) | 0.395 | 0.574 | 27/27 (1.00) | 27/27 (1.00) | 2/8 | 0/0 | 2 2 5 0 0 2 0 3 2 0 0 |

## 4. Terhadap target K4 (digital PDF, dev+val, router)

| metrik | target | sekarang | status |
|---|---|---|---|
| deteksi F1 | >= 0.90 | 0.656 | belum |
| header exact | >= 0.90 | 0.74 | belum |
| jumlah kolom exact | >= 0.90 | 0.88 | hampir |
| jumlah baris exact | >= 0.85 | 0.85 | tercapai (batas) |
| cell F1 | >= 0.85 | 0.664 | belum |
| amount di tabel cocok | >= 0.95 | 0.78 | belum |
| ringkasan tidak bocor ke item | >= 0.98 | 0.91 (6 dari 70 bocor) | belum |
| halaman multi-tabel benar penuh | >= 0.80 nyata / 100% fixture | fixture 4/4 vector, 3/4 phrase; nyata tidak terukur (gold tanpa halaman multi-tabel) | fixture tercapai untuk vector |

## 5. Sisa kegagalan (dengan bukti, bukan penyetelan)

- **Gold ikut berkontribusi.** Dari 16 gold dev+val yang tidak cocok setelah perbaikan, setidaknya 9 punya header gold yang tercemar teks sekitar (alamat, kontak, kalimat) pada gold v1; v2 memperbaiki sebagian (header_found 75 dari 83). Gold masih draf; spot-check pemilik belum diisi.
- **F6 (31 halaman dengan tabel ekstra)**: sebagian besar tabel sah non-line-item (detail pengiriman, stop, blok pihak) yang tidak ada di GT karena GT hanya memuat daftar barang. Peran `line_items` belum cukup membedakannya; ini batas klasifikasi struktural tanpa semantik.
- **Header/kolom yang masih salah** (F1 15, F2 7): sel yang digabung oleh jarak sempit (`Quantity UOM`) dan header bertingkat dengan baris vertikal berpusat; `segment_gap_em` tidak bisa memisahkan sel yang jaraknya sebanding dengan spasi kata.
- **F3 (21 halaman)**: baris pecah/tergabung pada tabel dengan sel tanpa nomor baris dan lanjutan.
- **Raster (phrase)**: tiga fixture gagal karena sel bersebelahan disatukan menjadi SATU kotak frasa OCR (blank_column, header_two_line, side_by_side); tidak terpisahkan dari geometri saja.
- **info_block_beside** (celah 10 pt) dan **multipage_no_header** (butuh penyambungan lintas halaman, Fase 4) tercatat sebagai batas.
- **Performa**: strategi `text` PyMuPDF memakan 1 sampai 5 detik per halaman padat dan tidak menambah tabel setelah arbitrase per tabel; dihapus dari konfigurasi produksi (router F1 0.586 dengan, 0.589 tanpa).
