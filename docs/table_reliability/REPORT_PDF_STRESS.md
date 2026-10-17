# REPORT_PDF_STRESS: PDF variatif dengan banyak tabel, PyMuPDF vs router

**Stempel:** halaman dibangkitkan secara deterministik (`app/tools/table_eval/stress.py`), digambar dengan PyMuPDF sehingga gold eksak (`constructed_exact`); tidak ada review manusia dan **bukan sampel dokumen produksi**. Hasilnya menunjukkan fitur tata letak mana yang membuat ekstraktor gagal, bukan seberapa sering fitur itu muncul. Aturan disetel pada seed 0 sampai 239; angka di bawah memakai seed 1000 sampai 1239 yang tidak dipakai saat menyetel.

## Pertanyaan: seberapa baik PyMuPDF pada data variatif dengan lebih dari satu tabel?
Jawaban singkat: PyMuPDF `find_tables` baik untuk header dan kolom pada tabel bergaris yang ia temukan, tetapi (a) tidak menemukan tabel tanpa garis sama sekali, (b) sebelum pembersihan, baris total ikut menjadi baris item, dan (c) pada halaman dengan banyak tabel ia hanya menemukan sebagian. Sebagai bagian dari router (PyMuPDF + geometri token) hasilnya jauh lebih baik.

Halaman vector, 240 halaman, 453 tabel gold, 1 sampai 4 tabel per halaman:

| | Deteksi P / R / F1 | Header exact | Kolom exact | Baris exact | Halaman benar sepenuhnya |
|---|---|---|---|---|---|
| PyMuPDF sendirian, sebelum | 0.98 / 0.46 / 0.63 | 169/196 (86%) | 208/208 (100%) | 84/208 (40%) | 20/240 (8%) |
| PyMuPDF sendirian, sesudah | 0.99 / 0.46 / 0.63 | 169/196 (86%) | 209/209 (100%) | 155/209 (74%) | 60/240 (25%) |
| Router, sebelum | 0.97 / 0.90 / 0.93 | 348/396 (88%) | 402/408 (99%) | 362/408 (89%) | 126/240 (52%) |
| **Router, sesudah** | 0.96 / 0.92 / 0.94 | **391/395 (99%)** | 415/417 (100%) | **409/417 (98%)** | **209/240 (87%)** |

"Sebelum" adalah kode di commit `411c84d`; "sesudah" adalah kode sekarang, pada halaman yang sama. Header/kolom/baris exact dihitung pada tabel yang terdeteksi; recall deteksi menunjukkan berapa banyak tabel yang terdeteksi.

Menurut fitur (router sesudah | PyMuPDF sendirian sesudah), halaman benar sepenuhnya:

| Fitur | Router | PyMuPDF | Catatan |
|---|---|---|---|
| Tabel bergaris | 88% | 54% | PyMuPDF deteksi 0.99, baris 73% |
| Tabel tanpa garis | 88% | 0% | PyMuPDF recall 0.00; router recall 0.88 |
| 1 tabel per halaman | 97% | 33% | |
| 2 tabel per halaman | 74% | 27% | recall router 0.82 (tabel berdampingan) |
| 3 atau lebih tabel | 93% | 10% | PyMuPDF recall 0.40 |
| Header bertingkat | 79% | 0% | header exact router 97% (sebelum 49%) |
| Tabel tanpa header | 66% | 14% | recall router 0.85 (sebelum 0.68) |
| Header membungkus | 84% | 17% | |
| Baris membungkus | 85% | 25% | |
| Baris ringkasan di dalam tabel | 87% | 23% | |
| Blok ringkasan terpisah | 92% | 38% | |
| **Tabel berdampingan** | **37%** | 23% | recall 0.47; kelemahan terbesar (OS-015) |

Kode kegagalan router (halaman): F1 header 4, F2 kolom 2, F3 baris 9, F4 satu tabel menutup dua tabel 14, F5 satu tabel terbelah 1, F8 tabel terlewat 5, F9 ringkasan bocor 4, F10 item dianggap ringkasan 2.

Simulasi OCR raster (mode `phrase`: sel bersebelahan digabung dalam satu kotak frasa, token diperkirakan, jitter dan miring 0.4 derajat; bukan Paddle sungguhan): halaman benar 16% -> 26%, header exact 45% -> 55%, kolom 77% -> 81%. Jalur raster jauh lebih lemah dari PDF digital.

## Data nyata (dev+val, 70 tabel gold; gold draf sintetis)
| | Sebelum (`411c84d`) | Sesudah |
|---|---|---|
| Router tanpa tipe, F1 deteksi | 0.685 | 0.697 |
| Jalur API, deteksi P / R / F1 | 0.550 / 0.857 / 0.670 (setelah D-042) | 0.569 / 0.886 / 0.693 |
| Jalur API, baris exact | 51/60 | 55/62 |
| Surat jalan, F1 deteksi | 0.690 | 0.759 |
| Nominal di dalam tabel | 155/181 | 152/181 (turun 3, semua dari invoice_14, OS-017) |
| E2E invoice (28 PDF, 77 baris GT) | 63 baris, 16 dokumen tepat | 61 baris, 18 dokumen tepat |
| Holdout, 13 tabel (run 5) | F1 0.720 (run 4) | 0.692 (satu tabel prediksi tambahan; tidak disetel dari hasil ini) |

`pytest`: 359 lulus, 1 dilewati. Fixture konstruksi (23 tabel): tidak berubah, 0.978 pada mode vector dan phrase; `multipage_no_header` yang dulu batas diketahui kini lulus pada mode vector.

## Yang dikerjakan (keputusan D-044 sampai D-048)
1. Label grup header menempel ke semua kolom yang dicakup (D-044).
2. Pembersihan tabel PyMuPDF: baris ringkasan dipisah, header-berisi-data menjadi baris, kotak tabel dipotong (D-045).
3. Baris ringkasan berlabel asing dan perhitungan kolom tanpa baris total (D-046).
4. Penemuan tabel tanpa header (D-047).
5. Dibatalkan karena merugikan: aturan "header di kedua sisi celah" untuk tabel berdampingan; penalti sel tergabung (D-048).

## Yang tidak terverifikasi
- Seberapa mirip halaman acak dengan dokumen produksi tidak diketahui; gold nyata tidak punya halaman multi-tabel atau tabel berdampingan.
- Hasil raster nyata (PaddleOCR) untuk perbaikan ini tidak diukur; hanya simulasi.
- Aturan struktural (baris ringkasan berlabel asing, tabel tanpa header) bisa keliru pada dokumen dengan tata letak yang tidak ada di generator; dijaga oleh 80 halaman uji (`tests/unit/test_table_stress.py`) dan dev+val nyata, bukan oleh dokumen produksi.
- Satu tabel prediksi tambahan pada holdout menunjukkan risiko tabel palsu dari penemuan tanpa header.

## Reproduksi
```powershell
.\.venv\Scripts\python.exe scripts\stress_tables.py --pages 240 --first-seed 1000 --mode vector --configs pymupdf,token_grid_none,router_none --out out\stress
.\.venv\Scripts\python.exe scripts\stress_tables.py --pages 240 --first-seed 1000 --mode phrase --configs token_grid_none,router_none --out out\stress_phrase
```
