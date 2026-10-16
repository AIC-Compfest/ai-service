## Fase 6: ringkasan
Stempel: gold draf sintetis; sampel raster-twin kecil (12 tabel, hanya invoice). Angka OCR di bawah mengukur stabilitas dan kecepatan, bukan akurasi OCR.
- Dikerjakan (commit `feat(ocr): configurable PaddleOCR models, thread-safe predictor, isolated per-image runner ...`):
  1. `scripts/paddle_isolated_run.py` + `scripts/experiments/ocr_one.py`: satu proses anak per gambar dengan timeout, mencatat kode keluar, waktu, jumlah baris, dan ekor stderr. Crash atau hang native Paddle tidak bisa menjatuhkan pemanggil.
  2. Setelan baru `PADDLE_DET_MODEL`, `PADDLE_REC_MODEL`, `PADDLE_DET_LIMIT_SIDE_LEN` (default kosong = perilaku Paddle semula).
  3. Kunci `RLock` pada inisialisasi dan `predict` Paddle (predictor tidak aman antar-thread; endpoint kini berjalan di threadpool, Fase 7).
  4. Raster-twin di evaluator: `api_raster_clean` (halaman 1 dirender 200 DPI) dan `api_raster_degraded` (blur, noise, JPEG q55, rotasi 0.4 derajat, seed tetap); opsi `--limit/--max-pages/--types`; `scripts/compare_ocr_runs.py`.
- Bukti:
  - [diuji] 52 dari 52 PNG dataset (invoice 20, PoD 11, rate card 7, surat jalan 14) selesai di venv utama dengan model mobile: 0 crash, 0 timeout, 0 halaman kosong. Rata-rata 40.9 detik per gambar (maks 69.6), 54 baris per halaman (19 sampai 139), rata-rata confidence 0.977 (minimum per halaman 0.849), 75 tabel ditemukan. Ringkasan: `out/paddle_all_mobile/summary.json` (tidak di-commit; folder `out/` diabaikan git).
  - [diuji] Model server default (tiga gambar invoice): 96 sampai 121 detik per gambar (inisialisasi 22 detik, inferensi sekitar 69 detik), tanpa crash.
  - [diuji] Model mobile vs server pada tiga gambar yang sama: waktu 106.1 -> 37.6 detik; kesepakatan angka 0.99; kesepakatan baris teks persis 0.86 (0.77 / 0.93 / 0.89). Kesepakatan bukan akurasi: kedua model bisa salah bersama, dan perbedaan sebagian karena pemisahan baris.
  - [diuji] Penyebab lambat: bukan jumlah thread (1 thread 29 detik, 3 thread 26 detik, 8 thread 21 sampai 34 detik pada gambar 800x978 dengan model mobile) dan bukan ukuran gambar. oneDNN tidak bisa dipakai: `NotImplementedError ... ConvertPirAttribute2RuntimeAttribute` (Paddle 3.3.1); flag `FLAGS_enable_pir_api=0` tidak membantu.
  - [diuji] Raster-twin, 12 invoice (6 template, 12 tabel gold, dev+val), jalur API penuh:

| Mode | Deteksi P / R / F1 | Header exact | Kolom exact | Baris exact | Cell F1 | Nominal di tabel |
|---|---|---|---|---|---|---|
| PDF vektor (dokumen sama) | 0.471 / 0.667 / 0.552 | 2/8 | 8/8 | 5/8 | 0.536 | 48/74 |
| Raster bersih 200 DPI | 0.667 / 0.667 / 0.667 | 2/8 | 5/8 | 7/8 | 0.512 | 42/74 |
| Raster terdegradasi | 0.529 / 0.750 / 0.621 | 2/8 | 6/9 | 7/9 | 0.469 | 48/74 |

  - [diuji] `pytest`: 334 lulus, 1 dilewati (test Paddle sungguhan, butuh `RUN_PADDLE_TESTS=1`).
- Temuan:
  - Dugaan "Paddle crash pada PNG nyata" (T-28) tidak direproduksi pada setelan ini (mkldnn mati, venv utama). Penyebab hang yang dilaporkan di `REPORT.md` lama tidak diketahui; kemungkinan konfigurasi lama (mkldnn hidup, atau EasyOCR/torch masih terpasang).
  - Masalah operasional utama adalah kecepatan: default Paddle (model server) sekitar 70 sampai 120 detik per halaman scan di CPU 6 inti. Itu terlalu lambat untuk permintaan sinkron.
  - Raster-twin tidak runtuh pada degradasi ringan, tetapi jumlah kolom lebih sering salah di raster (5/8 dan 6/9 vs 8/8) karena frasa OCR menggabungkan sel yang berdekatan. Ini sama dengan keterbatasan fixture mode phrase di Fase 3.
- Tidak terverifikasi / risiko:
  - Sampel raster-twin hanya 12 tabel, satu jenis dokumen; selisih antar mode (satu atau dua tabel) tidak signifikan secara statistik.
  - Twin dirender dari PDF digital; bukan foto kamera sungguhan dengan perspektif/pencahayaan. Gambar PNG dataset tidak punya gold tabel, jadi akurasi tabel pada raster nyata belum terukur.
  - Hipotesis H1 (opencv contrib + headless) dan H2 (DLL torch/easyocr) tidak diuji: tidak ada kegagalan yang perlu ditriase, sehingga `.venv-exp` tidak dibuat. Paddle 3.0.x dengan oneDNN (kemungkinan 5 sampai 10 kali lebih cepat) belum dicoba karena butuh unduhan ratusan MB dan perubahan versi.
  - Isolasi proses untuk produksi (worker terpisah dengan timeout) belum dibuat; yang ada hanya kunci thread dan runner terisolasi untuk eksperimen.
  - Apakah `test_e2e_multipage` benar-benar menjalankan Paddle untuk halaman scan tidak dicek.
  - Model mobile belum menjadi default karena tidak ada ukuran akurasi OCR terhadap kebenaran.
- Keputusan yang dibutuhkan dari pemilik: apakah model mobile menjadi default (3 kali lebih cepat, kesepakatan angka 0.99, akurasi belum terukur), dan apakah Paddle 3.0.x di `.venv-exp` perlu dicoba untuk mempercepat inferensi.
- Langkah berikutnya: Fase 7 (laporan) dan Fase 8.
