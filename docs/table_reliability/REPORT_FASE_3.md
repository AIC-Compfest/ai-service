## Fase 3: ringkasan

Stempel: GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact` belum direview manusia. Semua angka nyata memakai gold v2 (D-033).

- Dikerjakan (commit `feat(tables): geometry-first table discovery ...`, `perf(tables): drop PyMuPDF text strategy ...`, `docs(tables): phase 3 ...`):
  1. Geometri teks (`geometry.py`): garis visual dari pusat y, segmen sel dari jarak relatif terhadap em, blok dari pitch lokal, pemisahan tabel berdampingan hanya bila ada header berulang (F4/F5).
  2. Header berskor (`header.py`): header pertama yang meyakinkan membuka tabel; baris label:value, prosa, dan baris penuh digit ditolak; header dua baris dan label grup digabung (F1).
  3. Kolom dari header DAN data (`columns.py`): batas di ruang kosong antara ekstensi data, garis vertikal PDF bila melingkupi header, kolom tanpa header untuk sel yatim, sel yatim yang menempel digabung, pemisahan frasa OCR terestimasi di batas kolom (F2, raster).
  4. Baris dan akhir tabel (`rows.py`): label ringkasan dari YAML diuji SEBELUM aturan lanjutan, catatan/footer mengakhiri tabel, baris item berawalan "Total" tetap item (F3, F9, F10).
  5. Peran dan arbitrase (`scoring.py`, `router.py`): `line_items/summary/info/other` dengan alasan, kualitas per tabel, satu tabel terbaik per kelompok tumpang tindih lintas provider (F6, F7).
  6. Strategi `text` PyMuPDF dihapus dari konfigurasi produksi (performa, D-034). Alias (langkah 8) tidak diubah: tidak terbukti menjadi penyebab pada dev.
- Bukti:
  - [diuji] `pytest`: 287 lulus sebelum perubahan konfigurasi terakhir; `test_table_structure.py` (20), `test_table_geometry.py` (16), router/kandidat/pymupdf lulus setelahnya.
  - [diuji] Fixture konstruksi mode vector: deteksi F1 0.553 -> 0.978, halaman multi-tabel benar penuh 1/4 -> 4/4, ringkasan bocor 6/7 -> 0/7, baris exact 8/13 -> 22/22.
  - [diuji] Data nyata dev+val (router tanpa tipe, gold v2): deteksi P/R/F1 0.150/0.443/0.224 -> 0.536/0.843/0.656; header exact 26/29 -> 42/57; baris exact 23/31 -> 50/59; cell F1 0.449 -> 0.664; amount di tabel cocok 85/181 -> 141/181.
  - [diuji] Holdout (run 2 dari 2, 13 tabel gold): router tanpa tipe F1 0.233 -> 0.667; router bertipe 0.462 -> 0.720.
- Metrik sebelum -> sesudah: lihat `METRICS_AFTER_ALGO.md` (per jenis dokumen dan tabel F1–F11). Surat jalan paling baik (F1 0.724, cell F1 0.818), invoice paling lemah (F1 0.600).
- Target K4: belum tercapai pada deteksi F1 (0.656 vs 0.90), header exact (0.74 vs 0.90), cell F1 (0.664 vs 0.85), amount (0.78 vs 0.95) dan kebocoran ringkasan (0.91 vs 0.98); tercapai pada jumlah baris exact (0.85) dan multi-tabel fixture vector (4/4). Alasan sisa kegagalan ada di METRICS_AFTER_ALGO.md bagian 5. Tidak ada ambang yang disetel supaya target lolos.
- Temuan baru / koreksi:
  - Perintah evaluasi router bertipe sempat macet 25 menit; penyebabnya biaya `find_tables(strategy="text")` (1 sampai 5 detik per halaman padat), bukan loop. Diselesaikan dengan D-034. Proses python yang menggantung dihentikan.
  - Gold v1 mencemari header dengan teks sekitar; diperbaiki (D-033) dan semua prediksi lama dinilai ulang.
  - Pada tabel digital, PyMuPDF `rows[r].cells[c]` (Fase 2) tetap dipakai bila tabel ber-garis menang arbitrase; `token_grid` menang pada tabel tanpa garis.
  - Alias header (T-19) dan `ambiguous_ok` tidak dimodifikasi; tidak terukur sebagai penyebab.
- Tidak terverifikasi / risiko:
  - Raster nyata (Paddle) belum terukur; fixture OCR-simulasi hanya aproksimasi, dan tiga dari 18 fixture gagal di mode itu.
  - Gold draf belum direview manusia; holdout hanya 13 tabel.
  - Peran `info/other` ditentukan struktur saja; tabel sah non-line-item masih lolos sebagai `line_items` (F6 = 31).
  - `info_block_beside` (celah 10 pt) tetap gagal; tidak ada tabel berdampingan nyata di gold.
- Keputusan yang dibutuhkan dari pemilik: tidak ada.
- Langkah berikutnya: Fase 4 (sambungkan jalur sadar-tipe ke API: tabel dicari setelah klasifikasi dengan halaman PDF hidup, stitching per segmen, extractor memakai `role=line_items`, keputusan flag K2).
