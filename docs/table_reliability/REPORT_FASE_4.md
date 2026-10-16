## Fase 4: ringkasan
Stempel: GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact` belum direview manusia. Angka nyata memakai gold v2 (D-033), split dev+val. Holdout tidak dipakai di Fase 4 (tetap 2 run).
- Dikerjakan (commit `feat(tables): type-aware table discovery in pipeline, continuation and stitching (K1, F11)` dan commit penyempurnaan Fase 4):
  1. Penemuan tabel sadar-tipe dipindah ke `Phase5DocumentPipeline` setelah klasifikasi halaman, dengan halaman PDF hidup (K1). `OCREngine` hanya menemukan tabel tanpa tipe untuk pemanggil mandiri (`discover_tables`).
  2. Lanjutan tabel tanpa header antar-halaman (`continue_table`) dan `stitch_tables` ditulis ulang (tidak mengubah input, urutan natural, tidak menyambung setelah baris ringkasan). Evidence baru: `table_id`, `continued_from`, `summary_rows`, `role`, `table_groups` (semua aditif).
  3. Extractor invoice dan rate agreement memfilter tabel berdasarkan `role=line_items` (tabel lama tanpa `role` tetap diterima).
  4. Pengaman lanjutan (D-036): hanya tabel data (>= 3 baris, >= 3 kolom), tanpa baris ringkasan, dan tabel terbawah halaman yang dilanjutkan. Ditemukan dari selisih API vs router pada dokumen rate card (tabel key-value 1 baris "dilanjutkan").
  5. Default `ENABLE_TABLE_EXTRACTOR` menjadi `true` (D-035).
  6. Candidate generation + LightGBM dipasang di jalur ekstraksi (`extraction/ml/service.py`, D-037): `/v1/extract/pipeline/full` menambah `ml_extraction` (per segmen dan dokumen). Mode bayangan: extractor aturan tetap utama.
- Bukti:
  - [diuji] `pytest`: 305 lulus (penuh, setelah semua perubahan Fase 4). Test baru: `test_table_pipeline_integration.py` (12), `test_ml_extraction_service.py` (6).
  - [diuji] Data nyata dev+val, jalur API penuh (`api_flag_on`): deteksi P/R/F1 0.523/0.829/0.641; header exact 41/56; kolom 53/58; baris exact 49/58; cell F1 0.666; amount@table 141/181; ringkasan bocor 6/70. Flag mati: 0 tabel prediksi (recall 0).
  - [diuji] Pembanding flag-on lama (router bertipe, gold v2): F1 0.456; header 27/34; kolom 33/36; baris 23/36; amount@table 93/181.
  - [diuji] Router mandiri tanpa tipe (kode sekarang): F1 0.656. Selisih ke API 0.015 berasal dari 4 dokumen: invoice_18 (2: kolom 9 vs 7, versi API lebih benar), surat_jalan_5_variasi_2 (tabel API 2 baris vs 5 baris router, kemungkinan klasifikasi halaman) dan rate_card_13_variasi_2 (satu lanjutan sah yang tidak ada di gold).
  - [diuji] E2E invoice (28 GT): flag mati 0 baris; flag hidup 56/77 baris, jumlah baris tepat 14/28 (sebelum: 50/77 dan 6/28). File: `BASELINE/e2e_on_after_p4.txt`.
  - [diuji] Bundle LightGBM POD memberi skor pada kandidat dokumen nyata (`pod-10-v1`: `pihak_pertama_nama` status review, skor 0.486, 1.2 detik).
- Gate K2: lolos menurut aturan yang tertulis di D-031, dengan catatan. Rasio exact header (0.79 -> 0.73) dan kolom (0.92 -> 0.91) sedikit turun karena himpunan tabel cocok membesar (34 -> 56); angka absolut naik. Tidak ada perbandingan berpasangan per tabel. Ini keputusan penilaian (D-035), mudah dibatalkan dengan `ENABLE_TABLE_EXTRACTOR=false`.
- Target K4: belum tercapai (F1 0.64 vs 0.90; header 0.73 vs 0.90; baris 0.84 vs 0.85; cell F1 0.67 vs 0.85; amount 0.78 vs 0.95).
- Temuan baru / koreksi:
  - Jalur produksi sebelumnya TIDAK memanggil candidate generation maupun LightGBM; hanya extractor aturan. Sekarang dipanggil (D-037), tetapi belum ada model yang berguna (OS-012).
  - Nama field model (`pihak_pertama_nama`) berbeda dari nama field extractor (`receiver_name`); promosi nilai ML belum pernah terpicu.
  - Test lama `test_page_kind` memakai fake tanpa argumen baru; diperbarui menerima `**options`.
- Tidak terverifikasi / risiko:
  - Gold draf belum direview manusia; holdout 13 tabel tidak diukur ulang di fase ini.
  - Halaman lanjutan UNKNOWN tetap tidak ikut `line_items` (OS-011).
  - Model LightGBM: tidak ada metrik kualitas yang layak; hanya diuji mekanik inferensinya.
  - Latensi ML ~1.2 detik per dokumen POD (satu sampel); belum diukur pada dokumen multi-halaman.
- Keputusan yang dibutuhkan dari pemilik: label manusia untuk melatih model selain POD (OS-012); persetujuan tetap memakai flag default `true`.
- Langkah berikutnya: Fase 5 (validasi aritmetika deterministik dari registry YAML dan dipanggil dari pipeline, tolerans K3, sumber confidence).
