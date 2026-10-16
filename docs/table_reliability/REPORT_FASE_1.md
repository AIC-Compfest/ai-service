## Fase 1: ringkasan

Stempel: GT sintetis dan structural gold `synthetic_grounded` / `constructed_exact` belum direview manusia.

- Dikerjakan:
  - Portabilitas test: `tests/support/paths.py`, skip beralasan untuk fixture eksternal, seed noise signature, ekspektasi mime kanonik PNG (commit `test: make external fixtures portable ...`).
  - Toolkit evaluasi `app/tools/table_eval/` (normalize, model, metrics F1-F11, manifest, grounding, synthetic, runner) plus `tests/unit/test_table_eval.py` (62 test dengan `test_table_gold_tooling.py`).
  - `scripts/eval_tables.py` ditulis ulang (T-22), `scripts/benchmark_tables.py` ditulis ulang (T-23), `scripts/generate_table_draft.py` diperbaiki, `scripts/build_gold.py`, `scripts/run_table_predictions.py`, `scripts/compare_reports.py`, `scripts/make_spot_check.py`.
  - Artefak: `gold/manifest.json` (205 entri), `gold/row_content/` (124), `gold/regression/` (83 tabel), `gold/constructed/` (18), `gold/bootstrap_report.json`, `gold/gt_field_inventory.json`, `config/gold/gt_field_aliases.yaml`.
  - Keputusan D-027 sampai D-030.
- Bukti:
  - [diuji] `pytest tests/unit/test_table_eval.py tests/unit/test_table_gold_tooling.py tests/unit/test_benchmark_tables_dataset.py`: lulus. Oracle prediction (gold sebagai prediksi) menghasilkan F1 = 1.0 pada ke-18 fixture, kedua view.
  - [diuji] `benchmark_tables.py --splits dev --limit 6`: 6 dokumen, tanpa error, warm-up raster hanya dilakukan bila ada gambar, tanpa `engine.reader`.
  - [diuji] GT invoice: pemetaan template 7 benar (nomor invoice cocok); 28 dokumen; split dev 60 / val 44 / holdout 20 entri.
  - Grounding: 83 tabel pada 83 dari 124 dokumen bergold, 9 tabel gagal dengan alasan tercatat di `gold/bootstrap_report.json`.
- Metrik sebelum -> sesudah: tidak ada perubahan kode produksi. Baseline ada di `METRICS_BASELINE.md`. Ringkas (dev+val, 70 tabel gold):

  | konfigurasi | detection P / R / F1 | header exact | rows exact | amount di tabel cocok |
  |---|---|---|---|---|
  | token_grid tanpa tipe (jalur API) | 0 / 0 / 0 (269 tabel palsu) | n/a | n/a | 0/181 |
  | router tanpa tipe | 0.145 / 0.429 / 0.217 | 23/30 | 23/30 | 81/181 |
  | PyMuPDF | 0.380 / 0.500 / 0.432 | 24/35 | 23/35 | 89/181 |
  | router bertipe | 0.398 / 0.500 / 0.443 | 24/35 | 23/35 | 89/181 |

  Fixture konstruksi (23 tabel, 4 halaman multi-tabel): mode vector router 0.553 F1 dan 1/4 halaman multi-tabel benar; mode raster-simulasi 0/23 terdeteksi.
- Selisih terhadap baseline kasar Fase 0: recall amount kasar (digit ada di sel mana pun) 89,1% untuk router, tetapi amount di TABEL YANG COCOK hanya 45%. Angka kasar menutupi kegagalan struktur.
- Temuan baru / koreksi:
  - Gold nyata belum punya halaman dengan 2 tabel; multi-tabel nyata (12 halaman invoice menurut sensus) hanya bisa diukur lewat fixture konstruksi dan overlay.
  - Gold `synthetic_grounded` perlu ruling PDF untuk kolom tanpa kunci GT. Itu sudah ditambahkan, tetapi kolom dan baris pada tabel tanpa ruling tetap bisa tidak lengkap (contoh: sel yang jatuh antar-baris pada `rate_card_13_variasi_2` hal. 3).
  - 36 dari 38 tabel token_grid bertipe selebar >90% halaman (T-04).
  - `evaluate_row_content` memperlihatkan router mengeluarkan 993 baris untuk 248 baris GT (precision 0.12).
- Tidak terverifikasi / risiko:
  - Kualitas structural gold belum dinilai manusia (`out/review/gold_check/CHECKLIST.md` belum diisi). Target K4 tetap provisional.
  - Gold grounded dibangun dengan heuristik sendiri (urutan kunci GT sebagai urutan kolom, ruling PDF); bias ke dokumen dengan GT berkunci jelas.
  - Raster/Paddle belum diukur.
  - Holdout run 1 sudah terpakai (agregat saja).
- Keputusan yang dibutuhkan dari pemilik: tidak ada. Opsional: isi `CHECKLIST.md`.
- Langkah berikutnya: Fase 2 (perbaikan P0 plumbing: bbox PyMuPDF, strategi PyMuPDF, satuan `invoice.py`/`rate_agreement.py`, cache config, `parse_table_columns`).
