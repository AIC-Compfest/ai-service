## Fase 0: ringkasan

Stempel: GT sintetis dan structural gold `synthetic_grounded` belum direview manusia.

- Dikerjakan: commit docs pending (`docs(tables): snapshot ...`), tag `baseline-before-audit-fix`, branch `fix/table-reliability-p0`; `.gitattributes` minimal (`* text=auto`, tanpa renormalize; `git status` tetap bersih setelahnya); skrip `scripts/baseline/*` (tiga skrip Lampiran A plus `census_tables.py`); `scripts/review_overlay.py`; `ENV_FACTS.md`; `AUDIT_VERIFICATION.md`.
- Bukti:
  - [diuji] `pytest -q`: 150 passed, 1 failed (`test_document_ingestion_jpg`), 247 s.
  - [diuji] bbox PyMuPDF: 1.574 dari 1.627 sel salah (96,7%) pada 52 tabel. Identik dengan dokumen lama.
  - [diuji] recall amount (28 dokumen, 55 amount): token_grid 29/55 = 52,7%; pymupdf 40/55 = 72,7%; router 49/55 = 89,1%.
  - [diuji] E2E: flag mati 0/28 dokumen bertabel, 0 line item dari 77 GT; flag aktif 28/28 bertabel, 50 line item dari 77, jumlah baris persis sama 6/28.
  - [diuji] sensus 159 halaman digital (invoice 40, POD 35, rate card 52, surat jalan 32): PyMuPDF(lines) menemukan ≥1 tabel di 36/40, 16/35, 28/52, 26/32 halaman; ≥2 tabel di 12, 2, 8, 6 halaman. PyMuPDF(text) selalu 1 tabel selebar halaman. Router(None) mengembalikan 126/99/127/70 tabel (rata-rata 2 sampai 4 per halaman) yang hampir semuanya potongan teks kop/syarat.
  - [diuji] overlay `out/review/phase0/{inv10v1,rc13v1,...}.png` memperlihatkan kegagalan: tabel item asli `invoice_10_Variasi_1` tidak terdeteksi sama sekali dan digantikan "tabel" palsu dari kop dan syarat; pada `rate_card_13_variasi_1`, paragraf Notes menjadi 22 baris item.
- Metrik sebelum -> sesudah: tidak ada perubahan kode produksi di fase ini.
- Temuan baru / koreksi:
  - y-band lama dihapus di `3e4326c`, bukan `fc41cca`.
  - Pemetaan template 7 terverifikasi (nomor invoice cocok) sehingga GT invoice = 28 dokumen, 77 line item.
  - Model `.joblib` dilatih dengan scikit-learn 1.7.2, venv memakai 1.5.2.
  - `parse_table_columns` dengan header kosong memetakan satu nilai ke 10 field.
  - Docker CLI terpasang tetapi daemon/WSL tidak berjalan; mengikuti pemilik, build = `blocked`.
  - Hanya satu scikit-learn terpasang (klaim sub-agent tentang dua dist-info tidak terkonfirmasi).
- Tidak terverifikasi / risiko: perilaku Paddle (Fase 6); tabel berdampingan nyata di dataset (sensus hanya menghitung per halaman, bukan posisi). Router(None) menghasilkan ratusan tabel palsu, sehingga angka "presence" yang tinggi pada flag aktif menyesatkan.
- Keputusan yang dibutuhkan dari pemilik: tidak ada.
- Langkah berikutnya: Fase 1 (pengukuran: manifest, gold, fixture konstruksi, evaluator).
