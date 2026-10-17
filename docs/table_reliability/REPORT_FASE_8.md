## Fase 8: ringkasan
Stempel: gold draf sintetis (`synthetic_grounded`, `constructed_exact`), belum direview manusia. Holdout: 13 tabel, dipakai 4 kali (run 1 baseline Fase 1, run 2 akhir Fase 3, run 3 awal Fase 8 sebelum D-042, run 4 setelah D-042); tidak ada keputusan yang diambil dari hasil holdout.
- Dikerjakan:
  1. Evaluasi akhir: dev+val jalur API, holdout run 3 dan 4, fixture vector dan phrase, skrip E2E (flag hidup dan mati), bbox PyMuPDF, skrip recall lama. File: `BASELINE/*_final.txt`.
  2. Menyelidiki pertentangan antara skrip recall lama (router 89.1% -> 76.4%) dan evaluator (naik). Dugaan awal saya (artefak metrik) terbukti salah: itu kemunduran nyata pada `invoice_13` (item tiga baris dengan angka di tengah). Perbaikan D-042 (`geometry.block_periodic_*`, `rows.lead_line_rows`) mengembalikan skrip lama ke 49/55 dan menaikkan metrik evaluator.
  3. Percobaan penalti sel tergabung dibatalkan karena tidak berpengaruh (D-043).
  4. Rekonsiliasi dokumen: PLAN, REPORT, dan GOLD_REVIEW diberi banner dan koreksi pada klaim yang terbukti keliru (dataset, Python 3.10, Paddle "tidak menghasilkan evidence"). README ditulis ulang (Fase 7).
  5. Koreksi nama model Paddle default (PP-OCRv6 medium, bukan "server PP-OCRv5") di config, README, `.env.example`, REPORT_FASE_6, D-040.
  6. OS-009 sampai OS-014 dicatat; `REPORT_FINAL.md` ditulis.
- Bukti:
  - [diuji] `pytest`: 337 lulus, 1 dilewati.
  - [diuji] dev+val, jalur API, sebelum -> sesudah D-042: F1 deteksi 0.641 -> 0.670; nominal di tabel 141/181 -> 155/181; kolom exact 53/58 -> 55/60; header 41/56 -> 42/56; baris exact 49/58 -> 51/60; cell F1 0.666 -> 0.673.
  - [diuji] E2E invoice (28 GT): 56/77 -> 63/77 baris, tepat 14/28 -> 16/28; flag mati tetap 0.
  - [diuji] Fixture: vector 0.978 dan phrase 0.978 (tidak berubah oleh D-042 setelah penjaga header-lain ditambahkan; versi pertama sempat menurunkan phrase ke 0.933 dan terdeteksi sebelum commit).
  - [diuji] Holdout run 4 (kode akhir) identik dengan run 3.
- Tidak terverifikasi / risiko: lihat `REPORT_FINAL.md` bagian 7.
- Langkah berikutnya: tidak ada fase tersisa di rencana; butuh keputusan pemilik (REPORT_FINAL bagian 8).
