## Fase 5: ringkasan
Stempel: GT sintetis; tidak ada gold manusia. Angka cakupan di bawah adalah cakupan aturan, bukan metrik kualitas tabel.
- Dikerjakan:
  1. Registri aritmetika di YAML (`config/validation/arithmetic.yaml`): peran header (quantity, rate, weight, line_total, gross, discount, tax, net), `ambiguous` ("jumlah"), aturan baris (`qty_x_rate`, `weight_x_rate`, `qty_x_rate_gross`, `gross_minus_discount`) dengan `unless` (aturan produk dilewati bila ada kolom diskon/pajak), toleransi per mata uang (K3).
  2. `validation/arithmetic.py` ditulis ulang: `validate_rows` (daftar sel), `validate_table` (adapter), `validate_document_totals`; aman terhadap nilai kosong dan baris pendek; `suggested_value` hanya bila selisih satu karakter; kolom persen tidak pernah mengambil peran uang.
  3. `validation/document.py` + `/v1/extract/pipeline/full`: `arithmetic_validation` per segmen dan tingkat dokumen (aturan baris pada tabel `line_items`, subtotal terhadap jumlah total baris, grand total terhadap subtotal + pajak). Aditif, tidak mengubah nilai atau konfidensi.
  4. Sumber konfidensi: `ExtractedField.confidence_source` (aditif), `extraction/confidence.py`, konfidensi tabel di evidence kini skor struktural (dulu 0.0), `line_items`/`rate_matrix` dari tabel memakai skor struktural terlemah (maks 0.95), `CONFIDENCE.md`.
  5. Impor melingkar `validation` <-> `candidates` diperbaiki (impor malas di `router.py`).
- Bukti:
  - [diuji] `pytest` penuh: 324 lulus (sebelumnya 305). Test baru: 10 di `test_arithmetic_validation.py`, 8 di `test_arithmetic_evidence_and_confidence.py`. Dua perilaku lama sengaja berubah (K3): saran koreksi hanya bila selisih satu karakter; tanpa toleransi relatif 1%.
  - [diuji] Cakupan pada 28 invoice GT (`BASELINE/arithmetic_coverage_after_p5.txt`): aturan baris berlaku pada 7 dari 28 dokumen; 21 dari 132 baris tabel dievaluasi (16%), 21 lulus. Registri sendiri pada nilai GT: `quantity x unit_price = amount` berlaku pada 23 dari 25 item.
  - [diuji] Subtotal: 6 lulus, 4 gagal, 6 dilewati, 12 tidak berlaku. Grand total: 7 lulus, 9 gagal, 12 tidak berlaku.
  - [diuji] Contoh nyata: `invoice_15_variasi_1` extractor mengeluarkan grand_total 1.300.098.765.432 (GT 9.350.000); pengecekan menandainya GAGAL, jadi menangkap kesalahan ekstraksi. `invoice_4_variasi_1`: tax 5.304 dan grand_total 3.000.000 (GT total 12.645.000), juga ditandai.
  - [diuji] `ENABLE_ARITHMETIC_VALIDATION` (campuran skor router) hidup vs mati pada dev+val: identik (F1 0.641, header 41/56, baris 49/58). Tetap false.
- Target K4: tidak berubah (Fase 5 tidak menyentuh deteksi struktur).
- Temuan baru / koreksi:
  - Konfidensi tabel di evidence selalu 0.0 sebelumnya; kini skor struktural 0.57 sampai 1.0 (median 0.91) pada invoice dev+val.
  - Sebagian FAIL adalah alarm palsu, bukan kesalahan ekstraksi: `invoice_3` (kolom total terbaca "Jumlah" padahal "Tarif" yang benar, subtotal GT sama dengan hasil ekstraksi) dan dokumen dengan diskon tingkat dokumen (`invoice_15`, diskon 330.000 tidak dimodelkan).
  - Grand total yang diturunkan extractor dari subtotal + pajak membuat pengecekan sirkular (OS-008); ditulis sebagai `caveats` di respons.
  - Sesi PowerShell mempertahankan variabel antar-panggilan; satu edit YAML sempat tertimpa isi variabel lama dan diperbaiki segera (diverifikasi dengan membaca ulang file).
- Tidak terverifikasi / risiko:
  - Cakupan rendah (16% baris): tabel dengan header tak dikenal, tanpa kolom harga, atau data tidak berbentuk qty x harga tidak dicek.
  - Alias header ditambah dengan melihat header dev+val saja (holdout tidak dilihat), tetapi evaluasi cakupan memakai seluruh 28 invoice.
  - Tidak ada tabel gold tentang apakah FAIL benar-benar kesalahan; hanya contoh yang diperiksa manual.
  - Perubahan nilai `confidence` `line_items` (0.90 -> skor struktural) dapat memengaruhi konsumen downstream yang memakai ambang; Go backend tidak diperiksa (read-only, di luar cakupan).
- Keputusan yang dibutuhkan dari pemilik: apakah konsumen backend memakai ambang pada `confidence` `line_items` (perubahan nilai dari 0.90 tetap menjadi 0.57 sampai 0.95).
- Langkah berikutnya: Fase 6 (Paddle: triase di `.venv-exp`, isolasi proses, evaluasi raster).
