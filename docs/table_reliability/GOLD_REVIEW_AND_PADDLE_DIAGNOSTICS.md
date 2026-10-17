# Review Gold, Evaluasi Akurasi, dan Diagnostik PaddleOCR di Windows

> **Pembaruan Fase 8 (2026-10-16):** dokumen ini adalah catatan historis. Status terkini, metrik, dan hal yang belum terverifikasi ada di `REPORT_FINAL.md`; keputusan di `DECISIONS.md`; hal di luar cakupan di `OUT_OF_SCOPE_NOTES.md`. Bagian yang terbukti keliru sudah dikoreksi di tempat dan ditandai **[dikoreksi Fase 8]**.

## Tujuan dan status saat ini

Panduan ini menetapkan proses *human-review* yang dapat direproduksi (*reproducible*)
untuk evaluasi tabel, baris, sel, dan OCR pada `D:\AIC\DATASET\Dataset`. Dokumen
ini juga mendefinisikan *diagnostic gate* Windows yang wajib lolos sebelum
mengevaluasi PaddleOCR pada citra raster.

Fakta saat ini:

- Korpus memiliki empat jenis dokumen: Invoice, PoD, Rate Card, dan Surat Jalan.
  Masing-masing memiliki folder sumber `image` dan `pdf`.
- Deteksi keberadaan (*presence*) tabel/vektor PDF berhasil diukur setelah
  perbaikan koordinat. Namun, *presence* bukanlah akurasi.
- **[dikoreksi Fase 8]** PaddleOCR 3.7 + PaddlePaddle 3.3.1 (CPU, oneDNN mati) menghasilkan *evidence* untuk 52 dari 52
  PNG dataset (`REPORT_FASE_6.md`). Kecepatan masih menjadi kendala (model default PP-OCRv6 medium sekitar 70 sampai 120 detik
  per halaman). Akurasi OCR terhadap teks yang direview manusia belum diukur; jangan melaporkan metrik OCR sebelum ada gold teks.
- `app/tools/prelabel_tables.py` dapat membuat draf yang belum direview (*unreviewed*).
  Draf ini bukan merupakan *source of truth*. Saat ini, `scripts/eval_tables.py`
  hanya mengevaluasi kecocokan baris persis (*exact rows*) dan menolak file gold
  yang tidak memiliki `reviewed: true`.

## Aturan review mutlak (Non-negotiable)

1. Lakukan review terhadap halaman asli yang dirender, jangan pernah hanya
   mengandalkan output OCR.
2. Pertahankan teks sumber secara persis apa adanya pada `raw_text`; letakkan
   nilai kanonikal opsional pada `normalized_text`. Dilarang mengoreksi teks
   sumber secara diam-diam.
3. Tandai konten yang tidak terbaca sebagai `"[ILLEGIBLE]"`; jangan menebak karakter.
4. Jangan memasukkan blok judul, alamat, total/ringkasan, tanda tangan, atau teks
   footer sebagai baris item (*item rows*). Total/ringkasan harus dimasukkan ke
   `summary_rows`.
5. Satu reviewer menyiapkan draf dan reviewer kedua menyetujuinya (*approve*).
   Reviewer tidak boleh menyetujui draf buatannya sendiri.
6. `reviewed: true` berarti seluruh sel dan batas baris telah diverifikasi. Ini
   bukan pernyataan bahwa model OCR sudah benar.
7. Setiap perubahan wajib mencantumkan ID reviewer, timestamp UTC, dan alasan
   ringkas.

## Pemilihan dataset dan pembagian (split)

Gunakan hanya struktur direktori sumber berikut:

```
D:\AIC\DATASET\Dataset\invoice\{image,pdf}
D:\AIC\DATASET\Dataset\PoD\{image,pdf}
D:\AIC\DATASET\Dataset\rate card\{image,pdf}
D:\AIC\DATASET\Dataset\surat jalan\{image,pdf}
```

Buat manifest sebelum melakukan review. Manifest wajib mencatat *relative path*,
SHA-256, jenis dokumen, jenis sumber (*source kind*), jumlah halaman, status
reviewer, dan *split*. Gunakan *stratified split*: setiap jenis dokumen dan
setiap jenis sumber harus ada dalam set review. Simpan dokumen bertemplat serupa/
duplikat pada satu *split* yang sama untuk mencegah hasil evaluasi yang terinflasi
semu. Rekomendasi pembagian:

- **development**: menyelesaikan perbedaan pemahaman skema dan ketidaksepakatan reviewer;
- **validation**: menentukan *threshold* operasional atau membandingkan varian;
- **holdout**: tidak disentuh sama sekali hingga tahap *release candidate*.

Jangan mengganti file yang hilang dengan file bernama serupa. Setiap perubahan
pada manifest memulai versi evaluasi yang baru.

## Format file gold

Simpan satu file per dokumen di `data/gold/tables/<document_id>.json`. Skema
berikut adalah target review yang diwajibkan; field yang belum digunakan oleh
skrip saat ini sengaja dipertahankan untuk evaluasi sel/OCR di masa mendatang.

```json
{
  "document_id": "stable-relative-path-hash-or-id",
  "source": {"relative_path": "invoice/pdf/example.pdf", "sha256": "...", "source_kind": "pdf"},
  "document_type": "INVOICE",
  "reviewed": false,
  "review": {"schema_version": "1", "prepared_by": "", "prepared_at_utc": "", "approved_by": "", "approved_at_utc": "", "notes": ""},
  "pages": [{
    "page_number": 1,
    "tables": [{
      "table_id": "doc-p1-t1",
      "bbox_norm": [0.0, 0.0, 1.0, 1.0],
      "headers": [{"raw_text": "Qty", "normalized_text": "quantity", "bbox_norm": [0.0, 0.0, 0.0, 0.0]}],
      "rows": [{
        "row_id": "doc-p1-t1-r1",
        "kind": "item",
        "cells": [{"column_index": 0, "raw_text": "2", "normalized_text": "2", "bbox_norm": [0.0, 0.0, 0.0, 0.0], "legibility": "clear"}]
      }],
      "summary_rows": []
    }]
  }]
}
```

`bbox_norm` menggunakan format `[x1, y1, x2, y2]` dalam rentang `[0, 1]`. Sel
dapat bernilai `null` untuk sel gabungan (*merged cell*) yang memang kosong.
Sel yang hilang atau tidak terbaca bukanlah sel kosong; gunakan `[ILLEGIBLE]`
dan `legibility: "unclear"`.

## Alur kerja human review

1. Buat atau salin draf unreviewed. Pertahankan status `reviewed: false`.
2. Buka halaman sumber dengan perbesaran 100% atau lebih; catat nilai SHA-256
   pada file gold.
3. Tandai setiap batas tabel dan header. Pisahkan tabel independen yang berada
   pada halaman yang sama.
4. Transkripsikan header dan setiap baris item dari kiri ke kanan. Pertahankan
   *wrapping* teks yang berada di dalam sel logis yang sama. Tandai baris
   ringkasan/total (*summary rows*) secara terpisah.
5. Verifikasi aritmetika tabel menggunakan `Decimal`: kuantitas × tarif dibandingkan
   dengan total baris (*line total*), serta subtotal/pajak/total akhir jika tersedia.
   Catat jika ada perbedaan; jangan pernah menimpa teks sumber hanya agar
   aritmetika cocok.
6. Reviewer kedua mengulangi pemeriksaan secara independen (*blind review* jika
   memungkinkan).
7. Selesaikan perbedaan pembacaan dengan merujuk langsung ke citra/PDF sumber.
   Eskalasikan konten yang benar-benar tidak terbaca; jangan memvotingnya menjadi
   nilai tebakan.
8. Isi metadata reviewer, ubah menjadi `reviewed: true`, dan kunci file yang
   telah direview.

## Metrik dan pelaporan

Laporkan setiap metrik secara terpisah berdasarkan jenis dokumen dan jenis
sumber, disertai pembilang (*numerator*), penyebut (*denominator*), item yang
dikecualikan, serta rentang kepercayaan (*confidence interval*) jika ukuran
sampel mencukupi.

| Metrik | Aturan pencocokan | Yang tidak dibuktikan oleh metrik ini |
|---|---|---|
| Precision/Recall deteksi tabel | IoU >= threshold yang disepakati dan halaman cocok | Kebenaran isi sel |
| Precision/Recall/F1 baris | Urutan sekuens nilai sel ternormalisasi yang persis sama | Kualitas OCR tingkat karakter |
| Akurasi sel | Nilai sel ternormalisasi persis sama, dilaporkan per kolom | Pengelompokan baris yang benar |
| OCR CER | Jarak edit karakter Levenshtein / total karakter gold | Struktur tabel |
| OCR WER | Jarak edit kata / total kata gold | Kebenaran numerik |
| Numeric exact rate | Kecocokan persis setelah normalisasi Decimal | Kebenaran OCR untuk kolom teks |
| Arithmetic pass rate | Nilai gold/prediksi memenuhi relasi matematika yang terdokumentasi | Akurasi OCR |

Kecualikan hanya item yang bertanda `[ILLEGIBLE]`, dan laporkan jumlahnya. Jangan
menganggap *confidence* sebagai akurasi. `eval_tables.py` saat ini hanya mendukung
metrik baris persis; IoU tabel, sel, CER/WER, dan skor aritmetika memerlukan
ekstensi evaluator lanjutan terhadap skema gold ini.

## Perintah manual

Jalankan regression test terarah:

```powershell
cd D:\AIC\ai-service
$env:ENABLE_TABLE_EXTRACTOR = "true"
$env:ENABLE_OCR_PREPROCESS = "true"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_table_router.py tests/unit/test_pymupdf_table_provider.py -q
```

Evaluasi file gold yang sudah direview beserta JSON prediksi menggunakan evaluator saat ini:

```powershell
.\.venv\Scripts\python.exe scripts\eval_tables.py --gold <gold.json> --prediction <prediction.json>
```

Jika evaluator menolak file yang belum direview (*unreviewed*), hal tersebut adalah
perilaku normal yang disengaja sebagai *quality gate*.

## Gate diagnostik PaddleOCR di Windows

Jalankan perintah-perintah berikut di sesi PowerShell yang bersih. Simpan seluruh
output stdout/stderr dan *exit code* untuk setiap pengetesan. Jangan pernah
mendiagnosis sekaligus terhadap 52 gambar.

```powershell
cd D:\AIC\ai-service
.\.venv\Scripts\python.exe -c "import paddle, paddleocr, pandas; print(paddle.__version__); print(paddleocr.__version__); print(pandas.__version__)"
.\.venv\Scripts\python.exe -c "import platform; print(platform.platform()); print(platform.processor())"
Get-ChildItem .\.venv\Lib\site-packages\paddle\base\core*.pyd
```

Untuk satu gambar nyata, jalankan proses dengan *fault handling* aktif dan segera
tangkap *exit code*-nya:

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
.\.venv\Scripts\python.exe -u -X faulthandler -c "# panggil PaddleOCR pada tepat satu gambar pilihan"
Write-Host "exit code: $LASTEXITCODE"
```

Interpretasi hasil:

- Python traceback: simpan trace; ini merupakan error aplikasi/dependensi.
- Exit non-zero tanpa traceback: periksa Windows Event Viewer > Windows Logs
  > Application untuk proses Python tersebut dan DLL penyebab crash.
- `0xc000007b`, `0xc0000135`, atau error pemuatan DLL: perbaiki Microsoft Visual C++
  Redistributable dan pastikan keselarasan arsitektur 64-bit antara Python dan DLL Paddle.
- Illegal instruction: periksa dukungan instruksi CPU dan gunakan *wheel*/lingkungan
  CPU yang didukung; jangan menyembunyikan masalah ini dengan sekadar mencoba ulang (*retry*).
- Terminasi hening (*silent termination*) hanya pada gambar nyata: catat dimensi
  gambar, mode warna, dan versi paket yang persis; uji gambar versi *downscaled*
  hanya sebagai bahan diagnosis, bukan sebagai bukti produksi.

Simpan matriks versi yang mencakup Python, PaddlePaddle, PaddleOCR, PaddleX, pandas,
NumPy, OpenCV, CPU, Windows build, perintah yang dijalankan, hash sumber, *exit code*,
dan hasil. Ubah hanya satu variabel dalam satu waktu. Jangan melakukan downgrade/upgrade
paket secara acak atau meng-commit versi dependensi sebelum satu gambar nyata
berhasil diproses secara konsisten berulang kali.

## Protokol batch setelah keberhasilan satu gambar

1. Jalankan ulang gambar yang sama sebanyak 10 kali dalam proses terpisah;
   harus menghasilkan 10 output evidence dan catat latensinya.
2. Jalankan satu gambar per jenis dokumen dan catat seluruh hasilnya.
3. Jalankan seluruh 52 gambar dalam *child process* terisolasi dengan batas waktu
   (*timeout*) per file.
4. Buat laporan JSON/CSV berisi path, SHA-256, dimensi, *exit code*, waktu proses,
   jumlah baris, jumlah tabel, pilihan preprocessing, dan pesan error.
5. Jangan menyamarkan kegagalan dengan diam-diam mencoba ulang hingga berhasil.
   Laporkan tingkat keberhasilan (*success rate*) dan setiap path yang gagal.
6. Baru setelah itu, bandingkan varian mentah (*raw*) dan varian preprocessing
   terhadap data gold yang telah direview.

## Serah terima (Handoff) Gemini Antigravity

Gunakan konteks berikut secara utuh (*verbatim*), kemudian lampirkan file ini
beserta output stdout/stderr kegagalan yang persis:

```text
Repository: D:\AIC\ai-service, branch feat/table-reliability.
PaddleOCR 3.7.0 / PaddlePaddle 3.3.1 terpasang pada Windows CPU. MKLDNN dinonaktifkan
karena sebelumnya memicu error konversi oneDNN PIR. Model PaddleOCR berhasil dimuat
dari cache dan inferensi pada gambar kosong (blank image) mengembalikan OCRResult.
Namun pada gambar PNG nyata berukuran 800x684, proses mencapai status "ready" lalu
keluar (exit) tanpa menghasilkan evidence, tanpa Python traceback, atau tanpa output
faulthandler, termasuk saat dijalankan dengan cpu_threads=1 dan MKLDNN dinonaktifkan.
Jangan mengklaim bahwa OCR raster sudah berfungsi. Lakukan diagnosis terhadap exit code,
modul bermasalah pada Windows Event Viewer, kompatibilitas arsitektur/runtime DLL,
dan uji matriks versi satu per satu variabel. Jangan mengubah backend-api atau
menambahkan hardcode khusus dokumen. Ekstraksi tabel vektor PDF berjalan baik:
137 PDF / 159 halaman berhasil diproses dengan deteksi keberadaan tabel setelah
pemotongan bbox (bbox clipping), namun ini bukan akurasi. Human gold review wajib
dilakukan sebelum melaporkan akurasi tabel, baris, sel, atau OCR.
```
