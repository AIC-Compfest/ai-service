# ai-service
Document understanding service — bagian dari **Evidence-First AI 3PL Invoice
Reconciliation Engine** (blueprint di repo `infra`, `docs/PRD_MVP.md`).

## Tanggung jawab
Service ini **hanya** menangani:
- klasifikasi dokumen per halaman dan segmentasi (INVOICE / POD / SURAT_JALAN / RATE_AGREEMENT / UNKNOWN)
- ingest dan OCR: PDF digital dibaca PyMuPDF, halaman scan/gambar dibaca PaddleOCR
- deteksi tabel (header, kolom, baris, ringkasan, beberapa tabel per halaman, lanjutan antar-halaman)
- ekstraksi field per jenis dokumen (extractor berbasis aturan; kandidat + LightGBM sebagai jalur tambahan)
- validasi aritmetika deterministik sebagai bukti tambahan
- deteksi keberadaan tanda tangan/stempel (bukan verifikasi identitas)

Service ini **stateless**: tidak menyimpan data bisnis dan tidak membuat keputusan finansial
(itu tanggung jawab `backend-api`). Confidence tinggi bukan kebenaran bisnis.

## Alur (ringkas)
`upload` → ingest (PNG kanonik per halaman) → page-kind (digital/scan) → teks PyMuPDF atau PaddleOCR →
klasifikasi per halaman → segmentasi → tabel sadar-tipe → extractor aturan → jalur ML (kandidat + LightGBM) →
validasi aritmetika → respons. Penjelasan dan status tiap tahap: `docs/table_reliability/REPORT_FINAL.md`.

## Kontrak
Setiap extracted value membawa `value`, `confidence`, `confidence_source`, `source` (document_id, page, bbox) dan `method`.
Field yang tidak bisa diekstrak adalah `is_unknown=true`, tidak dikarang (PRD §60). Respons `/v1/extract/pipeline/full`
menambah `ml_extraction`, `arithmetic_validation` dan `table_groups`; semuanya aditif.

## Konfigurasi (env, lihat `.env.example`)
| Variabel | Default | Arti |
|---|---|---|
| `ENABLE_TABLE_EXTRACTOR` | `true` | Tabel di evidence dan line item invoice. `false` = tanpa tabel |
| `ENABLE_ML_EXTRACTION` | `true` | Kandidat + LightGBM pada `/v1/extract/pipeline/full` (hanya bayangan; model belum berguna) |
| `ENABLE_ARITHMETIC_EVIDENCE` | `true` | `arithmetic_validation` di respons |
| `ENABLE_ARITHMETIC_VALIDATION` | `false` | Campuran skor aritmetika pada pemilihan tabel (tidak berdampak terukur) |
| `MAX_UPLOAD_SIZE_MB` | `25` | Di atasnya HTTP 413 |
| `PAGE_OUTPUT_DIR` / `PAGE_RETENTION_HOURS` | `out/pages` / `24` | Lokasi PNG halaman dan masa simpan |
| `PADDLE_DET_MODEL`, `PADDLE_REC_MODEL`, `PADDLE_DET_LIMIT_SIDE_LEN` | kosong | Model Paddle. Default PaddleOCR 3.7 untuk bahasa Inggris = PP-OCRv6 medium (sekitar 70 s per halaman scan di CPU 6 inti); PP-OCRv5 mobile sekitar 3 kali lebih cepat |

## Jalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # runtime + Paddle + alat uji
cp .env.example .env
uvicorn app.main:app --reload --port 8100
```
Windows: `.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8100`. Perintah dapat dijalankan dari direktori mana pun
selama `app` ada di `PYTHONPATH`; semua file config dan model dicari relatif terhadap paket, bukan direktori kerja.

## Docker
`docker build -t ai-service .` (Paddle ikut; `--build-arg WITH_PADDLE=0` untuk tanpa OCR raster). Image berjalan sebagai
pengguna non-root dan punya `HEALTHCHECK` ke `/healthz`. Model PaddleOCR diunduh saat pemakaian pertama (butuh jaringan).

## Test dan evaluasi
```bash
pytest                                   # 300+ test, tanpa PaddleOCR nyata
RUN_PADDLE_TESTS=1 pytest tests/unit/test_raster_path.py   # menjalankan PaddleOCR sungguhan (lambat)
python scripts/eval_tables.py --help     # evaluator tabel; lihat docs/table_reliability/
```

## Status
Perbaikan keandalan tabel: `docs/table_reliability/REPORT_FINAL.md` (metrik sebelum/sesudah, target yang belum tercapai,
hal yang tidak terverifikasi). Gold tabel masih draf sintetis, belum direview manusia.
