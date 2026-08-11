# ai-service

Document understanding service — bagian dari **Evidence-First AI 3PL Invoice
Reconciliation Engine** (lihat blueprint di repo `infra`, file
`docs/PRD_MVP.md`).

## Tanggung jawab

Service ini **hanya** menangani:

- document classification (INVOICE / POD / SURAT_JALAN / RATE_AGREEMENT / UNKNOWN)
- image preprocessing (OpenCV: deskew, perspective correction, dll)
- OCR + layout understanding (PP-OCRv6 + PP-StructureV3, fallback PaddleOCR-VL)
- structured field extraction per jenis dokumen
- signature presence detection (bukan verifikasi identitas)

Service ini **stateless** — tidak menyimpan data bisnis, tidak melakukan
pricing/reconciliation/keputusan finansial apapun (itu tanggung jawab
`backend-api`). Lihat PRD §70 poin 1 & 3: jangan ganti deterministic pricing
dengan LLM, jangan anggap confidence tinggi = business truth.

## Kontrak

Semua response mengikuti Evidence Model (PRD §18): setiap extracted value
membawa `value`, `confidence`, `source` (document_id, page, bbox), dan
`method`. Field yang tidak bisa diekstrak harus `UNKNOWN` /
`INSUFFICIENT_EVIDENCE`, tidak boleh dikarang (PRD §60).

## Jalankan lokal

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8100
```

## Test

```bash
pytest
```

## Status

🚧 Scaffold awal — endpoint masih stub, model inference (PP-OCRv6 dkk) belum
diintegrasikan. Lihat `docs/ROADMAP.md`.
