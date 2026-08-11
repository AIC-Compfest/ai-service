# Roadmap — ai-service

Mengikuti Implementation Order dari PRD blueprint (§44), bagian yang relevan
untuk service ini:

- [x] Fase 0: scaffold repo, config, health check
- [ ] Fase 1: integrasi dengan canonical schema (dari repo `infra`/`schemas`)
- [ ] Fase 2: document classifier (INVOICE/POD/SURAT_JALAN/RATE_AGREEMENT/UNKNOWN)
- [ ] Fase 3: image preprocessing (OpenCV — deskew, perspective correction, dst)
- [ ] Fase 4: OCR pipeline (PP-OCRv6 + PP-StructureV3, fallback PaddleOCR-VL)
- [ ] Fase 5: structured extraction per document type (Invoice/POD/SuratJalan/RateAgreement extractor terpisah — PRD §23)
- [ ] Fase 6: signature presence detection (PRD §33)
- [ ] Fase 7: evaluation harness terhadap synthetic ground truth

**Penting:** jangan bangun satu fungsi generik `extract_document()` — harus
extractor terpisah per document type (PRD §23). Jangan panggil VLM berat
untuk semua dokumen jika jalur OCR/layout biasa sudah cukup (PRD §22).
