# ENV_FACTS

Dikumpulkan di Fase 0 (2026-10-16) lewat perintah, bukan tebakan. Tag: `[diuji]`.

## Mesin [diuji]

| Item | Nilai |
|---|---|
| OS | Microsoft Windows NT 10.0.26200.0 |
| CPU | AMD Ryzen 5 5500U with Radeon Graphics (6 core / 12 thread) |
| RAM | 7,4 GB |
| GPU | AMD Radeon(TM) Graphics (terintegrasi; tidak dipakai Paddle CPU) |
| Docker | `docker version`: hanya Client 29.4.0 yang terdeteksi. Pemilik menyatakan Docker tidak tersedia, jadi build image = `blocked` |
| WSL | Ubuntu dan docker-desktop terpasang, keduanya `Stopped` (WSL2) |

## Python dan paket di `.venv` utama [diuji: `pip list`, `pip check`]

- Python 3.11.9. `pip check`: No broken requirements found.
- paddlepaddle 3.3.1, paddleocr 3.7.0, paddlex 3.7.2
- numpy 1.26.4, pandas 2.3.3, pillow 10.4.0, PyMuPDF 1.28.2
- opencv-python-headless 4.10.0.84 **dan** opencv-contrib-python 4.10.0.84 (keduanya, T-27 terkonfirmasi)
- easyocr 1.7.2, torch 2.14.1, torchvision 0.29.1 (sisa EasyOCR, T-27 terkonfirmasi), scikit-image 0.26.0
- scikit-learn 1.5.2 (hanya satu versi terpasang; klaim "dua dist-info" TIDAK terkonfirmasi lewat `pip list`)
- lightgbm 4.7.0, RapidFuzz 3.14.6, PyYAML 6.0.2, fastapi 0.115.0
- Temuan baru: model `app/models/*.joblib` dilatih dengan scikit-learn 1.7.2, dimuat dengan 1.5.2
  (`InconsistentVersionWarning`). `requirements.txt` hanya menulis `scikit-learn>=1.5.0`.

## Hasil test baseline [diuji]

`pytest -q`: 150 passed, 1 failed (`test_document_ingestion_jpg`), 27 warnings, 247 detik. Lihat `BASELINE/pytest_before.txt`.

## Catatan Paddle

Versi di atas adalah yang dipakai D-023. Status inferensi PNG nyata: 52 dari 52 PNG dataset berhasil di venv ini (model mobile, mkldnn mati), tanpa crash; rinci di `REPORT_FASE_6.md`.
