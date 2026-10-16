## Fase 7: ringkasan
- Dikerjakan (commit `feat(runtime): shared pipeline, upload limit, page retention, pinned dependencies, Dockerfile ...`):
  1. `requirements.txt` dipin ke versi yang dipakai suite uji; `requirements-paddle.txt` terpisah; `requirements-dev.txt` memuat keduanya. `streamlit`, `pandas`, dan `scipy` dikeluarkan dari manifest runtime karena tidak ada satu pun kode Python yang mengimpornya (grep seluruh repo di luar `.venv`). `opencv-contrib` tidak dimasukkan; tidak ada API contrib yang dipakai.
  2. `Dockerfile` multi-stage, pengguna non-root (uid 10001), `COPY config` dan `models`, `HEALTHCHECK` ke `/healthz`, `PYTHONUNBUFFERED`, argumen `WITH_PADDLE`; `.dockerignore` baru.
  3. Pipeline tunggal (`app/core/runtime.py`, `lifespan`), menggantikan `Phase5DocumentPipeline()` per permintaan (classifier dimuat ulang tiap permintaan sebelumnya) (K10).
  4. Endpoint berat menjadi `def` sinkron (dijalankan FastAPI di threadpool) dan batas `MAX_UPLOAD_SIZE_MB` ditegakkan (HTTP 413) di semua endpoint unggah; sebelumnya setelan itu tidak dipakai.
  5. `PAGE_OUTPUT_DIR` dan `PAGE_RETENTION_HOURS`: PNG halaman tidak lagi menumpuk tanpa batas (hapus folder lebih tua dari 24 jam, hanya sub-folder langsung dari direktori keluaran). Kontrak `page_image.path` tetap relatif terhadap direktori kerja seperti sebelumnya.
  6. Setelan yang terbukti tidak terpakai dihapus: `ocr_engine`, `ocr_fallback_engine`, `ocr_backend` (grep: hanya `config.py` dan `.env.example`).
  7. Path absolut dihapus dari `test_hard_surat_jalan.py`, `test_ocr_dataset_comprehensive.py`, `test_ocr_manual.py`, `audit_ocr_candidate_dataset.py` (kini `VERIFLOW_DATASET_ROOT` atau `../DATASET/Dataset`, dan path gambar mengikuti struktur `Dataset/<jenis>/image`). Skrip tersebut tidak dijalankan ulang.
  8. `.env.example` dan `README.md` diperbarui.
- Bukti:
  - [diuji] `pytest`: 334 lulus, 1 dilewati. `tests/unit/test_runtime.py` (6): pipeline satu instans, 413 di lima endpoint, retensi, nol jam menonaktifkan, PNG ke direktori konfigurasi, dan impor aplikasi serta pencarian config/model dari direktori kerja lain (subprocess).
  - [kode] `app` hanya mengimpor paket yang ada di manifest runtime (daftar impor dari seluruh `app/` di luar `tools/`).
- Tidak terverifikasi (blocked):
  - Build image Docker: tidak ada daemon (hanya klien; WSL berhenti). Dockerfile belum pernah dibangun atau dijalankan.
  - `uvicorn` tidak terpasang di venv utama, jadi server sesungguhnya belum dijalankan. Pengganti: tes subprocess dari direktori lain yang mengimpor aplikasi dan membaca config dan model, ditambah `TestClient`.
  - Kecocokan versi di container: model klasifikasi di-pickle dengan scikit-learn 1.7.2 dan 1.9.0, venv dan manifest memakai 1.5.2 (`InconsistentVersionWarning`). Hasil klasifikasi lulus tes, tetapi risiko hasil keliru tersembunyi belum dihapus; model sebaiknya dilatih ulang pada versi yang dipin.
  - Perilaku konkurensi di bawah beban nyata (beberapa permintaan sekaligus) tidak diuji; hanya kunci Paddle yang diuji dengan predictor palsu.
  - Perubahan sinkron/threadpool tidak diuji terhadap klien backend sungguhan (Go, di luar cakupan).
- Keputusan yang dibutuhkan dari pemilik: bangun image di mesin dengan Docker dan jalankan `/healthz`; tentukan apakah klasifikasi dilatih ulang pada scikit-learn yang dipin.
- Langkah berikutnya: Fase 8.
