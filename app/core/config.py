"""
Application settings. Semua threshold/config sensitif ke bisnis TIDAK boleh
di-hardcode di sini (PRD §62) — service ini hanya baca operational config
milik dirinya sendiri (port, log level, model path). Business threshold
(materiality, matching score, dll) tetap tinggal di backend-api / repo infra.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8100

    # PRD §20 — di bawah nilai ini, classification harus dianggap gagal
    # dan diteruskan sebagai low-confidence result ke backend-api untuk
    # keputusan human-review. Nilai ini HANYA default operasional service,
    # keputusan akhir threshold tetap di backend-api (configs/thresholds.yaml).
    default_classification_confidence_floor: float = 0.0

    ocr_engine: str = "pp-ocr-v6"  # placeholder, diaktifkan Fase 4
    ocr_fallback_engine: str = "paddleocr-vl-1.6"

    max_upload_size_mb: int = 25

    # -------------------------------------------------------------------------
    # Feature Flags — feat/table-reliability
    # Default SEMUA flag: aman (perilaku lama). Rollback ke perilaku lama
    # cukup dengan mengubah config ini kembali ke default.
    # -------------------------------------------------------------------------

    # Langkah 1: Deteksi halaman digital vs scan per halaman
    # False = pakai logika lama (len(words) < 5)
    # True  = pakai PageKindDetector dari app/services/preprocessing/page_kind.py
    page_kind_routing: bool = False

    # Threshold untuk PageKindDetector (Langkah 1)
    # Rasio luas union bbox kata terhadap luas halaman — di atas ini = DIGITAL
    page_kind_word_coverage_min: float = 0.05
    # Rasio luas gambar terhadap luas halaman — di atas ini bisa SCAN
    page_kind_image_coverage_scan_threshold: float = 0.70
    # Rasio satu gambar terhadap luas halaman untuk dianggap full-page scan.
    page_kind_full_page_image_threshold: float = 0.90
    # Legacy word-count is retained only as the final scan safety net.
    page_kind_legacy_word_count_floor: int = 5

    # Langkah 2: Router provider tabel
    # Urutan provider yang dicoba per halaman.
    # Tersedia: "token_grid", "pymupdf_find_tables", "ppstructure"
    # Digital PDF pages try PyMuPDF before geometry-only token-grid discovery.
    table_provider_order: list[str] = ["pymupdf_find_tables", "token_grid"]
    # False keeps the legacy OCR evidence contract untouched. True routes
    # table discovery through the scored provider router.
    enable_table_extractor: bool = False
    # False keeps each page table independent. True may join consecutive-page
    # tables only when their normalized headers and provider match.
    enable_table_stitching: bool = False

    # Langkah 3: Validasi aritmetika
    # False = tidak menjalankan validasi aritmetika (perilaku lama)
    # True  = jalankan validasi dan turunkan confidence berdasarkan hasil
    enable_arithmetic_validation: bool = False

    # Toleransi relatif untuk cek qty * rate ≈ line_total  (1% default)
    arith_row_tolerance_relative: float = 0.01
    # Toleransi absolut (Rupiah) untuk pembulatan
    arith_row_tolerance_absolute_idr: float = 1.0

    # Langkah 6: PaddleOCR is the only raster OCR backend.
    ocr_backend: str = "paddleocr"
    paddle_language: str = "en"
    paddle_enable_mkldnn: bool = False
    paddle_cpu_threads: int = 8

    # Langkah 7: Preprocessing untuk scan/foto
    # False = tidak ada preprocessing tambahan (perilaku lama)
    # True  = aktifkan deskew, perspective correction, adaptive upscale
    enable_ocr_preprocess: bool = False


settings = Settings()
