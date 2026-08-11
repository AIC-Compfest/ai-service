"""
Application settings. Semua threshold/config sensitif ke bisnis TIDAK boleh
di-hardcode di sini (PRD §62) — service ini hanya baca operational config
milik dirinya sendiri (port, log level, model path). Business threshold
(materiality, matching score, dll) tetap tinggal di backend-api / repo infra.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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


settings = Settings()
