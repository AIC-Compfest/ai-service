"""
Application settings. Semua threshold/config sensitif ke bisnis TIDAK boleh
di-hardcode di sini (PRD §62) — service ini hanya baca operational config
milik dirinya sendiri (port, log level, model path). Business threshold
(materiality, matching score, dll) tetap tinggal di backend-api / repo infra.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/ai-service; independent of the process working directory (T-24, T-25).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Directory that holds candidates/, training/ ... YAML. Override with CONFIG_DIR (e.g. in a container).
    config_dir: Path = PACKAGE_ROOT / "config"

    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8100

    # PRD §20 — di bawah nilai ini, classification harus dianggap gagal
    # dan diteruskan sebagai low-confidence result ke backend-api untuk
    # keputusan human-review. Nilai ini HANYA default operasional service,
    # keputusan akhir threshold tetap di backend-api (configs/thresholds.yaml).
    default_classification_confidence_floor: float = 0.0


    max_upload_size_mb: int = 25
    # Canonical page PNGs written per request (evidence page_image.path points here; relative to the working directory).
    page_output_dir: Path = Path("out/pages")
    # Folders older than this are deleted (0 = never). The service is stateless; the PNGs are a cache for viewers.
    page_retention_hours: int = 24

    # -------------------------------------------------------------------------
    # Feature Flags — feat/table-reliability
    # Default SEMUA flag: aman (perilaku lama). Rollback ke perilaku lama
    # cukup dengan mengubah config ini kembali ke default.
    # -------------------------------------------------------------------------

    # Langkah 2: Router provider tabel
    # Docling TableFormer is the only table provider.
    table_provider_order: list[str] = ["docling"]
    # False keeps the legacy OCR evidence contract untouched. True routes
    # table discovery through the scored provider router.
    # Default true since Fase 4 (D-035): with false the API returns no tables and no invoice line items.
    enable_table_extractor: bool = True
    # False keeps each page table independent. True may join consecutive-page
    # tables only when their normalized headers and provider match.
    enable_table_stitching: bool = False

    # Docling is the single document engine (text layer, RapidOCR for scans, TableFormer): see requirements-docling.txt.
    # TableFormer mode: "fast" (~7 s/page on 6 CPU threads, measured) or "accurate" (~150 s/page on the same CPU).
    docling_table_mode: str = "fast"
    # Explicit thread count: Docling's default oversubscribes the CPU (measured 20x slower than 6 threads).
    docling_num_threads: int = 6
    # Directory with pre-downloaded Docling models (offline). Empty = Hugging Face cache.
    docling_artifacts_path: str = ""

    # Langkah 3: Validasi aritmetika
    # False = tidak menjalankan validasi aritmetika (perilaku lama)
    # True  = jalankan validasi dan turunkan confidence berdasarkan hasil
    enable_arithmetic_validation: bool = False
    # Additive "arithmetic_validation" evidence on /v1/extract/pipeline/full (row rules + subtotal/grand total). Never
    # changes an extracted value. The router blend above (structural score) stays off until measured.
    enable_arithmetic_evidence: bool = True

    # Toleransi relatif untuk cek qty * rate ≈ line_total  (1% default)
    arith_row_tolerance_relative: float = 0.01
    # Toleransi absolut (Rupiah) untuk pembulatan
    arith_row_tolerance_absolute_idr: float = 1.0

    # Candidate generation + LightGBM scoring on the extraction path (additive, see extraction/ml/service.py).
    # Rule-based extractors stay primary; an ML value only replaces a rule field that is unknown AND has the same name,
    # and only when the model is "accepted" (score >= t_auto of its bundle).
    enable_ml_extraction: bool = True
    # `pod_phase2` is an old, empty staging directory in this repository.
    # Point inference at the versioned bundle set that actually contains the
    # production INVOICE, POD, SURAT_JALAN and RATE_AGREEMENT field models.
    ml_model_set_dir: Path = PACKAGE_ROOT / "models" / "all_v3"

    # Layout zones for extractors, as FRACTIONS of the page (bbox_norm is 0..1). Legacy 0..1000 fixtures are
    # converted by ``to_unit_bbox`` before comparing. These replace pixel-like literals (T-13).
    invoice_header_max_y: float = 0.15  # vendor name is searched above this y
    invoice_summary_min_y: float = 0.25  # subtotal/tax/total keywords are searched below this y first
    invoice_table_min_y: float = 0.20  # AWB fallback scans the table band between these y values
    invoice_table_max_y: float = 0.70
    invoice_row_y_tolerance: float = 0.025  # lines within this vertical distance belong to the same row
    rate_agreement_header_max_y: float = 0.20


settings = Settings()


def candidates_config_dir() -> Path:
    """Directory of the candidate/table YAML files, resolved from settings (never from the CWD)."""
    return Path(settings.config_dir) / "candidates"
