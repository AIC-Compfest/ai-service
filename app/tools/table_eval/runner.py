"""Run production table components on PDFs/fixtures and emit prediction evidence for the evaluator.

Configurations:
  token_grid_none / token_grid_typed   TokenGridTableProvider, doc_type None / known
  pymupdf                              PyMuPDFTableProvider
  router_none / router_typed           TableExtractor over the configured provider order
  api_flag_off / api_flag_on           full Phase5DocumentPipeline with ENABLE_TABLE_EXTRACTOR off / on
                                       (what the HTTP API actually returns in ``pages[].tables``)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import yaml

from app.core.config import settings

CONFIGS = ("token_grid_none", "token_grid_typed", "pymupdf", "router_none", "router_typed", "api_flag_off", "api_flag_on")
AI_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = AI_ROOT / "config" / "candidates"

_cache: dict[str, Any] = {}


def _providers():
    from app.services.candidates.tables import PyMuPDFTableProvider, TableExtractor, TokenGridTableProvider

    if "grid" not in _cache:
        _cache["grid"] = TokenGridTableProvider(CONFIG_DIR / "line_item_aliases.yaml", CONFIG_DIR / "abbreviations.yaml")
        _cache["mupdf"] = PyMuPDFTableProvider(yaml.safe_load((CONFIG_DIR / "table_resolution.yaml").read_text(encoding="utf-8")).get("pymupdf"))
        _cache["router"] = TableExtractor.from_config(CONFIG_DIR, settings.table_provider_order)
    return _cache


def tables_for_page(config: str, page, doc_type: str | None, document_id: str) -> list[dict]:
    """Evidence dicts (``table_evidence``) for one candidate ``Page`` under a provider configuration."""
    from app.services.candidates.pipeline import table_evidence

    providers = _providers()
    typed = doc_type if config.endswith("_typed") else None
    if config.startswith("token_grid"):
        tables = providers["grid"].tables(page, typed, document_id)
    elif config == "pymupdf":
        tables = providers["mupdf"].tables(page, doc_type, document_id)
    elif config.startswith("router"):
        tables = providers["router"].extract(page, typed, document_id)
    else:
        raise ValueError(f"not a provider configuration: {config}")
    return [table_evidence(table) for table in tables]


def predict_pdf(pdf_path: Path, config: str, doc_type: str | None, document_id: str) -> dict:
    """Prediction evidence for a real PDF using the OCR engine's own vector word tokens."""
    import cv2
    import numpy as np

    from app.services.candidates.context import build_page
    from app.services.ocr.engine import OCREngine
    from app.services.preprocessing.image_preprocessor import ImagePreprocessor

    if config.startswith("api_"):
        from app.services.pipeline import Phase5DocumentPipeline

        previous = settings.enable_table_extractor
        settings.enable_table_extractor = config == "api_flag_on"
        try:
            pipeline = _cache.setdefault("pipeline", Phase5DocumentPipeline())
            evidence = pipeline.process_document(str(pdf_path), filename=Path(pdf_path).name, document_id=f"pred_{config}_{document_id.replace('/', '_')}")
        finally:
            settings.enable_table_extractor = previous
        return {"document_id": document_id, "config": config, "pages": [{"page_number": p["page_number"], "tables": p.get("tables", [])} for p in evidence["pages"]]}

    engine = _cache.setdefault("engine", OCREngine())
    document = pymupdf.open(pdf_path)
    pages = []
    for number, page in enumerate(document, start=1):
        png = page.get_pixmap(dpi=100).tobytes("png")
        quality = ImagePreprocessor.estimate_quality(cv2.imdecode(np.frombuffer(png, np.uint8), 1))
        evidence = engine.process_pdf_page(page, number, quality, canonical_image_bytes=png).to_dict()
        candidate_page = build_page({"page_number": number, "lines": evidence["lines"], "_source_page": page})
        pages.append({"page_number": number, "tables": tables_for_page(config, candidate_page, doc_type, document_id)})
    return {"document_id": document_id, "config": config, "pages": pages}


def predict_fixture(pdf_bytes: bytes, config: str, mode: str, document_id: str, seed: int = 0) -> dict:
    """Prediction on a constructed fixture. ``vector`` keeps the live PDF page; ``phrase`` simulates raster OCR."""
    from app.services.candidates.context import build_page
    from app.tools.table_eval.synthetic import to_evidence_page

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for number, page in enumerate(document, start=1):
        evidence = to_evidence_page(page, number, mode, seed)
        evidence["_source_page"] = page if mode == "vector" else None
        pages.append({"page_number": number, "tables": tables_for_page(config, build_page(evidence), None, document_id)})
    return {"document_id": document_id, "config": config, "mode": mode, "pages": pages}
