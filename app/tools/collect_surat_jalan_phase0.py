"""Collect OCR/rule/ML evidence for every local Full_Dataset Surat Jalan PDF."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import fitz

from app.core.config import settings
from app.services.extraction.label_value import merge_review_fields
from app.services.extraction.ml.service import run_ml_extraction
from app.services.extraction.surat_jalan import SuratJalanExtractor
from app.services.ocr import docling_engine
from app.services.ocr.docling_engine import DoclingOCREngine


class UnavailableDoclingRunner:
    """Use the active PyMuPDF text fallback when local Docling native import is unavailable."""
    def convert_page(self, **_kwargs):
        return None


def _pages(pdf_path: Path) -> list[dict[str, Any]]:
    original = fitz.open(pdf_path)
    result = []
    engine = DoclingOCREngine()
    for index, page in enumerate(original):
        one_page = fitz.open()
        one_page.insert_pdf(original, from_page=index, to_page=index)
        payload = one_page.tobytes()
        rect = page.rect
        result.append(engine.process(payload, is_pdf=True, page_number=index + 1, quality={},
                                     page_width=round(rect.width), page_height=round(rect.height)))
        one_page.close()
    original.close()
    return result


def _field(field) -> dict[str, Any]:
    return {"field": field.field, "value": field.value, "is_unknown": field.is_unknown,
            "confidence": field.confidence, "confidence_source": field.confidence_source,
            "review_status": field.review_status, "source_kind": field.source_kind,
            "raw_text": field.raw_text, "source": field.source.model_dump()}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    dataset = root / "reports" / "Full_Dataset" / "surat jalan" / "pdf"
    out_path = root / "docs" / "phase0" / "runs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docling_engine.get_runner = lambda: UnavailableDoclingRunner()
    runs = []
    for pdf_path in sorted(dataset.glob("*.pdf")):
        pages = _pages(pdf_path)
        evidence = {"pages": pages}
        doc_id = f"PHASE0-{pdf_path.stem}"
        rule_fields = SuratJalanExtractor().extract(doc_id, evidence)
        ml = run_ml_extraction(doc_id, "SURAT_JALAN", evidence)
        fields = merge_review_fields(doc_id, evidence, "SURAT_JALAN", rule_fields, ml)
        line_lengths = [len(line.get("text", "")) for page in pages for line in page.get("lines", [])]
        line_heights = [line["bbox_norm"][3] - line["bbox_norm"][1] for page in pages for line in page.get("lines", []) if line.get("bbox_norm")]
        runs.append({
            "source_file": pdf_path.name,
            "pages": [{"page_number": page["page_number"], "line_count": len(page["lines"]),
                       "longest_line_length": max((len(line["text"]) for line in page["lines"]), default=0),
                       "median_line_height": statistics.median([line["bbox_norm"][3] - line["bbox_norm"][1] for line in page["lines"] if line.get("bbox_norm")]) if page["lines"] else 0,
                       "engine_used": page["engine_used"], "line_split_count": page.get("line_split_count", 0),
                       "tables": [{"table_id": table.get("table_id"), "role": table.get("role"), "headers": table.get("headers"),
                                   "row_count": len(table.get("rows", [])), "selection_rank": None} for table in page.get("tables", [])]}
                      for page in pages],
            "ocr_summary": {"line_count": len(line_lengths), "longest_line_length": max(line_lengths, default=0),
                            "median_line_height": statistics.median(line_heights) if line_heights else 0},
            "rule_output": [_field(field) for field in rule_fields],
            "ml_output": {"status": ml.get("status"), "model_set": ml.get("model_set"),
                          "fields": [{"field": item.get("field"), "top_candidates": item.get("top_candidates", [])}
                                     for item in ml.get("fields", [])]},
            "review_output": [_field(field) for field in fields],
            "persisted_predictions": {"status": "not_read", "reason": "No local backend API is running; no remote database rows were queried or written."},
        })
    out_path.write_text(json.dumps({"model_set_dir": str(settings.ml_model_set_dir), "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} runs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
