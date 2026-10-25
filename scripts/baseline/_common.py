"""Shared helpers for ad-hoc baseline scripts (NOT a replacement for the Fase 1 evaluator)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]  # <repo>/ai-service
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

DATASET_ROOT = Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset"))
INVOICE_DIR = DATASET_ROOT / "invoice"
INVOICE_PDF_DIR = INVOICE_DIR / "pdf"
CONFIG_DIR = AI_ROOT / "config" / "candidates"

# GT has ~43 line-item field names; extend after the Fase 1 inventory.
AMOUNT_KEYS = ("amount", "total", "line_total")


def pdf_key(name: str) -> str:
    """Lowercase and strip whitespace: 'invoice_8 _Variasi_2.pdf' -> 'invoice_8_variasi_2.pdf'."""
    return re.sub(r"\s+", "", name).lower()


def load_invoice_gt() -> dict[str, dict]:
    """Map normalized PDF file name -> GT document (SYNTHETIC GT, not human gold).

    GT entries named 'invoice_variasi_N.pdf' carry template_source 'contoh-invoice-7'; they are
    mapped to 'invoice_7_variasi_N.pdf'. This mapping is a HYPOTHESIS: verify it by comparing
    invoice_number ('0409015') with the PDF text before trusting it.
    """
    gt: dict[str, dict] = {}
    for gt_file in sorted(INVOICE_DIR.glob("ground_truth_variasi_*.json")):
        data = json.loads(gt_file.read_text(encoding="utf-8"))
        variation = data.get("variation")
        for doc in data["documents"]:
            key = pdf_key(doc["file_name"])
            if re.fullmatch(r"invoice_variasi_\d+\.pdf", key):
                match = re.search(r"(\d+)\s*$", doc.get("template_source", ""))
                if match:
                    key = f"invoice_{match.group(1)}_variasi_{variation}.pdf"
            gt[key] = doc
    return gt


def gt_amounts(doc: dict) -> list[str]:
    """One amount per GT line item (first positive numeric among AMOUNT_KEYS) as integer strings."""
    out: list[str] = []
    for item in doc.get("line_items", []):
        for key in AMOUNT_KEYS:
            value = item.get(key)
            if isinstance(value, (int, float)) and value > 0:
                out.append(str(int(value)))
                break
    return out


def digits(text: str | None) -> str:
    return re.sub(r"[^0-9]", "", text or "").lstrip("0")
