"""Field-level regression check for InvoiceExtractor on cached evidence (synthetic GT, not human gold).

    .\\.venv\\Scripts\\python.exe scripts\\experiments\\invoice_fields_check.py [--evidence out/cache/invoice_evidence] [--tables strip]

Used in Fase 2 to make sure the unit fix in invoice.py does not reduce header-field accuracy.
``--tables strip`` removes pages[].tables so the line-item fallback path (AWB scan) is exercised.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(AI_ROOT / "scripts" / "baseline"))

from _common import load_invoice_gt  # noqa: E402

from app.services.extraction.invoice import InvoiceExtractor  # noqa: E402


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").casefold())


def _num(value):
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=AI_ROOT / "out" / "cache" / "invoice_evidence")
    parser.add_argument("--tables", choices=("keep", "strip"), default="keep")
    args = parser.parse_args()
    gt = load_invoice_gt()
    extractor = InvoiceExtractor()
    tally = {"docs": 0, "invoice_number": 0, "vendor_name": 0, "subtotal": 0, "grand_total": 0, "line_items_docs": 0, "line_items_rows": 0, "gt_rows": 0}
    for path in sorted(args.evidence.glob("*.json")):
        key = path.stem if path.stem.endswith(".pdf") else path.name[: -len(".json")]
        document = gt[key]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if args.tables == "strip":
            for page in evidence["pages"]:
                page["tables"] = []
        fields = {f.field: f for f in extractor.extract("x", evidence)}
        tally["docs"] += 1
        tally["invoice_number"] += _norm(document.get("invoice_number")) in _norm(fields["invoice_number"].value) and bool(fields["invoice_number"].value)
        seller = (document.get("seller") or {}).get("name") or ""
        got = _norm(fields["vendor_name"].value)
        tally["vendor_name"] += bool(got) and bool(seller) and (_norm(seller) in got or got in _norm(seller))
        totals = document.get("totals") or {}
        expected_subtotal = _num(totals.get("subtotal") or totals.get("total_charges"))
        expected_total = _num(totals.get("grand_total") or totals.get("total_due") or totals.get("total"))
        tally["subtotal"] += expected_subtotal is not None and _num(fields["subtotal"].value) == expected_subtotal
        tally["grand_total"] += expected_total is not None and _num(fields["grand_total"].value) == expected_total
        items = fields["line_items"].value or []
        tally["line_items_docs"] += bool(items)
        tally["line_items_rows"] += len(items)
        tally["gt_rows"] += len(document.get("line_items", []))
    print(json.dumps(tally))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
