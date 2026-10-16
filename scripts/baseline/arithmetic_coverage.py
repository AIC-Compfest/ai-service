"""How much of the invoice set the arithmetic registry actually covers (Fase 5).

Run from ai-service:  .\\.venv\\Scripts\\python.exe scripts\\baseline\\arithmetic_coverage.py
Per document: tables, tables with an applicable rule, rows evaluated, row PASS/FAIL, subtotal and grand-total checks.
It also checks the registry itself on the GROUND TRUTH line items (does `quantity * unit_price = amount` hold in the data?).
Coverage numbers are not a quality metric for table extraction; a FAIL can be a wrong cell or a wrong rule.
"""
import json
from decimal import Decimal

from _common import AI_ROOT, INVOICE_PDF_DIR, load_invoice_gt, pdf_key

from app.services.extraction.invoice import InvoiceExtractor
from app.services.pipeline import Phase5DocumentPipeline
from app.services.validation.document import validate_extraction

gt = load_invoice_gt()
pipeline, extractor = Phase5DocumentPipeline(), InvoiceExtractor()
docs = []
for pdf in sorted(INVOICE_PDF_DIR.glob("*.pdf")):
    key = pdf_key(pdf.name)
    if key not in gt:
        continue
    evidence = pipeline.process_document(str(pdf), filename=pdf.name, document_id=f"arith_{key}")
    fields = extractor.extract("x", evidence)
    result = validate_extraction(fields, evidence["pages"])
    totals = result.get("totals", {})
    checks = {c["field"]: c["status"] for c in totals.get("checks", [])}
    docs.append({
        "file": key, "status": result["status"], **result.get("coverage", {}),
        "rows_pass": sum(t["passed"] for t in result.get("tables", [])), "rows_evaluated_checks": sum(t["evaluated"] for t in result.get("tables", [])),
        "rows_fail": result.get("failed_rows_total", 0), "subtotal": checks.get("subtotal", "-"), "grand_total": checks.get("grand_total", "-"),
    })

print("documents", len(docs), "| with any rule applicable:", sum(1 for d in docs if d.get("tables_with_rules", 0) > 0))
print("rows total", sum(d.get("rows_total", 0) for d in docs), "| rows evaluated", sum(d.get("rows_evaluated", 0) for d in docs),
      "| checks PASS", sum(d["rows_pass"] for d in docs), "of", sum(d["rows_evaluated_checks"] for d in docs))
for name in ("subtotal", "grand_total"):
    values = [d[name] for d in docs]
    print(name, {v: values.count(v) for v in sorted(set(values))})

# registry sanity on the ground truth itself
valid = total = 0
for doc in gt.values():
    for item in doc["line_items"]:
        q, p, a = item.get("quantity"), item.get("unit_price"), item.get("amount")
        if None not in (q, p, a):
            total += 1
            valid += abs(Decimal(str(q)) * Decimal(str(p)) - Decimal(str(a))) <= 1
print("GT: quantity * unit_price = amount holds in", valid, "of", total, "items")
out_dir = AI_ROOT / "out" / "baseline"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "arithmetic_coverage.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")
