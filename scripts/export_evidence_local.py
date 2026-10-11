import json
from pathlib import Path

from app.services.pipeline import Phase5DocumentPipeline


source = Path("reports/synthetic_3pl_bundle_4docs.pdf")
output = Path("reports/out/evidence/invoice1.json")
output.parent.mkdir(parents=True, exist_ok=True)

evidence = Phase5DocumentPipeline().process_document(
		source,
		document_id="invoice-001",
)
output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {output}")
