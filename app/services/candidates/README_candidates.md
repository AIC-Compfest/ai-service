# Candidate Dataset Builder

This module creates deterministic, label-ready CSVs from OCR evidence. It is a training-data tool and is intentionally parallel to the legacy extractors; it does not replace or change the runtime extraction API.

## Processing order

For a real PDF or image, use this order:

```text
PDF/image -> ingestion/preprocessing/OCR -> evidence JSON -> candidate generation -> CSV labeling
```

The candidate module does not accept a PDF directly. It needs an evidence JSON object containing `pages`, `lines`, OCR tokens, token confidence, and `bbox_norm` values. The existing `Phase5DocumentPipeline` produces that object.

You can skip the OCR step when an evidence JSON file already exists, for example from a previous Phase 5 run or a hand-built fixture.

## Environment setup

Run commands from the repository root. PowerShell activation can be unreliable on some machines, so `conda run` is the most portable form:

```powershell
conda run -n aic_compfest python --version
conda run -n aic_compfest pytest -q tests/unit/test_candidate_generation.py
```

When activation works, this is equivalent:

```powershell
conda activate aic_compfest
pytest -q tests/unit/test_candidate_generation.py
```

The test file covers Indonesian number/date parsing and the worked invoice candidate contract. To run the complete repository suite:

```powershell
conda run -n aic_compfest pytest -q
```

## Step 1: Produce OCR evidence from a real document

Create a temporary script such as `scripts/export_evidence_local.py`:

```python
import json
from pathlib import Path

from app.services.pipeline import Phase5DocumentPipeline


source = Path("data/invoice.pdf")
output = Path("out/evidence/invoice.json")
output.parent.mkdir(parents=True, exist_ok=True)

evidence = Phase5DocumentPipeline().process_document(
		source,
		document_id="invoice-001",
)
output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {output}")
```

Run it from the repository root:

```powershell
conda run -n aic_compfest python scripts/export_evidence_local.py
```

The Phase 5 pipeline performs document ingestion, image quality preprocessing, vector-PDF extraction or EasyOCR fallback, page classification, segmentation, layout evidence, and signature detection. Candidate generation only consumes the resulting `pages` section; classification and signature results are not used to invent candidates.

For a scanned document, EasyOCR may initialize and require its local model assets. No candidate-generation code makes network calls. For a digital PDF, the pipeline normally uses the PyMuPDF vector path when enough text objects are available.

## Step 2: Create a manifest

Create `data/candidate_manifest.csv`:

```csv
document_id,evidence_json_path,doc_type,vendor_group,template_group,source_file
invoice-001,out/evidence/invoice.json,INVOICE,UNKNOWN,UNKNOWN,data/invoice.pdf
```

Required columns are `document_id`, `evidence_json_path`, and `doc_type`. Optional `page_start` and `page_end` columns select a logical document segment from a shared multi-page evidence file. `vendor_group` and `template_group` propagate to candidate rows and are reserved for later GroupKFold splits. `doc_type` must be one of `INVOICE`, `SURAT_JALAN`, `POD`, or `RATE_AGREEMENT` and should come from the dataset manifest, not from a new classifier call.

The evidence file may contain the complete Phase 5 result (`{"pages": [...]}` plus other keys). A minimal evidence file containing only `pages` is also valid.

## Step 3: Build candidate CSVs

```powershell
conda run -n aic_compfest python -m app.tools.build_candidate_dataset `
	--manifest data/candidate_manifest.csv `
	--out out/candidates/invoice-001 `
	--config config/candidates
```

Use `--limit 1` to test only the first manifest row:

```powershell
conda run -n aic_compfest python -m app.tools.build_candidate_dataset `
	--manifest data/candidate_manifest.csv `
	--out out/candidates/smoke `
	--config config/candidates `
	--limit 1
```

Relative paths in the manifest are resolved from the current working directory, so run the command from the repository root.

## One PDF containing four logical documents

Run Phase 5 once for the physical PDF. It produces one evidence JSON with all pages and a `segments` array. Inspect the segment boundaries:

```powershell
conda run -n aic_compfest python -c "import json; e=json.load(open('out/evidence/bundle_4docs.json', encoding='utf-8')); print(*[(s['segment_number'], s['document_type'], s['start_page'], s['end_page']) for s in e['segments']], sep='\n')"
```

If the output is, for example, Invoice pages 1-2, Surat Jalan pages 3-4, POD page 5, and Rate Agreement pages 6-7, make four manifest rows with distinct logical IDs and page ranges:

```csv
document_id,evidence_json_path,doc_type,vendor_group,template_group,page_start,page_end,source_file
bundle-001-invoice,out/evidence/bundle_4docs.json,INVOICE,UNKNOWN,UNKNOWN,1,2,reports/synthetic_3pl_bundle_4docs.pdf
bundle-001-surat-jalan,out/evidence/bundle_4docs.json,SURAT_JALAN,UNKNOWN,UNKNOWN,3,4,reports/synthetic_3pl_bundle_4docs.pdf
bundle-001-pod,out/evidence/bundle_4docs.json,POD,UNKNOWN,UNKNOWN,5,5,reports/synthetic_3pl_bundle_4docs.pdf
bundle-001-rate-agreement,out/evidence/bundle_4docs.json,RATE_AGREEMENT,UNKNOWN,UNKNOWN,6,7,reports/synthetic_3pl_bundle_4docs.pdf
```

Then run the normal CLI command. The page filter is applied before candidate generation, while original OCR page numbers remain in each row. Do not use the same `document_id` for all four rows: it identifies the logical training document, not only the physical PDF.

## What it generates

The output directory contains one file for each candidate type actually enabled by the field registry, plus a report:

```text
out/candidates/invoice-001/
	candidates_INVOICE_amount.csv
	candidates_INVOICE_date.csv
	candidates_INVOICE_enum.csv
	candidates_INVOICE_id.csv
	candidates_INVOICE_party.csv
	run_report.json
```

The exact set depends on `config/candidates/field_registry.yaml`. For the current baseline registry, the implemented header candidate types are:

| Candidate type | Examples | Main parsed value |
| --- | --- | --- |
| `amount` | `Rp1.250.000`, `137.500` | Decimal string |
| `date` | `05 Juli 2026` | ISO date |
| `time` | `14:30 WIB` | `HH:MM[:SS]` |
| `id` | `INV-2026-000123` | Literal OCR text |
| `party` | `PT Maju Jaya` | Normalized text |
| `enum` | `Rp`, `IDR`, `USD` | Currency code such as `IDR` |

Every candidate row includes the literal OCR span, page and normalized bounding box, parsed value, deterministic candidate ID, OCR quality features, geometry, duplicate/rank features, lexical `kw_*` features, and `ctx_left`, `ctx_right`, `ctx_above`, `ctx_below`, `ctx_line`, and `ctx_sent` context columns. `label__<field>`, `label_status`, and `label_notes` are intentionally empty for manual labeling.

`run_report.json` contains document/candidate counts, empty pages, cap overflows, and generation timings. It is useful as a quick smoke check:

```powershell
Get-Content out/candidates/invoice-001/run_report.json
```

## Direct Python usage

The CLI is the normal entry point, but the generator can be called directly when another tool already has evidence in memory:

```python
from app.services.candidates.pipeline import CandidatePipeline

pipeline = CandidatePipeline("config/candidates")
rows = pipeline.generate(
		document_id="invoice-001",
		doc_type="INVOICE",
		ocr_result=evidence,
		run_id="local-smoke",
)

for row in rows:
		print(row["m_cand_type"], row["m_raw_text"], row["m_parsed_value"])
```

## Configuration

`config/candidates/field_registry.yaml` is the source of truth for which candidate types are enabled for each document type. Add or remove header fields there rather than hardcoding field names in Python. `candidate_generation.yaml` contains per-page caps.

Candidate spans are copied from OCR tokens and use normalized `[0, 1]` coordinates. The input OCR contract uses `[0, 1000]` coordinates; the candidate page normalizer converts them.

## Current implementation scope

Implemented now: registry loading; OCR page/token normalization; amount, date, time, ID, party, enum, quantity, weight, city, address, percent, days, and text generation; fuzzy header matching; token-grid tables; line-item CSVs; deterministic IDs; lexical/type-specific features; training policy/schema output; cap/empty-page/timing reports; and focused tests.

The output also includes `li_columns_<DOC>.csv`, `li_cells_<DOC>.csv`, `li_rows_<DOC>.csv`, `li_preview.jsonl`, `doc_field_status_template.csv`, `feature_manifest.json`, `training_schema.json`, and a run-local `training_policy.yaml`. Raw values, parsed values, context text, metadata, and labels are tagged for dropping by the training policy.

Remaining acceptance work is tracked in [VERIFICATION_REPORT.md](../../../VERIFICATION_REPORT.md): full table/matrix behavior, exact feature vectors, formal CSV validation, ground-truth recall, performance, and no-network tests. The modified field list was not supplied, so the registry represents a baseline subset. CV signature fields remain outside candidate generation by design.