# Labeling, Training, and ML Inference

Phase 2 consumes Phase 1 candidate CSVs and OCR evidence. It keeps feature CSVs immutable and stores labels separately in long-format CSV/JSONL files.

## Environment

```powershell
conda run -n aic_compfest python --version
conda run -n aic_compfest pytest -q tests/unit/test_labeling_phase2.py tests/unit/test_training_phase2.py tests/unit/test_ml_inference_phase2.py
```

Required Phase 2 packages are declared in `requirements.txt`: LightGBM, pandas, and optional Streamlit review support.

## Prepare the supplied POD data

The supplied data contains 17 base templates and two variations per template. The truth uses BAST-style fields, so those fields are included in the POD registry.

```powershell
conda run -n aic_compfest python -m app.tools.labeling prepare `
  --source-dir reports/POD_Training `
  --out data/pod_phase2 `
  --config config/candidates
```

This runs the existing Phase 5 ingestion/OCR/layout pipeline once per PDF and writes:

- `data/pod_phase2/evidence/*.json`
- `data/pod_phase2/manifest.csv`

The manifest contains `is_base`, `base_document_id`, `template_group`, `page_start`, and `page_end`. The first variation is treated as the base for each template. Review this assignment before production labeling.

## Generate candidates

```powershell
conda run -n aic_compfest python -m app.tools.build_candidate_dataset `
  --manifest data/pod_phase2/manifest.csv `
  --out data/pod_phase2/candidates `
  --config config/candidates
```

Every candidate row contains `m_pipeline_version`. Labels are refused during export if their pipeline version is stale.

## Truth and review

For a real LLM workflow, `llm-base` should call a configured provider and write validated truth JSON for base documents. The current local path imports the supplied ground truth without network access:

```powershell
conda run -n aic_compfest python -m app.tools.labeling llm-base `
  --ground-truth reports/POD_Training/ground_truth_all_17_documents.json `
  --out data/pod_labels
```

Ground and store labels separately:

```powershell
conda run -n aic_compfest python -m app.tools.labeling ground `
  --truth-dir data/pod_labels/truth_llm `
  --candidate-dir data/pod_phase2/candidates `
  --labels data/pod_labels `
  --config config/candidates
```

All new labels start as `pending_review`. For a controlled local smoke test only, promote the fake/local labels:

```powershell
conda run -n aic_compfest python -m app.tools.labeling accept-local --labels data/pod_labels
```

Do not use `accept-local` for production labels; human audit states are required by the training policy.

Build base profiles and transfer labels to variants:

```powershell
conda run -n aic_compfest python -m app.tools.labeling build-templates `
  --candidate-dir data/pod_phase2/candidates `
  --labels data/pod_labels `
  --out data/pod_labels/templates

conda run -n aic_compfest python -m app.tools.labeling transfer `
  --candidate-dir data/pod_phase2/candidates `
  --labels data/pod_labels `
  --templates data/pod_labels/templates `
  --manifest data/pod_phase2/manifest.csv
```

## Export labels

```powershell
conda run -n aic_compfest python -m app.tools.labeling export `
  --candidate-dir data/pod_phase2/candidates `
  --labels data/pod_labels
```

This writes `candidate_labels_*` files and `label_report.json`. Feature CSVs remain unchanged.

## Train one field model

```powershell
conda run -n aic_compfest python -m app.tools.training run `
  --candidate-dir data/pod_phase2/candidates `
  --out models/pod_phase2 `
  --doc-type POD `
  --field pihak_pertama_nama
```

Training uses only eligible `auto_accepted`, `human_confirmed`, and `human_corrected` labels. `pending_review` and `unreadable` rows are excluded. Folds are grouped by `template_group`, never by random row.

The bundle contains a hash-verified `model.joblib`, `bundle.json`, and OOF predictions. Categorical vocabularies are stored in the bundle for deterministic inference.

## Inference

```python
import json
from app.services.extraction.ml.predictor import MLFieldExtractor

evidence = json.loads(open("data/pod_phase2/evidence/pod-1-v1.json", encoding="utf-8").read())
extractor = MLFieldExtractor("models/pod_phase2")
result = extractor.extract("pod-1-v1", "POD", evidence)
fields = result.to_extracted_fields()
```

Scores below `t_unknown` abstain with `value=None` and `is_unknown=True`. Scores between `t_unknown` and `t_auto` remain reviewable.

## Current Phase 2 scope

Implemented: local truth import/FakeLLM, atomic label store and audit log, deterministic grounding, base template profiles, variant transfer, label export, grouped LightGBM S1 training, bundle hash verification, categorical encoding, deterministic decode, and POD end-to-end smoke data.

Still required before production: real provider adapters and image prompt orchestration, Streamlit review screens, full multiclass/LambdaRank strategy comparison, calibration/threshold search from OOF, line-item label transfer/export, checkpoint/resume search, and API route wiring. These are recorded in [VERIFICATION_REPORT_PHASE2.md](VERIFICATION_REPORT_PHASE2.md).