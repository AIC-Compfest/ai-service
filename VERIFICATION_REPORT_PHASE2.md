# Verification Report - Phase 2 - 2026-10-12

## Summary

The Phase 2 core workflow is implemented and exercised locally. The supplied POD dataset was prepared into 34 OCR evidence files, candidates were generated, base truth was grounded, 17 base template profiles were built, variant labels were transferred, labels were exported, a POD LightGBM bundle was trained, and inference loaded the bundle successfully.

## Requirements

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Phase 1 version/registry prerequisites | PASS | Candidate rows and `run_report.json` | Pipeline version and strict registry validation are active. |
| Base/variant manifest | PASS | `data/pod_phase2/manifest.csv` | 34 rows, `is_base`, `base_document_id`, and `template_group`. |
| Label store | PASS | `tests/unit/test_labeling_phase2.py` | Atomic upsert, idempotency, audit log. |
| Grounding | PASS | `tests/unit/test_labeling_phase2.py` | Exact, fuzzy, missed, and duplicate behavior covered. |
| Template transfer | PARTIAL | `data/pod_labels/templates` | Core transfer works; adaptive verification and line-item transfer remain. |
| Label export | PASS | `data/pod_phase2/candidates/candidate_labels_*.csv` | Labels remain separate from feature CSVs. |
| Grouped CV | PASS | `tests/unit/test_training_phase2.py` | Template groups are not split across folds. |
| LightGBM bundle | PASS | `models/pod_phase2/POD.pihak_pertama_nama.binary.v1` | Bundle saves and loads with hash verification. |
| Calibration/threshold search | PARTIAL | Bundle thresholds | Default thresholds are stored; OOF calibration search remains. |
| ML inference/decode | PASS | `tests/unit/test_ml_inference_phase2.py` and POD smoke run | Abstention and categorical encoding work. |
| Real LLM egress/review UI | UNVERIFIED | Local FakeLLM only | No provider API key was supplied; no network call was made. |
| Full strategy matrix/S2/S3/L1/L2 | UNVERIFIED | — | Only deterministic S1 binary training is implemented. |
| Full line-item labeling/training | UNVERIFIED | — | Phase 1 table outputs exist; Phase 2 label transfer is header-focused. |

## Test results

```text
Phase 1 candidate tests: 12 passed
Phase 2 tests: 5 passed
Full repository suite: 110 passed, 4 skipped, 1 existing fixture failure
```

The existing failure is `TestPhase5Components.test_document_ingestion_jpg`: the repository's selected `sample_pod` file is detected as `image/png`, while the legacy test expects `image/jpeg`. It is unrelated to labeling/training changes.

## Known training result

The representative `POD.pihak_pertama_nama.binary.v1` smoke model trained and loaded, but its OOF threshold report had zero accepted coverage. This is expected for the current small/partially grounded local smoke labels and is not reported as model quality success.

## Risks and next steps

Human audit is still required before production training. Provider adapters, review UI, multiclass/ranking strategies, calibration, line-item labels, checkpointed search, and FastAPI route integration remain explicit follow-up work.