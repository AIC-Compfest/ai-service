# Phase 2 Changelog

## 2026-10-12

- Added Phase 1 pipeline version stamps, strict registry validation, typed base/variant manifest generation, closed categorical vocab metadata, and leakage-safe training-column assembly.
- Added `TruthRecord`, `FieldLabel`, status/review enums, atomic `LabelStore`, audit log, deterministic grounding, stale-version export protection, and candidate-label CSV export.
- Added local FakeLLM/truth import, POD dataset preparation from OCR evidence, template profiles, anchor/context/position transfer, and labeling CLI stages.
- Added grouped template cross-validation, LightGBM S1 binary training, OOF predictions, thresholds, hash-verified bundles, categorical vocab persistence, and ML decoding/inference adapters.
- Added Phase 2 tests and documentation.
- Verified the supplied 17-template/34-PDF POD set through OCR preparation, candidate generation, truth grounding, 17 template profiles, variant transfer, export, model training, and inference.

## Limitations

The supplied local truth import is a deterministic smoke path, not an LLM provider implementation. Production review and provider adapters, S2/S3/L1/L2 strategy comparison, calibration, checkpointing, full line-item labeling, and API routing remain to be completed.