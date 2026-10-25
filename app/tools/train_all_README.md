## Prerequisites
read `make_candidate_labels_README.md`. Or you have run `labeling.py` if candidates were not labeled manually

## Run it
```powershell
conda run -n aic_compfest python -m app.tools.train_all `
  --candidate-dir data\candidates_v3 --out models\v3
```
Run it from the repo root. Add `--dry-run` first to see what it will do without training.

| Option                                            | Default                         | What it does                                                                                                                                               |
| ------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--candidate-dir`                                 | required                        | Folder with the candidate CSVs, `feature_manifest.json`, `training_policy.yaml`, `training_schema.json`                                                    |
| `--out`                                           | `models`                        | Where model bundles and reports are written                                                                                                                |
| `--config`                                        | `config/training/training.yaml` | Base training config                                                                                                                                       |
| `--doc-types INVOICE POD ...`                     | all                             | Train only these document types                                                                                                                            |
| `--fields INVOICE.grand_total pihak_pertama_nama` | all                             | Train only these fields (`DOC.field` or bare name)                                                                                                         |
| `--set KEY=VALUE`                                 | none                            | Override config for this run, e.g. `--set lgbm_defaults.n_estimators=200`. Repeatable. The effective config is saved to `<out>/training_config_used.yaml`. |
| `--workers N`                                     | 1                               | Train N fields in parallel processes                                                                                                                       |
| `--skip-existing`                                 | off                             | Keep bundles already in `--out` instead of retraining                                                                                                      |
| `--fail-fast`                                     | off                             | Stop after the first failed field                                                                                                                          |
| `--dry-run`                                       | off                             | Print the preflight table and exit                                                                                                                         |
| `--no-progress`                                   | off                             | No bar; log and summary are still written                                                                                                                  |
| `--field-source`                                  | `auto`                          | `schema` uses `training_schema.json`, `registry` uses `field_registry.yaml`. `auto` picks the schema if it's in the candidate folder.                      |
| `--registry`, `--log`                             | repo defaults                   | Field registry path and log file path                                                                                                                      |