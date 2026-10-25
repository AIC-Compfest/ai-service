## How to run make candidate labels
### Run it

Run from the repo root:

powershell

```powershell
# 1. Build the label files from your manual labels
conda run -n aic_compfest python -m app.tools.make_candidate_labels `
  --candidate-dir data\candidates_v3

# 2. Check that every field is ready to train
conda run -n aic_compfest python -m app.tools.train_all `
  --candidate-dir data\candidates_v3 --out models\v3 --dry-run

# 3. Train
conda run -n aic_compfest python -m app.tools.train_all `
  --candidate-dir data\candidates_v3 --out models\v3
```

Step 2 should show `ok` for all 33 fields. Step 3 trains them.

### Options

|Option|Default|Purpose|
|---|---|---|
|`--candidate-dir`|required|Folder with your `candidates_*.csv` files. The label files are written here too, since the trainer reads them from this folder.|
|`--review-state`|`auto_accepted`|Review state given to labeled cells. You can also use `human_confirmed` or `human_corrected`.|
|`--label-source`|`manual`|Written to `label_source__<field>`.|
|`--report`|`<candidate-dir>/label_report.json`|Where the report goes.|
|`--overwrite`|off|Without this, the script refuses to overwrite existing `candidate_labels_*.csv` files.|
|`--dry-run`|off|Validates and prints per-field counts without writing anything.|

### What it does

- Reads each `candidates_<DOC>_<type>.csv` and writes `candidate_labels_<DOC>_<type>.csv`. Each output keeps the original columns and adds `label_state__`, `label_review__` and `label_source__` for each field.
- `1` becomes `present` and `0` becomes `absent` in `label_state__`. Your manual labels don't record "missed", so the script can't tell those cases apart.
- Blank labels stay blank, with an empty review state, so they can't be trained as negatives.
- Your original `candidates_*.csv` files are never modified.
- Before writing anything, it stops if a label cell has a value other than 0, 1 or blank. It also stops if the same field is labeled in two files of the same document type, since the trainer reads only one file per document type.