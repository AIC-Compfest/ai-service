# Quick Fase 3 loop: predict + evaluate (dev+val, never holdout) for one provider config, plus constructed fixtures.
param([string]$Config = "token_grid_none", [string]$Tag = "p3")
$p = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "out\reports\$Tag" | Out-Null
& $p scripts\run_table_predictions.py --config $Config --splits dev,val --out "out\pred\$Tag\$Config" 2>&1 | Select-Object -Last 1
& $p scripts\eval_tables.py --gold-dir gold\regression --pred-dir "out\pred\$Tag\$Config" --manifest gold\manifest.json --splits dev,val --row-content-dir gold\row_content --allow-unreviewed --report "out\reports\$Tag\$Config.json" | Out-Null
foreach ($m in "vector", "phrase") {
  & $p scripts\run_table_predictions.py --config $Config --constructed $m --out "out\pred_fx\$Tag\${m}_$Config" 2>&1 | Out-Null
  & $p scripts\eval_tables.py --gold-dir gold\constructed --pred-dir "out\pred_fx\$Tag\${m}_$Config" --allow-unreviewed --report "out\reports\$Tag\fx_${m}_$Config.json" | Out-Null
}
$r = "out\reports\$Tag\"
& $p scripts\compare_reports.py --group all --label "real:$Config=${r}$Config.json" --label "fx_vector:$Config=${r}fx_vector_$Config.json" --label "fx_phrase:$Config=${r}fx_phrase_$Config.json"
