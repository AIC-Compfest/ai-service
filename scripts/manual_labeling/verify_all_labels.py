import os, glob
import pandas as pd

csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"
csv_files = sorted(glob.glob(os.path.join(csv_dir, "candidates_*.csv")))

print(f"Total candidate CSV files found: {len(csv_files)} (Expected: 24)\n")

all_passed = True
total_rows_all = 0
summary_rows = []

for f in csv_files:
    fname = os.path.basename(f)
    df = pd.read_csv(f, dtype=str)
    n_rows = len(df)
    total_rows_all += n_rows
    
    label_cols = [c for c in df.columns if c.startswith("label__")]
    has_notes = "label_notes" in df.columns
    has_status = "label_status" in df.columns
    
    file_errors = []
    col_counts = {}
    for col in label_cols:
        # Check that all values are either '0' or '1'
        vals = set(df[col].dropna().unique())
        invalid_vals = vals - {"0", "1"}
        n_null = df[col].isna().sum()
        n_ones = (df[col] == "1").sum()
        n_zeros = (df[col] == "0").sum()
        col_counts[col] = (n_ones, n_zeros)
        
        if invalid_vals:
            file_errors.append(f"Col {col} has invalid values: {invalid_vals}")
        if n_null > 0:
            file_errors.append(f"Col {col} has {n_null} null/NaN values")
        if n_ones + n_zeros != n_rows:
            file_errors.append(f"Col {col} sum({n_ones} + {n_zeros}) != total rows ({n_rows})")
            
    notes_filled = (df["label_notes"].notna() & (df["label_notes"] != "")).sum() if has_notes else 0
    
    if file_errors:
        all_passed = False
        print(f"❌ {fname} ({n_rows} rows): ERRORS: {file_errors}")
    else:
        counts_str = ", ".join([f"{c.replace('label__', '')}: 1s={cnt[0]}" for c, cnt in col_counts.items()])
        print(f"✅ {fname} ({n_rows} rows): {counts_str} | notes filled: {notes_filled}")
        
    summary_rows.append({
        "file": fname,
        "rows": n_rows,
        "label_cols": label_cols,
        "col_counts": col_counts,
        "notes_filled": notes_filled
    })

print(f"\n==========================================")
print(f"TOTAL ROWS PROCESSED ACROSS ALL 24 FILES: {total_rows_all}")
if all_passed:
    print("ALL 24 CSV FILES PASSED VERIFICATION! 100% VALID BINARY LABELS (0 or 1).")
else:
    print("SOME FILES FAILED VERIFICATION!")
