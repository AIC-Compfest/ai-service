import json, os, glob, re
import pandas as pd

# Load ground truths
gt_base = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset"
with open(os.path.join(gt_base, "invoice", "ground_truth_variasi_1.json"), encoding="utf-8") as f:
    inv1 = json.load(f)
with open(os.path.join(gt_base, "invoice", "ground_truth_variasi_2.json"), encoding="utf-8") as f:
    inv2 = json.load(f)
with open(os.path.join(gt_base, "pod", "ground_truth_all_17_documents.json"), encoding="utf-8") as f:
    pod = json.load(f)
with open(os.path.join(gt_base, "surat jalan", "ground_truth_surat_jalan.json"), encoding="utf-8") as f:
    sj = json.load(f)
with open(os.path.join(gt_base, "rate card", "ground_truth_rate_cards_1_7.json"), encoding="utf-8") as f:
    rc1_7 = json.load(f)
with open(os.path.join(gt_base, "rate card", "ground_truth_rate_card_8_15.json"), encoding="utf-8") as f:
    rc8_15 = json.load(f)

csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

for doc_type in ["INVOICE", "POD", "RATE_AGREEMENT", "SURAT_JALAN"]:
    sample_csv = glob.glob(os.path.join(csv_dir, f"candidates_{doc_type}_*.csv"))[0]
    df = pd.read_csv(sample_csv, dtype=str)
    unique_docs = sorted(df["m_document_id"].unique())
    print(f"\n{doc_type}: {len(unique_docs)} unique documents. Samples:")
    print("  ", unique_docs[:5])
