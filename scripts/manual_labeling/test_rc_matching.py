import json, os, re
import pandas as pd

gt_base = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset"
csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

with open(os.path.join(gt_base, "rate card", "ground_truth_rate_cards_1_7.json"), encoding="utf-8") as f:
    rc1_7 = json.load(f)
with open(os.path.join(gt_base, "rate card", "ground_truth_rate_card_8_15.json"), encoding="utf-8") as f:
    rc8_15 = json.load(f)["documents"]

rc_gt = {}
for k, v in rc1_7.items():
    for vk, vd in v.items():
        rc_gt[f"{k}_{vk}".lower()] = vd
for d in rc8_15:
    rc_gt[d["file_name"].replace(".pdf", "").lower()] = d["fields"]

def clean_alphanum(s):
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()

def clean_amount(s):
    if not s or pd.isna(s):
        return ""
    # remove currency symbols, dots, commas, slashes, text
    # e.g. "Rp. 12.500.000" -> 12500000
    # "3.500 / KG" -> 3500
    # "3635.0" -> 3635
    s = str(s).split("/")[0].split("-")[0].strip()
    digits = re.findall(r"\d+", s)
    if not digits:
        return ""
    # if decimal point: e.g. "3635.0"
    if "." in s and len(digits) == 2 and len(digits[1]) in [1, 2]:
        return digits[0]
    return "".join(digits)

print("=== 1. Testing RATE AGREEMENT ID ===")
df_id = pd.read_csv(os.path.join(csv_dir, "candidates_RATE_AGREEMENT_id.csv"), dtype=str)
matched_id = 0
for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    # Loose id keys
    id_candidates = [
        ("load_number", gt.get("load_number")),
        ("load_id", gt.get("load_id")),
        ("nomor", gt.get("nomor")),
        ("work_order", gt.get("work_order")),
        ("order_number", gt.get("order_number")),
        ("references", gt.get("references")),
    ]
    found = False
    for k, val in id_candidates:
        if not val: continue
        val_clean = clean_alphanum(val)
        m = grp[grp["m_parsed_value"].apply(clean_alphanum) == val_clean]
        if len(m) > 0:
            matched_id += 1
            found = True
            break
        m_sub = grp[grp["m_parsed_value"].apply(clean_alphanum).apply(lambda x: len(x) >= 5 and (x in val_clean or val_clean in x))]
        if len(m_sub) > 0:
            matched_id += 1
            found = True
            break
    if not found and any(v for k, v in id_candidates):
        print(f"  Unmatched RC ID for {clean_doc}: GT={[v for k, v in id_candidates if v]}, top candidates={grp['m_parsed_value'].tolist()[:3]}")

print(f"RC ID matched: {matched_id} / {df_id['m_document_id'].nunique()}")

print("\n=== 2. Testing RATE AGREEMENT Enum (Currency) ===")
df_enum = pd.read_csv(os.path.join(csv_dir, "candidates_RATE_AGREEMENT_enum.csv"), dtype=str)
print("RC Enum docs:", df_enum["m_document_id"].nunique(), "unique currencies:", df_enum["m_parsed_norm"].unique())

print("\n=== 3. Testing RATE AGREEMENT Amount ===")
df_amt = pd.read_csv(os.path.join(csv_dir, "candidates_RATE_AGREEMENT_amount.csv"), dtype=str)
matched_amt = 0
for doc_id, grp in df_amt.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    # collect all prices in gt
    prices = set()
    for k in ["price_1", "price_2", "adm_fee", "grand_total", "total_cost_usd", "total_usd", "total_amount_usd", "carrier_pay"]:
        if k in gt and gt[k]:
            prices.add((k, clean_amount(gt[k])))
    if "items" in gt and isinstance(gt["items"], list):
        for itm in gt["items"]:
            if "harga" in itm: prices.add(("items.harga", clean_amount(itm["harga"])))
    if "carrier_fees" in gt and isinstance(gt["carrier_fees"], list):
        for fee in gt["carrier_fees"]:
            if "cost_usd" in fee: prices.add(("carrier_fees.cost_usd", clean_amount(fee["cost_usd"])))
    if "payment" in gt and isinstance(gt["payment"], dict):
        if "rate" in gt["payment"]: prices.add(("payment.rate", clean_amount(gt["payment"]["rate"])))
    if "rate_breakdown" in gt and isinstance(gt["rate_breakdown"], dict):
        for rk, rv in gt["rate_breakdown"].items():
            prices.add((f"rate_breakdown.{rk}", clean_amount(rv)))
    if "rate_details" in gt and isinstance(gt["rate_details"], dict):
        for rk, rv in gt["rate_details"].items():
            prices.add((f"rate_details.{rk}", clean_amount(rv)))
            
    # Check matching
    found = False
    for k, p in prices:
        if not p: continue
        m = grp[grp["m_parsed_norm"].apply(clean_amount) == p]
        if len(m) > 0:
            matched_amt += 1
            found = True
            break
    if not found:
        print(f"  Unmatched RC Amount for {clean_doc}: GT prices={[p for k, p in prices if p]}, cand norms={grp['m_parsed_norm'].tolist()[:4]}")

print(f"RC Amount matched: {matched_amt} / {df_amt['m_document_id'].nunique()}")

print("\n=== 4. Testing RATE AGREEMENT City ===")
df_city = pd.read_csv(os.path.join(csv_dir, "candidates_RATE_AGREEMENT_city.csv"), dtype=str)
print("RC City candidates:", len(df_city), "across", df_city["m_document_id"].nunique(), "docs")
print(df_city[["m_document_id", "m_parsed_value"]].head(10))

print("\n=== 5. Testing RATE AGREEMENT Date ===")
df_date = pd.read_csv(os.path.join(csv_dir, "candidates_RATE_AGREEMENT_date.csv"), dtype=str)
print("RC Date candidates:", len(df_date), "across", df_date["m_document_id"].nunique(), "docs")
