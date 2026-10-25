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
    s = str(s).split("/")[0].split("-")[0].strip()
    digits = re.findall(r"\d+", s)
    if not digits:
        return ""
    if "." in s and len(digits) == 2 and len(digits[1]) in [1, 2]:
        return digits[0]
    return "".join(digits)

month_map = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05", "juni": "06",
    "juli": "07", "agustus": "08", "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def parse_date_to_iso(dt_str):
    if not dt_str:
        return ""
    # "14 JUNI 2026"
    m = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})", str(dt_str))
    if m:
        day, mon, year = m.groups()
        mon_num = month_map.get(mon.lower(), "")
        if mon_num:
            return f"{year}-{mon_num}-{int(day):02d}"
    # "08/04/2025" -> MM/DD/YYYY in US templates
    m2 = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(dt_str))
    if m2:
        m_num, d, y = m2.groups()
        return f"{y}-{int(m_num):02d}-{int(d):02d}"
    # "2026-06-14"
    m3 = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(dt_str))
    if m3:
        y, m_num, d = m3.groups()
        return f"{y}-{int(m_num):02d}-{int(d):02d}"
    return ""


print("=== 1. Labeling candidates_RATE_AGREEMENT_id.csv ===")
path_id = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_id.csv")
df_id = pd.read_csv(path_id, dtype=str)
df_id["label__load_number"] = "0"

for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    
    id_candidates = [
        ("load_number", gt.get("load_number")),
        ("load_id", gt.get("load_id")),
        ("nomor", gt.get("nomor")),
        ("work_order", gt.get("work_order")),
        ("order_number", gt.get("order_number")),
        ("references", gt.get("references")),
    ]
    
    matched_idx = []
    matched_k = ""
    for k, val in id_candidates:
        if not val: continue
        val_clean = clean_alphanum(val)
        if len(val_clean) < 3: continue
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == val_clean or (len(pv_clean) >= 5 and (pv_clean in val_clean or val_clean in pv_clean)):
                matched_idx.append(idx)
        if matched_idx:
            matched_k = k
            break
            
    for idx in matched_idx:
        df_id.at[idx, "label__load_number"] = "1"
        if matched_k != "load_number":
            df_id.at[idx, "label_notes"] = f"gt_field:{matched_k}"

print(f"Labeled RC ID: {sum(df_id['label__load_number'] == '1')} positive labels across {df_id['m_document_id'].nunique()} docs.")
df_id.to_csv(path_id, index=False)


print("\n=== 2. Labeling candidates_RATE_AGREEMENT_amount.csv ===")
path_amt = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_amount.csv")
df_amt = pd.read_csv(path_amt, dtype=str)
df_amt["label__total"] = "0"

for doc_id, grp in df_amt.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    
    prices = []
    # Check all loose price fields
    for k in ["total_cost_usd", "total_usd", "total_amount_usd", "grand_total", "carrier_pay", "price_1", "price_2", "adm_fee"]:
        if k in gt and gt[k]:
            prices.append((k, clean_amount(gt[k])))
    for k in ["air_biasa_1", "air_biasa_2", "air_express", "sea_kg", "sea_m3", "air_charge_1", "air_charge_2", "sea_charge"]:
        if k in gt and gt[k]:
            prices.append((k, clean_amount(gt[k])))
    if "carrier_fees" in gt and isinstance(gt["carrier_fees"], list):
        for fee in gt["carrier_fees"]:
            if "cost_usd" in fee: prices.append(("carrier_fees.cost_usd", clean_amount(fee["cost_usd"])))
    if "payment" in gt and isinstance(gt["payment"], dict):
        if "rate" in gt["payment"]: prices.append(("payment.rate", clean_amount(gt["payment"]["rate"])))
    if "rate_breakdown" in gt and isinstance(gt["rate_breakdown"], dict):
        for rk, rv in gt["rate_breakdown"].items():
            prices.append((f"rate_breakdown.{rk}", clean_amount(rv)))
    if "rate_details" in gt and isinstance(gt["rate_details"], dict):
        for rk, rv in gt["rate_details"].items():
            prices.append((f"rate_details.{rk}", clean_amount(rv)))
    if "items" in gt and isinstance(gt["items"], list):
        for itm in gt["items"]:
            if "harga" in itm: prices.append(("items.harga", clean_amount(itm["harga"])))
            if "harga_satuan" in itm: prices.append(("items.harga_satuan", clean_amount(itm["harga_satuan"])))
            if "jumlah" in itm: prices.append(("items.jumlah", clean_amount(itm["jumlah"])))
            
    # For matching candidate norm
    for idx, row in grp.iterrows():
        cnorm = clean_amount(row["m_parsed_norm"])
        cval = clean_amount(row["m_parsed_value"])
        if not cnorm and not cval:
            continue
        # Check against prices
        for k, p in prices:
            if not p: continue
            # Exact match, or match when cents are appended (e.g. 475 vs 47500)
            if cnorm == p or cval == p or (len(cnorm) == len(p) + 2 and cnorm.startswith(p) and cnorm.endswith("00")):
                df_amt.at[idx, "label__total"] = "1"
                df_amt.at[idx, "label_notes"] = f"gt_field:{k}"
                break

print(f"Labeled RC Amount: {sum(df_amt['label__total'] == '1')} positive labels across {df_amt['m_document_id'].nunique()} docs.")
df_amt.to_csv(path_amt, index=False)


print("\n=== 3. Labeling candidates_RATE_AGREEMENT_city.csv ===")
path_city = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_city.csv")
df_city = pd.read_csv(path_city, dtype=str)
df_city["label__destination"] = "0"

for doc_id, grp in df_city.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    
    destinations = []
    for k in ["destination_1", "destination_2", "city", "frangko_city", "recipient_city"]:
        if k in gt and gt[k]: destinations.append((k, gt[k]))
    if "items" in gt and isinstance(gt["items"], list):
        for itm in gt["items"]:
            if "tujuan" in itm: destinations.append(("items.tujuan", itm["tujuan"]))
    # US pickup/delivery stop city
    for k in ["shipper_pickup_stop_1", "consignee_delivery_stop_2"]:
        if k in gt and isinstance(gt[k], dict) and gt[k].get("city_state_zip"):
            destinations.append((f"{k}.city", gt[k]["city_state_zip"]))
            
    for idx, row in grp.iterrows():
        val = clean_alphanum(row["m_parsed_value"])
        if len(val) < 3: continue
        for k, d in destinations:
            d_clean = clean_alphanum(d)
            if val in d_clean or d_clean in val:
                df_city.at[idx, "label__destination"] = "1"
                df_city.at[idx, "label_notes"] = f"gt_field:{k}"
                break

print(f"Labeled RC City: {sum(df_city['label__destination'] == '1')} positive labels across {df_city['m_document_id'].nunique()} docs.")
df_city.to_csv(path_city, index=False)


print("\n=== 4. Labeling candidates_RATE_AGREEMENT_date.csv ===")
path_date = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_date.csv")
df_date = pd.read_csv(path_date, dtype=str)
df_date["label__rate_card_date"] = "0"

for doc_id, grp in df_date.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    
    dates = []
    for k in ["tanggal", "date", "date_active", "effective_date", "ship_date", "pickup_date", "signature_date", "todays_date"]:
        if k in gt and gt[k]:
            dates.append((k, parse_date_to_iso(gt[k])))
            
    matched_idx = []
    matched_k = ""
    for k, iso in dates:
        if not iso: continue
        for idx, row in grp.iterrows():
            if row["m_parsed_norm"] == iso:
                matched_idx.append(idx)
        if matched_idx:
            matched_k = k
            break
            
    for idx in matched_idx:
        df_date.at[idx, "label__rate_card_date"] = "1"
        if matched_k:
            df_date.at[idx, "label_notes"] = f"gt_field:{matched_k}"

print(f"Labeled RC Date: {sum(df_date['label__rate_card_date'] == '1')} positive labels across {df_date['m_document_id'].nunique()} docs.")
df_date.to_csv(path_date, index=False)


print("\n=== 5. Labeling candidates_RATE_AGREEMENT_enum.csv ===")
path_enum = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_enum.csv")
df_enum = pd.read_csv(path_enum, dtype=str)
df_enum["label__currency"] = "0"

for doc_id, grp in df_enum.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    # Cards 1 to 7 are IDR, Cards 8 to 15 are USD
    is_us = any(f"rate_card_{i}_" in clean_doc for i in range(8, 16))
    target_curr = "usd" if is_us else "idr"
    
    matches = grp[grp["m_parsed_norm"] == target_curr]
    if len(matches) > 0:
        # Pick first occurrence
        best_idx = matches.index[0]
        df_enum.at[best_idx, "label__currency"] = "1"

print(f"Labeled RC Enum: {sum(df_enum['label__currency'] == '1')} positive labels across {df_enum['m_document_id'].nunique()} docs.")
df_enum.to_csv(path_enum, index=False)


print("\n=== 6. Labeling candidates_RATE_AGREEMENT_party.csv ===")
path_party = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_party.csv")
df_party = pd.read_csv(path_party, dtype=str)
df_party["label__vendor_name"] = "0"

for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("rate card--pdf--", "").lower()
    gt = rc_gt.get(clean_doc, {})
    
    vendors = []
    # Primary: carrier / company_name / broker
    for k in ["company_name", "company_header", "carrier", "carrier_legal_name", "company", "carrier_info", "broker", "broker_company"]:
        val = gt.get(k)
        if not val: continue
        if isinstance(val, dict) and val.get("name"):
            vendors.append((f"{k}.name", val["name"]))
        elif isinstance(val, str):
            vendors.append((k, val))
            
    matched_idx = []
    matched_k = ""
    for k, v in vendors:
        v_clean = clean_alphanum(v)
        if len(v_clean) < 4: continue
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == v_clean or (len(pv_clean) >= 6 and (pv_clean in v_clean or v_clean in pv_clean)):
                matched_idx.append(idx)
                matched_k = k
                break
        if matched_idx:
            break
            
    for idx in matched_idx:
        df_party.at[idx, "label__vendor_name"] = "1"
        df_party.at[idx, "label_notes"] = f"gt_field:{matched_k}"

print(f"Labeled RC Party: {sum(df_party['label__vendor_name'] == '1')} positive labels across {df_party['m_document_id'].nunique()} docs.")
df_party.to_csv(path_party, index=False)


print("\n=== 7. Labeling candidates_RATE_AGREEMENT_table.csv ===")
path_tab = os.path.join(csv_dir, "candidates_RATE_AGREEMENT_table.csv")
df_tab = pd.read_csv(path_tab, dtype=str)
df_tab["label__line_items"] = "0"

for doc_id, grp in df_tab.groupby("m_document_id"):
    # Rate table
    best_idx = None
    for idx, row in grp.iterrows():
        rt = str(row["m_raw_text"]).lower()
        if any(w in rt for w in ["tujuan", "rute", "harga", "rate", "cost", "carrier fees", "description", "charge"]):
            best_idx = idx
            break
    if best_idx is None:
        li = grp[grp["m_evidence_role"] == "line_items"]
        if len(li) > 0:
            best_idx = li.index[0]
        else:
            best_idx = grp.index[0]
            
    df_tab.at[best_idx, "label__line_items"] = "1"

print(f"Labeled RC Table: {sum(df_tab['label__line_items'] == '1')} line item tables across {df_tab['m_document_id'].nunique()} docs.")
df_tab.to_csv(path_tab, index=False)

print("\n>>> ALL 7 RATE AGREEMENT CSV FILES SUCCESSFULLY LABELED! <<<")
