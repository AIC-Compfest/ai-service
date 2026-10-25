import json, os, re
import pandas as pd

gt_base = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset"
csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

with open(os.path.join(gt_base, "invoice", "ground_truth_variasi_1.json"), encoding="utf-8") as f:
    inv1 = json.load(f)["documents"]
with open(os.path.join(gt_base, "invoice", "ground_truth_variasi_2.json"), encoding="utf-8") as f:
    inv2 = json.load(f)["documents"]

inv_gt = {}
for d in inv1 + inv2:
    inv_gt[d["file_name"].replace(".pdf", "").lower()] = d

inv_evidence_truth = {
    "invoice_5_variasi_1": {
        "invoice_number": "INV/2017/0001",
        "invoice_date": "2017-12-20",
        "customer_name": "Adam Aria",
        "grand_total": "34595294",
        "currency": "IDR"
    },
    "invoice_5_variasi_2": {
        "invoice_number": "INV/2024/0105",
        "invoice_date": "2024-01-15",
        "customer_name": "PT Tech Utama",
        "grand_total": "83150000",
        "currency": "IDR"
    },
    "invoice_6_variasi_1": {
        "invoice_number": "001",
        "invoice_date": "2024-02-19",
        "customer_name": "INDOREX",
        "vendor_name": "ANDRIS NARYONO",
        "grand_total": "2810000",
        "subtotal": "2810000",
        "currency": "IDR"
    },
    "invoice_6_variasi_2": {
        "invoice_number": "002",
        "invoice_date": "2024-02-25",
        "customer_name": "MAKMUR SEJAHTERA UTAMA",
        "vendor_name": "ANDRIS NARYONO",
        "grand_total": "5450000",
        "subtotal": "5450000",
        "currency": "IDR"
    },
    "invoice_7_variasi_1": {
        "invoice_number": "0409015",
        "invoice_date": "2009-04-08",
        "vendor_name": "ADR Cargo",
        "customer_name": "PT MITRA SINERGI UTAMA",
        "grand_total": "38850000",
        "subtotal": "38850000",
        "currency": "IDR"
    },
    "invoice_7_variasi_2": {
        "invoice_number": "0509027",
        "invoice_date": "2009-05-21",
        "vendor_name": "ADR Cargo",
        "customer_name": "PT NUSANTARA ELEKTRIK",
        "grand_total": "61671000",
        "subtotal": "61000000",
        "tax_amount": "671000",
        "currency": "IDR"
    },
    "invoice_8_variasi_1": {
        "invoice_number": "316/DL/VIII/2016",
        "invoice_date": "2016-08-27",
        "vendor_name": "PT. DIVA MITRA SINERGI",
        "customer_name": "PT. SINAR GUNDALING",
        "subtotal": "3900000",
        "grand_total": "3950000",
        "currency": "IDR"
    },
    "invoice_8 _variasi_2": {
        "invoice_number": "422/DL/IX/2024",
        "invoice_date": "2024-09-12",
        "vendor_name": "PT. DIVA MITRA SINERGI",
        "customer_name": "PT. MAKMUR SEJAHTERA",
        "subtotal": "15000000",
        "grand_total": "15000000",
        "currency": "IDR"
    },
    "invoice_8_variasi_2": {
        "invoice_number": "422/DL/IX/2024",
        "invoice_date": "2024-09-12",
        "vendor_name": "PT. DIVA MITRA SINERGI",
        "customer_name": "PT. MAKMUR SEJAHTERA",
        "subtotal": "15000000",
        "grand_total": "15000000",
        "currency": "IDR"
    },
    "invoice_9_variasi_1": {
        "invoice_number": "0100/INV/PRT/XI/2019",
        "invoice_date": "2019-11-13",
        "vendor_name": "CV. PENDIKS RENTAL TRANSINDO",
        "customer_name": "PT. Petromas Kencana",
        "grand_total": "5900000",
        "subtotal": "5900000",
        "currency": "IDR"
    },
    "invoice_9_variasi_2": {
        "invoice_number": "0518/INV/PRT/VIII/2026",
        "invoice_date": "2026-08-24",
        "vendor_name": "CV. PENDIKS RENTAL TRANSINDO",
        "customer_name": "PT. Bintang Jaya Konstruksi",
        "grand_total": "18500000",
        "subtotal": "18500000",
        "currency": "IDR"
    },
    "invoice_10_variasi_1": {
        "invoice_number": "ILS32/OUT/08-20/0015",
        "invoice_date": "2021-08-20",
        "vendor_name": "PT. INDONESIA LOGISTIK SERVIS",
        "customer_name": "MUHAMMAD RIZKY ARMILAH",
        "subtotal": "20000",
        "grand_total": "25000",
        "currency": "IDR"
    },
    "invoice_10_variasi_2": {
        "invoice_number": "ILS32/OUT/11-24/0288",
        "invoice_date": "2024-11-14",
        "vendor_name": "PT. INDONESIA LOGISTIK SERVIS",
        "customer_name": "PT. MAHAKAM MAKMUR UTAMA",
        "subtotal": "35000",
        "grand_total": "50000",
        "currency": "IDR"
    },
    "invoice_11_variasi_1": {
        "invoice_number": "903/DT2-JKT/VIII/23",
        "invoice_date": "2023-08-10",
        "vendor_name": "PT. HARAPAN BARU LINE",
        "customer_name": "PT. TRITUNGGAL LINTAS BENUA",
        "grand_total": "2250000",
        "subtotal": "2250000",
        "currency": "IDR"
    },
    "invoice_11_variasi_2": {
        "invoice_number": "114/DT4-JKT/X/24",
        "invoice_date": "2024-10-18",
        "vendor_name": "PT. HARAPAN BARU LINE",
        "customer_name": "CV. BORNEO LOGISTIK NUSANTARA",
        "grand_total": "8150000",
        "subtotal": "7800000",
        "currency": "IDR"
    }
}

def clean_alphanum(s):
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()

def clean_amount(s):
    if not s or pd.isna(s):
        return ""
    digits = re.findall(r"\d+", str(s).split("/")[0].split("-")[0].strip())
    if not digits:
        return ""
    return "".join(digits)

print("=== 1. Labeling candidates_INVOICE_id.csv ===")
path_id = os.path.join(csv_dir, "candidates_INVOICE_id.csv")
df_id = pd.read_csv(path_id, dtype=str)
df_id["label__invoice_number"] = "0"

for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("invoice--pdf--", "").lower()
    gt = inv_gt.get(clean_doc, {})
    gt_no = gt.get("invoice_number")
    note = ""
    
    if clean_doc in inv_evidence_truth and "invoice_number" in inv_evidence_truth[clean_doc]:
        gt_no = inv_evidence_truth[clean_doc]["invoice_number"]
        note = "from evidence, GT null"
        
    if not gt_no:
        continue
    target_clean = clean_alphanum(gt_no)
    
    matched_idx = []
    for idx, row in grp.iterrows():
        pv_clean = clean_alphanum(row["m_parsed_value"])
        if pv_clean == target_clean or (len(target_clean) >= 4 and pv_clean in target_clean):
            matched_idx.append(idx)
            
    for idx in matched_idx:
        df_id.at[idx, "label__invoice_number"] = "1"
        if note:
            df_id.at[idx, "label_notes"] = note

print(f"Labeled INVOICE ID: {sum(df_id['label__invoice_number'] == '1')} positive labels across {df_id['m_document_id'].nunique()} docs.")
df_id.to_csv(path_id, index=False)


print("\n=== 2. Labeling candidates_INVOICE_date.csv ===")
path_date = os.path.join(csv_dir, "candidates_INVOICE_date.csv")
df_date = pd.read_csv(path_date, dtype=str)
df_date["label__invoice_date"] = "0"
df_date["label__due_date"] = "0"

for doc_id, grp in df_date.groupby("m_document_id"):
    clean_doc = doc_id.replace("invoice--pdf--", "").lower()
    gt = inv_gt.get(clean_doc, {})
    
    inv_date = gt.get("invoice_date")
    due_date = gt.get("due_date")
    inv_note, due_note = "", ""
    
    if clean_doc in inv_evidence_truth:
        if "invoice_date" in inv_evidence_truth[clean_doc]:
            inv_date = inv_evidence_truth[clean_doc]["invoice_date"]
            inv_note = "from evidence, GT null"
        if "due_date" in inv_evidence_truth[clean_doc]:
            due_date = inv_evidence_truth[clean_doc]["due_date"]
            due_note = "from evidence, GT null"
            
    if inv_date:
        for idx, row in grp.iterrows():
            if row["m_parsed_norm"] == inv_date:
                df_date.at[idx, "label__invoice_date"] = "1"
                if inv_note:
                    df_date.at[idx, "label_notes"] = inv_note
                break
                
    if due_date:
        for idx, row in grp.iterrows():
            if row["m_parsed_norm"] == due_date:
                df_date.at[idx, "label__due_date"] = "1"
                if due_note:
                    existing = str(df_date.at[idx, "label_notes"]) if pd.notna(df_date.at[idx, "label_notes"]) else ""
                    df_date.at[idx, "label_notes"] = (existing + "; " + due_note).strip("; ")
                break

print(f"Labeled INVOICE Date: InvDate={sum(df_date['label__invoice_date'] == '1')}, DueDate={sum(df_date['label__due_date'] == '1')}.")
df_date.to_csv(path_date, index=False)


print("\n=== 3. Labeling candidates_INVOICE_enum.csv ===")
path_enum = os.path.join(csv_dir, "candidates_INVOICE_enum.csv")
df_enum = pd.read_csv(path_enum, dtype=str)
df_enum["label__currency"] = "0"

usd_templates = ["invoice_16", "invoice_17", "invoice_18", "invoice_20"]

for doc_id, grp in df_enum.groupby("m_document_id"):
    clean_doc = doc_id.replace("invoice--pdf--", "").lower()
    gt = inv_gt.get(clean_doc, {})
    curr = gt.get("currency")
    if not curr:
        curr = "USD" if any(t in clean_doc for t in usd_templates) else "IDR"
    target_curr = curr.lower()
    
    matches = grp[grp["m_parsed_norm"] == target_curr]
    if len(matches) > 0:
        best_idx = matches.index[0]
        df_enum.at[best_idx, "label__currency"] = "1"

print(f"Labeled INVOICE Enum: {sum(df_enum['label__currency'] == '1')} positive labels across {df_enum['m_document_id'].nunique()} docs.")
df_enum.to_csv(path_enum, index=False)


print("\n=== 4. Labeling candidates_INVOICE_amount.csv ===")
path_amt = os.path.join(csv_dir, "candidates_INVOICE_amount.csv")
df_amt = pd.read_csv(path_amt, dtype=str)
df_amt["label__grand_total"] = "0"
df_amt["label__subtotal"] = "0"
df_amt["label__tax_amount"] = "0"

for doc_id, grp in df_amt.groupby("m_document_id"):
    clean_doc = doc_id.replace("invoice--pdf--", "").lower()
    gt = inv_gt.get(clean_doc, {})
    totals = gt.get("totals", {}) if isinstance(gt.get("totals"), dict) else {}
    
    gt_total = None
    gt_total_k = ""
    for k in ["grand_total", "total_due", "total", "total_usd", "total_invoice_value_usd", "net_total", "total_value_usd"]:
        if totals.get(k) is not None:
            gt_total = totals.get(k)
            gt_total_k = k
            break
            
    gt_subtotal = None
    gt_sub_k = ""
    for k in ["subtotal", "total_charges", "gross_total", "items_subtotal_usd"]:
        if totals.get(k) is not None:
            gt_subtotal = totals.get(k)
            gt_sub_k = k
            break
            
    gt_tax = None
    gt_tax_k = ""
    for k in ["tax", "vat", "ppn"]:
        if totals.get(k) is not None:
            gt_tax = totals.get(k)
            gt_tax_k = k
            break
            
    note = ""
    if clean_doc in inv_evidence_truth:
        e = inv_evidence_truth[clean_doc]
        if "grand_total" in e:
            gt_total = e["grand_total"]
            gt_total_k = "evidence_grand_total"
        if "subtotal" in e:
            gt_subtotal = e["subtotal"]
            gt_sub_k = "evidence_subtotal"
        if "tax_amount" in e:
            gt_tax = e["tax_amount"]
            gt_tax_k = "evidence_tax_amount"
        note = "from evidence, GT null"
        
    t_clean = clean_amount(gt_total)
    sub_clean = clean_amount(gt_subtotal)
    tax_clean = clean_amount(gt_tax)
    
    # 1. Match grand_total
    if t_clean:
        for idx, row in grp.iterrows():
            cnorm = clean_amount(row["m_parsed_norm"])
            cval = clean_amount(row["m_parsed_value"])
            if cnorm == t_clean or cval == t_clean or (len(cnorm) == len(t_clean) + 2 and cnorm.startswith(t_clean) and cnorm.endswith("00")):
                df_amt.at[idx, "label__grand_total"] = "1"
                df_amt.at[idx, "label_notes"] = f"gt_field:{gt_total_k}" if gt_total_k != "grand_total" else note
                break
                
    # 2. Match subtotal
    if sub_clean and sub_clean != t_clean:
        for idx, row in grp.iterrows():
            cnorm = clean_amount(row["m_parsed_norm"])
            cval = clean_amount(row["m_parsed_value"])
            if cnorm == sub_clean or cval == sub_clean or (len(cnorm) == len(sub_clean) + 2 and cnorm.startswith(sub_clean) and cnorm.endswith("00")):
                df_amt.at[idx, "label__subtotal"] = "1"
                existing = str(df_amt.at[idx, "label_notes"]) if pd.notna(df_amt.at[idx, "label_notes"]) else ""
                n = f"gt_field:{gt_sub_k}" if gt_sub_k != "subtotal" else note
                df_amt.at[idx, "label_notes"] = (existing + f"; {n}").strip("; ")
                break
    elif sub_clean == t_clean and t_clean:
        for idx, row in grp.iterrows():
            if df_amt.at[idx, "label__grand_total"] == "1":
                df_amt.at[idx, "label__subtotal"] = "1"
                break
                
    # 3. Match tax
    if tax_clean and tax_clean != "0":
        for idx, row in grp.iterrows():
            cnorm = clean_amount(row["m_parsed_norm"])
            cval = clean_amount(row["m_parsed_value"])
            if cnorm == tax_clean or cval == tax_clean or (len(cnorm) == len(tax_clean) + 2 and cnorm.startswith(tax_clean) and cnorm.endswith("00")):
                df_amt.at[idx, "label__tax_amount"] = "1"
                existing = str(df_amt.at[idx, "label_notes"]) if pd.notna(df_amt.at[idx, "label_notes"]) else ""
                n = f"gt_field:{gt_tax_k}" if gt_tax_k != "tax_amount" else note
                df_amt.at[idx, "label_notes"] = (existing + f"; {n}").strip("; ")
                break

print(f"Labeled INVOICE Amount: GT={sum(df_amt['label__grand_total'] == '1')}, Sub={sum(df_amt['label__subtotal'] == '1')}, Tax={sum(df_amt['label__tax_amount'] == '1')}.")
df_amt.to_csv(path_amt, index=False)


print("\n=== 5. Labeling candidates_INVOICE_party.csv ===")
path_party = os.path.join(csv_dir, "candidates_INVOICE_party.csv")
df_party = pd.read_csv(path_party, dtype=str)
df_party["label__vendor_name"] = "0"
df_party["label__customer_name"] = "0"

for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("invoice--pdf--", "").lower()
    gt = inv_gt.get(clean_doc, {})
    
    vendor = gt.get("seller", {}).get("name") if isinstance(gt.get("seller"), dict) else None
    customer = gt.get("buyer", {}).get("name") if isinstance(gt.get("buyer"), dict) else None
    v_note, c_note = "", ""
    
    if clean_doc in inv_evidence_truth:
        e = inv_evidence_truth[clean_doc]
        if "vendor_name" in e:
            vendor = e["vendor_name"]
            v_note = "from evidence, GT null"
        if "customer_name" in e:
            customer = e["customer_name"]
            c_note = "from evidence, GT null"
            
    v_clean = clean_alphanum(vendor)
    c_clean = clean_alphanum(customer)
    
    # Match vendor
    if v_clean:
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == v_clean or (len(pv_clean) >= 5 and (pv_clean in v_clean or v_clean in pv_clean)):
                df_party.at[idx, "label__vendor_name"] = "1"
                if v_note: df_party.at[idx, "label_notes"] = v_note
                break
                
    # Match customer
    if c_clean:
        for idx, row in grp.iterrows():
            if df_party.at[idx, "label__vendor_name"] == "1":
                continue
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == c_clean or (len(pv_clean) >= 5 and (pv_clean in c_clean or c_clean in pv_clean)):
                df_party.at[idx, "label__customer_name"] = "1"
                if c_note:
                    existing = str(df_party.at[idx, "label_notes"]) if pd.notna(df_party.at[idx, "label_notes"]) else ""
                    df_party.at[idx, "label_notes"] = (existing + "; " + c_note).strip("; ")
                break

print(f"Labeled INVOICE Party: Vendor={sum(df_party['label__vendor_name'] == '1')}, Customer={sum(df_party['label__customer_name'] == '1')}.")
df_party.to_csv(path_party, index=False)


print("\n=== 6. Labeling candidates_INVOICE_table.csv ===")
path_tab = os.path.join(csv_dir, "candidates_INVOICE_table.csv")
df_tab = pd.read_csv(path_tab, dtype=str)
df_tab["label__line_items"] = "0"

for doc_id, grp in df_tab.groupby("m_document_id"):
    best_idx = None
    for idx, row in grp.iterrows():
        rt = str(row["m_raw_text"]).lower()
        if "syarat-syarat" in rt: continue
        if "peraturan airline" in rt: continue
        if "transfer full amount" in rt: continue
        if "please transfer" in rt: continue
        if "hormat kami" in rt and "penerima" in rt: continue
        if rt.strip().startswith("subtotal") or rt.strip().startswith("total"): continue
        if any(w in rt for w in ["description", "nama barang", "uraian", "keterangan", "ondol", "item #", "goods", "harga", "ton"]):
            best_idx = idx
            break
            
    if best_idx is None:
        li = grp[grp["m_evidence_role"] == "line_items"]
        if len(li) > 0:
            best_idx = li.index[0]
            
    if best_idx is not None:
        df_tab.at[best_idx, "label__line_items"] = "1"

print(f"Labeled INVOICE Table: {sum(df_tab['label__line_items'] == '1')} line item tables across {df_tab['m_document_id'].nunique()} docs.")
df_tab.to_csv(path_tab, index=False)

print("\n>>> ALL 6 INVOICE CSV FILES SUCCESSFULLY LABELED! <<<")
