import json, os, re
import pandas as pd

gt_path = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset\surat jalan\ground_truth_surat_jalan.json"
with open(gt_path, encoding="utf-8") as f:
    sj_docs = json.load(f)["documents"]

sj_gt = {d["file"].replace(".pdf", "").lower(): d["ground_truth"] for d in sj_docs}
csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

def clean_alphanum(s):
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()

month_map = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05", "juni": "06",
    "juli": "07", "agustus": "08", "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def parse_date_to_iso(dt_str):
    if not dt_str:
        return ""
    # "12 Maret 2018"
    m = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})", str(dt_str))
    if m:
        day, mon, year = m.groups()
        mon_num = month_map.get(mon.lower(), "")
        if mon_num:
            return f"{year}-{mon_num}-{int(day):02d}"
    # "12-09-2025" or "12-10-2025"
    m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", str(dt_str))
    if m2:
        d, m_num, y = m2.groups()
        return f"{y}-{int(m_num):02d}-{int(d):02d}"
    # "17/06/2003" or "3/14/2023"
    m3 = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(dt_str))
    if m3:
        p1, p2, y = m3.groups()
        if int(p1) > 12:
            return f"{y}-{int(p2):02d}-{int(p1):02d}"
        elif int(p2) > 12:
            return f"{y}-{int(p1):02d}-{int(p2):02d}"
        else:
            return f"{y}-{int(p2):02d}-{int(p1):02d}"
    return ""

print("=== 1. Labeling candidates_SURAT_JALAN_id.csv ===")
path_id = os.path.join(csv_dir, "candidates_SURAT_JALAN_id.csv")
df_id = pd.read_csv(path_id, dtype=str)
df_id["label__surat_jalan_number"] = "0"

for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    # Priority list of loose ID keys in SJ GT
    id_candidates = [
        ("nomor", gt.get("nomor")),
        ("no_surat_jalan", gt.get("no_surat_jalan")),
        ("no_dokumen", gt.get("no_dokumen")),
        ("no", gt.get("no")),
        ("order_number", gt.get("order_number")),
        ("no_invoice", gt.get("no_invoice")),
        ("no_resi", gt.get("no_resi")),
        ("shipment_no", gt.get("shipment_no")),
        ("nomor_invoice", gt.get("nomor_invoice")),
        ("invoice_no", gt.get("invoice_no")),
        ("po_no", gt.get("po_no")),
        ("nomor_order", gt.get("nomor_order")),
    ]
    
    matched_idx = []
    matched_k = ""
    for k, val in id_candidates:
        if not val:
            continue
        val_clean = clean_alphanum(val)
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == val_clean or (len(val_clean) >= 6 and (pv_clean in val_clean or val_clean in pv_clean)):
                matched_idx.append(idx)
        if matched_idx:
            matched_k = k
            break
            
    for idx in matched_idx:
        df_id.at[idx, "label__surat_jalan_number"] = "1"
        if matched_k != "nomor":
            df_id.at[idx, "label_notes"] = f"gt_field:{matched_k}"

print(f"Labeled SJ ID: {sum(df_id['label__surat_jalan_number'] == '1')} positive labels across {df_id['m_document_id'].nunique()} docs.")
df_id.to_csv(path_id, index=False)


print("\n=== 2. Labeling candidates_SURAT_JALAN_date.csv ===")
path_date = os.path.join(csv_dir, "candidates_SURAT_JALAN_date.csv")
df_date = pd.read_csv(path_date, dtype=str)
df_date["label__shipment_date"] = "0"

# Specific dates from evidence text for docs where date parsing differed or GT was empty
sj_evidence_dates = {
    "surat_jalan_4_variasi_1": "2023-10-03",
    "surat_jalan_4_variasi_2": "2023-11-21",
    "surat_jalan_7_variasi_2": "2025-12-03",
}

for doc_id, grp in df_date.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    dt_candidates = [
        ("tanggal", gt.get("tanggal")),
        ("tanggal_kirim", gt.get("tanggal_kirim")),
        ("tanggal_pengiriman", gt.get("tanggal_pengiriman")),
        ("date", gt.get("date")),
        ("tgl_kirim", gt.get("tgl_kirim")),
    ]
    
    matched_idx = []
    matched_k = ""
    
    for k, val in dt_candidates:
        if not val:
            continue
        iso = parse_date_to_iso(val)
        if not iso:
            continue
        for idx, row in grp.iterrows():
            if row["m_parsed_norm"] == iso:
                matched_idx.append(idx)
        if matched_idx:
            matched_k = k
            break
            
    if not matched_idx and clean_doc in sj_evidence_dates:
        target_iso = sj_evidence_dates[clean_doc]
        for idx, row in grp.iterrows():
            if row["m_parsed_norm"] == target_iso:
                matched_idx.append(idx)
        matched_k = "evidence_pickup_date"
        
    for idx in matched_idx:
        df_date.at[idx, "label__shipment_date"] = "1"
        if matched_k and matched_k != "tanggal":
            df_date.at[idx, "label_notes"] = f"gt_field:{matched_k}"

print(f"Labeled SJ Date: {sum(df_date['label__shipment_date'] == '1')} positive labels across {df_date['m_document_id'].nunique()} docs.")
df_date.to_csv(path_date, index=False)


print("\n=== 3. Labeling candidates_SURAT_JALAN_party.csv ===")
path_party = os.path.join(csv_dir, "candidates_SURAT_JALAN_party.csv")
df_party = pd.read_csv(path_party, dtype=str)
df_party["label__sender_name"] = "0"
df_party["label__receiver_name"] = "0"

for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    # Senders
    senders = []
    if "distributor" in gt and gt["distributor"]:
        senders.append(("distributor", gt["distributor"]))
    if "kop" in gt and isinstance(gt["kop"], dict):
        if gt["kop"].get("nama_kelompok"): senders.append(("kop.nama_kelompok", gt["kop"]["nama_kelompok"]))
        if gt["kop"].get("nama"): senders.append(("kop.nama", gt["kop"]["nama"]))
    if "perusahaan" in gt and isinstance(gt["perusahaan"], dict):
        if gt["perusahaan"].get("nama"): senders.append(("perusahaan.nama", gt["perusahaan"]["nama"]))
    if "toko" in gt and isinstance(gt["toko"], dict):
        if gt["toko"].get("nama"): senders.append(("toko.nama", gt["toko"]["nama"]))
    if "pengirim" in gt:
        if isinstance(gt["pengirim"], dict) and gt["pengirim"].get("nama"): senders.append(("pengirim.nama", gt["pengirim"]["nama"]))
        elif isinstance(gt["pengirim"], str): senders.append(("pengirim", gt["pengirim"]))
    if "agen_pengirim" in gt and isinstance(gt["agen_pengirim"], dict):
        if gt["agen_pengirim"].get("nama"): senders.append(("agen_pengirim.nama", gt["agen_pengirim"]["nama"]))
    if "dari" in gt and isinstance(gt["dari"], dict):
        if gt["dari"].get("nama"): senders.append(("dari.nama", gt["dari"]["nama"]))
    if "shipper" in gt:
        if isinstance(gt["shipper"], list) and len(gt["shipper"]) > 0: senders.append(("shipper", gt["shipper"][0]))
        elif isinstance(gt["shipper"], dict) and gt["shipper"].get("name"): senders.append(("shipper.name", gt["shipper"]["name"]))
        
    # Receivers
    receivers = []
    if "kepada" in gt and isinstance(gt["kepada"], dict):
        if gt["kepada"].get("toko"): receivers.append(("kepada.toko", gt["kepada"]["toko"]))
        if gt["kepada"].get("yth"): receivers.append(("kepada.yth", gt["kepada"]["yth"]))
        if gt["kepada"].get("nama"): receivers.append(("kepada.nama", gt["kepada"]["nama"]))
        if gt["kepada"].get("perusahaan"): receivers.append(("kepada.perusahaan", gt["kepada"]["perusahaan"]))
    if "penerima" in gt:
        if isinstance(gt["penerima"], dict):
            if gt["penerima"].get("nama"): receivers.append(("penerima.nama", gt["penerima"]["nama"]))
            if gt["penerima"].get("toko"): receivers.append(("penerima.toko", gt["penerima"]["toko"]))
        elif isinstance(gt["penerima"], str): receivers.append(("penerima", gt["penerima"]))
    if "consignee" in gt:
        if isinstance(gt["consignee"], list) and len(gt["consignee"]) > 0: receivers.append(("consignee", gt["consignee"][0]))
        elif isinstance(gt["consignee"], dict) and gt["consignee"].get("name"): receivers.append(("consignee.name", gt["consignee"]["name"]))
    if "tujuan_pengiriman" in gt and isinstance(gt["tujuan_pengiriman"], dict):
        if gt["tujuan_pengiriman"].get("nama"): receivers.append(("tujuan_pengiriman.nama", gt["tujuan_pengiriman"]["nama"]))
        
    # Match Sender
    matched_sender_k = ""
    for k, s_val in senders:
        s_clean = clean_alphanum(s_val)
        if len(s_clean) < 3: continue
        matched = False
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == s_clean or (len(pv_clean) >= 5 and (pv_clean in s_clean or s_clean in pv_clean)):
                df_party.at[idx, "label__sender_name"] = "1"
                df_party.at[idx, "label_notes"] = f"gt_field:{k}"
                matched = True
                break
        if matched:
            break
            
    # Match Receiver
    for k, r_val in receivers:
        r_clean = clean_alphanum(r_val)
        if len(r_clean) < 3: continue
        matched = False
        for idx, row in grp.iterrows():
            if df_party.at[idx, "label__sender_name"] == "1":
                continue
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == r_clean or (len(pv_clean) >= 5 and (pv_clean in r_clean or r_clean in pv_clean)):
                df_party.at[idx, "label__receiver_name"] = "1"
                existing = str(df_party.at[idx, "label_notes"]) if pd.notna(df_party.at[idx, "label_notes"]) else ""
                df_party.at[idx, "label_notes"] = (existing + f"; gt_field:{k}").strip("; ")
                matched = True
                break
        if matched:
            break

print(f"Labeled SJ Party: Sender={sum(df_party['label__sender_name'] == '1')}, Receiver={sum(df_party['label__receiver_name'] == '1')} across {df_party['m_document_id'].nunique()} docs.")
df_party.to_csv(path_party, index=False)


print("\n=== 4. Labeling candidates_SURAT_JALAN_address.csv ===")
path_addr = os.path.join(csv_dir, "candidates_SURAT_JALAN_address.csv")
df_addr = pd.read_csv(path_addr, dtype=str)
df_addr["label__sender_address"] = "0"
df_addr["label__receiver_address"] = "0"

for doc_id, grp in df_addr.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    # Senders address
    s_addrs = []
    if "kop" in gt and isinstance(gt["kop"], dict) and gt["kop"].get("alamat"):
        s_addrs.append(("kop.alamat", gt["kop"]["alamat"]))
    if "perusahaan" in gt and isinstance(gt["perusahaan"], dict) and gt["perusahaan"].get("alamat"):
        addr = " ".join(gt["perusahaan"]["alamat"]) if isinstance(gt["perusahaan"]["alamat"], list) else gt["perusahaan"]["alamat"]
        s_addrs.append(("perusahaan.alamat", addr))
    if "toko" in gt and isinstance(gt["toko"], dict) and gt["toko"].get("alamat"):
        s_addrs.append(("toko.alamat", gt["toko"]["alamat"]))
    if "shipper" in gt and isinstance(gt["shipper"], list) and len(gt["shipper"]) > 1:
        s_addrs.append(("shipper.address", " ".join(gt["shipper"][1:3])))
    if "pengirim" in gt and isinstance(gt["pengirim"], dict) and gt["pengirim"].get("alamat"):
        s_addrs.append(("pengirim.alamat", gt["pengirim"]["alamat"]))
    if "dari" in gt and isinstance(gt["dari"], dict) and gt["dari"].get("alamat"):
        s_addrs.append(("dari.alamat", gt["dari"]["alamat"]))
        
    # Receivers address
    r_addrs = []
    if "kepada" in gt and isinstance(gt["kepada"], dict):
        if gt["kepada"].get("di"): r_addrs.append(("kepada.di", gt["kepada"]["di"]))
        if gt["kepada"].get("alamat"):
            addr = " ".join(gt["kepada"]["alamat"]) if isinstance(gt["kepada"]["alamat"], list) else gt["kepada"]["alamat"]
            r_addrs.append(("kepada.alamat", addr))
    if "consignee" in gt and isinstance(gt["consignee"], list) and len(gt["consignee"]) > 1:
        r_addrs.append(("consignee.address", " ".join(gt["consignee"][1:3])))
    if "penerima" in gt and isinstance(gt["penerima"], dict) and gt["penerima"].get("alamat"):
        r_addrs.append(("penerima.alamat", gt["penerima"]["alamat"]))
    if "alamat_penerima" in gt and gt["alamat_penerima"]:
        r_addrs.append(("alamat_penerima", gt["alamat_penerima"]))
        
    for k, sa in s_addrs:
        sa_clean = clean_alphanum(sa)
        if len(sa_clean) < 6: continue
        matched = False
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if len(pv_clean) >= 6 and (pv_clean in sa_clean or sa_clean in pv_clean or (len(pv_clean) > 8 and pv_clean[:10] in sa_clean)):
                df_addr.at[idx, "label__sender_address"] = "1"
                df_addr.at[idx, "label_notes"] = f"gt_field:{k}"
                matched = True
                break
        if matched:
            break
            
    for k, ra in r_addrs:
        ra_clean = clean_alphanum(ra)
        if len(ra_clean) < 4: continue
        matched = False
        for idx, row in grp.iterrows():
            if df_addr.at[idx, "label__sender_address"] == "1":
                continue
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if len(pv_clean) >= 4 and (pv_clean in ra_clean or ra_clean in pv_clean or (len(pv_clean) > 8 and pv_clean[:10] in ra_clean)):
                df_addr.at[idx, "label__receiver_address"] = "1"
                existing = str(df_addr.at[idx, "label_notes"]) if pd.notna(df_addr.at[idx, "label_notes"]) else ""
                df_addr.at[idx, "label_notes"] = (existing + f"; gt_field:{k}").strip("; ")
                matched = True
                break
        if matched:
            break

print(f"Labeled SJ Address: Sender={sum(df_addr['label__sender_address'] == '1')}, Receiver={sum(df_addr['label__receiver_address'] == '1')}.")
df_addr.to_csv(path_addr, index=False)


print("\n=== 5. Labeling candidates_SURAT_JALAN_table.csv ===")
path_tab = os.path.join(csv_dir, "candidates_SURAT_JALAN_table.csv")
df_tab = pd.read_csv(path_tab, dtype=str)
df_tab["label__line_items"] = "0"

for doc_id, grp in df_tab.groupby("m_document_id"):
    best_idx = None
    for idx, row in grp.iterrows():
        rt = str(row["m_raw_text"]).lower()
        # Skip requisition, shipping terms, subtotal, and signature blocks
        if "requisitioner" in rt and "ship via" in rt:
            continue
        if "subtotal" in rt:
            continue
        if "penerima" in rt and "mengetahui" in rt:
            continue
        if "barang sudah diterima" in rt:
            continue
        if any(w in rt for w in ["nama barang", "nama produk", "deskripsi", "item #", "item no", "banyaknya"]):
            best_idx = idx
            break
            
    if best_idx is None:
        li_tables = grp[grp["m_evidence_role"] == "line_items"]
        if len(li_tables) > 0:
            best_idx = li_tables.index[0]
            
    if best_idx is not None:
        df_tab.at[best_idx, "label__line_items"] = "1"

print(f"Labeled SJ Table: {sum(df_tab['label__line_items'] == '1')} line item tables across {df_tab['m_document_id'].nunique()} docs.")
df_tab.to_csv(path_tab, index=False)

print("\n>>> ALL 5 SURAT JALAN CSV FILES SUCCESSFULLY LABELED! <<<")
