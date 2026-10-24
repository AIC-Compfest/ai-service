import json, os, re
import pandas as pd

gt_path = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset\surat jalan\ground_truth_surat_jalan.json"
with open(gt_path, encoding="utf-8") as f:
    sj_docs = json.load(f)["documents"]

sj_gt = {d["file"].replace(".pdf", "").lower(): d["ground_truth"] for d in sj_docs}
csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

def clean_str(s):
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
    # "17/06/2003"
    m3 = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(dt_str))
    if m3:
        p1, p2, y = m3.groups()
        # could be DD/MM/YYYY or MM/DD/YYYY
        if int(p1) > 12:
            return f"{y}-{int(p2):02d}-{int(p1):02d}"
        elif int(p2) > 12:
            return f"{y}-{int(p1):02d}-{int(p2):02d}"
        else:
            return f"{y}-{int(p2):02d}-{int(p1):02d}"
    return ""

# Test ID matching
print("=== Testing SURAT JALAN ID ===")
df_id = pd.read_csv(os.path.join(csv_dir, "candidates_SURAT_JALAN_id.csv"), dtype=str)
matched_id = 0
for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    # Try all loose ID keys in priority order
    id_candidates = [
        ("nomor", gt.get("nomor")),
        ("no_surat_jalan", gt.get("no_surat_jalan")),
        ("no_dokumen", gt.get("no_dokumen")),
        ("no", gt.get("no")),
        ("no_invoice", gt.get("no_invoice")),
        ("no_resi", gt.get("no_resi")),
        ("shipment_no", gt.get("shipment_no")),
        ("nomor_invoice", gt.get("nomor_invoice")),
        ("invoice_no", gt.get("invoice_no")),
        ("po_no", gt.get("po_no")),
        ("nomor_order", gt.get("nomor_order")),
    ]
    
    found = False
    for k, val in id_candidates:
        if not val:
            continue
        val_clean = clean_str(val)
        m = grp[grp["m_parsed_value"].apply(clean_str) == val_clean]
        if len(m) > 0:
            matched_id += 1
            found = True
            break
        # also check substring
        m_sub = grp[grp["m_parsed_value"].apply(clean_str).apply(lambda x: len(x) >= 6 and (x in val_clean or val_clean in x))]
        if len(m_sub) > 0:
            matched_id += 1
            found = True
            break
    if not found:
        print(f"  Unmatched SJ ID for {clean_doc}: GT keys={[k for k, v in id_candidates if v]}, candidates={grp['m_parsed_value'].tolist()[:4]}")

print(f"SJ ID matched: {matched_id} / {df_id['m_document_id'].nunique()}")

# Test Date matching
print("\n=== Testing SURAT JALAN Date ===")
df_date = pd.read_csv(os.path.join(csv_dir, "candidates_SURAT_JALAN_date.csv"), dtype=str)
matched_date = 0
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
    found = False
    for k, val in dt_candidates:
        if not val:
            continue
        iso = parse_date_to_iso(val)
        if not iso:
            continue
        m = grp[grp["m_parsed_norm"] == iso]
        if len(m) > 0:
            matched_date += 1
            found = True
            break
    if not found:
        print(f"  Unmatched SJ Date for {clean_doc}: GT={[v for k, v in dt_candidates if v]}, parsed_norms={grp['m_parsed_norm'].tolist()}")

print(f"SJ Date matched: {matched_date} / {df_date['m_document_id'].nunique()}")

# Test Party matching
print("\n=== Testing SURAT JALAN Party ===")
df_party = pd.read_csv(os.path.join(csv_dir, "candidates_SURAT_JALAN_party.csv"), dtype=str)
matched_sender, matched_rec = 0, 0
for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("surat jalan--pdf--", "").lower()
    gt = sj_gt.get(clean_doc, {})
    
    # Senders
    sender_vals = []
    if "kop" in gt and isinstance(gt["kop"], dict):
        sender_vals.append(gt["kop"].get("nama_kelompok") or gt["kop"].get("nama"))
    if "distributor" in gt: sender_vals.append(gt.get("distributor"))
    if "toko" in gt and isinstance(gt["toko"], dict): sender_vals.append(gt["toko"].get("nama"))
    if "pengirim" in gt:
        if isinstance(gt["pengirim"], dict): sender_vals.append(gt["pengirim"].get("nama") or gt["pengirim"].get("perusahaan"))
        else: sender_vals.append(gt.get("pengirim"))
    if "agen_pengirim" in gt and isinstance(gt["agen_pengirim"], dict): sender_vals.append(gt["agen_pengirim"].get("nama"))
    if "dari" in gt and isinstance(gt["dari"], dict): sender_vals.append(gt["dari"].get("nama"))
    
    # Receivers
    rec_vals = []
    if "kepada" in gt and isinstance(gt["kepada"], dict):
        rec_vals.append(gt["kepada"].get("toko") or gt["kepada"].get("yth") or gt["kepada"].get("nama"))
    if "penerima" in gt:
        if isinstance(gt["penerima"], dict): rec_vals.append(gt["penerima"].get("nama") or gt["penerima"].get("toko"))
        else: rec_vals.append(gt.get("penerima"))
    if "consignee" in gt and isinstance(gt["consignee"], dict): rec_vals.append(gt["consignee"].get("nama") or gt["consignee"].get("name"))
    if "tujuan_pengiriman" in gt and isinstance(gt["tujuan_pengiriman"], dict): rec_vals.append(gt["tujuan_pengiriman"].get("nama"))
    
    # Check matching
    for s in sender_vals:
        if not s: continue
        sc = clean_str(s)
        m = grp[grp["m_parsed_value"].apply(clean_str).apply(lambda x: len(x) >= 4 and (x in sc or sc in x))]
        if len(m) > 0:
            matched_sender += 1
            break
            
    for r in rec_vals:
        if not r: continue
        rc = clean_str(r)
        m = grp[grp["m_parsed_value"].apply(clean_str).apply(lambda x: len(x) >= 4 and (x in rc or rc in x))]
        if len(m) > 0:
            matched_rec += 1
            break

print(f"SJ Party matched: Sender={matched_sender}, Receiver={matched_rec} / {df_party['m_document_id'].nunique()}")

# Test Table matching
print("\n=== Testing SURAT JALAN Table ===")
df_tab = pd.read_csv(os.path.join(csv_dir, "candidates_SURAT_JALAN_table.csv"), dtype=str)
print("SJ Table count:", len(df_tab), "across", df_tab["m_document_id"].nunique(), "docs")
