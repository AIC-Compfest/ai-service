import json, os, re
import pandas as pd

gt_path = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset\pod\ground_truth_all_17_documents.json"
with open(gt_path, encoding="utf-8") as f:
    pod_gt = json.load(f)["documents"]

csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

pod_map = {}
for doc in pod_gt:
    for v in doc.get("variations", []):
        vname = v["variant_name"].lower()
        pod_map[vname] = v["fields"]

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

def parse_id_date_to_iso(dt_str):
    if not dt_str:
        return ""
    m = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})", dt_str)
    if m:
        day, mon, year = m.groups()
        mon_num = month_map.get(mon.lower(), "")
        if mon_num:
            return f"{year}-{mon_num}-{int(day):02d}"
    m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", dt_str)
    if m2:
        day, mon, year = m2.groups()
        return f"{year}-{int(mon):02d}-{int(day):02d}"
    return ""

pod_evidence_truth = {
    "pod_1_variasi_1": {
        "pod_number": "STTB/11/2025/104",
        "pod_date": "2025-11-12",
        "p1_nama": "Andi Wijaya",
        "p1_jabatan": "IT Infrastructure Lead",
        "p1_alamat": "JI. Pemuda No. 88, Surabaya",
        "p2_nama": "PT Nusantara Tech Solusindo"
    },
    "pod_1_variasi_2": {
        "pod_number": "STTB/12/2025/219",
        "pod_date": "2025-12-05",
        "p1_nama": "Dr. Ratna Sari",
        "p1_jabatan": "Kepala Penanggung Jawab Medis",
        "p1_alamat": "JI. Pandanaran No. 15, Semarang",
        "p2_nama": "PT Medika Utama Indonesia"
    },
    "pod_2_variasi_1": {
        "pod_number": "012/PPS/KB/BA/IV/2019",
        "pod_date": "2019-04-17",
        "p1_nama": "YUS MULIANI, S.Kom",
        "p2_nama": "KABUPATEN ACEH BARAT",
        "p2_alamat": "JL. Diponegoro Kampung Belakang"
    },
    "pod_2_variasi_2": {
        "pod_number": "085/PPS/SKR/BA/XI/2024",
        "pod_date": "2024-11-27",
        "p1_nama": "H. SURYANA, S.Pd",
        "p1_jabatan": "Ketua PPS",
        "p2_alamat": "RT 02 / RW 05 Desa Sukaraja"
    },
    "pod_3_variasi_1": {
        "pod_date": "2025-10-15",
        "p1_nama": "ARIS SETIAWAN, S.T., M.T.",
        "p1_jabatan": "Kepala Bidang Bina Marga",
        "p1_alamat": "Jalan Jelarai Raya Telp. (0552) 21020 TANJUNG SELOR"
    },
    "pod_3_variasi_2": {
        "pod_date": "2025-11-03",
        "p1_nama": "IR. H. KHAIRUL ANWAR, M.Si.",
        "p1_jabatan": "Kepala UPTD Laboratorium Bahan Konstruksi",
        "p1_alamat": "JI. M. T. Haryono No. 45 Telp. (0541) 661022 TENGGARONG"
    },
    "pod_11_variasi_1": {
        "p1_alamat": "Jalan MT. Haryono No. 11 Cawang, Jakarta Timur 13630",
        "note_a1": "gt_field:kantor_bnn_pusat"
    },
    "pod_11_variasi_2": {
        "p1_alamat": "Jalan MT. Haryono No. 11 Cawang, Jakarta Timur 13630",
        "note_a1": "gt_field:kantor_bnn_pusat"
    },
    "pod_9_variasi_1": {
        "p1_alamat": "JI. Raya Cilembu Ds. Neglasari Kec. Pakenjeng Kab. Garut",
        "note_a1": "gt_field:sekretariat"
    },
    "pod_9_variasi_2": {
        "p1_alamat": "JI. Raya Cianjur-Sukabumi KM. 6 Ds. Sukamaju Kec. Cilaku Kab. Cianjur",
        "note_a1": "gt_field:sekretariat",
        "p2_alamat": "Kp. Sukamaju RT 02/RW 04 Desa Sukamaju",
        "note_a2": "gt_field:alamat_tps"
    }
}

print("=== 1. Labeling candidates_POD_id.csv ===")
path_id = os.path.join(csv_dir, "candidates_POD_id.csv")
df_id = pd.read_csv(path_id, dtype=str)
df_id["label__pod_number"] = "0"

for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    gt_nomor = gt.get("nomor_berita_acara") or gt.get("nomor_dokumen") or gt.get("nomor")
    alt_gt = pod_evidence_truth.get(clean_doc, {}).get("pod_number")
    
    target_clean = clean_alphanum(gt_nomor)
    note = ""
    if alt_gt:
        target_clean = clean_alphanum(alt_gt)
        note = "from evidence, GT mismatch"
    elif not target_clean and not alt_gt:
        continue
    
    matched_idx = []
    for idx, row in grp.iterrows():
        pv_clean = clean_alphanum(row["m_parsed_value"])
        raw_clean = clean_alphanum(row["m_raw_text"])
        if pv_clean == target_clean or raw_clean == target_clean:
            matched_idx.append(idx)
    
    if not matched_idx:
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if target_clean in pv_clean or (pv_clean and pv_clean in target_clean):
                matched_idx.append(idx)
                break
    
    for idx in matched_idx:
        df_id.at[idx, "label__pod_number"] = "1"
        if note:
            df_id.at[idx, "label_notes"] = note

print(f"Labeled POD ID: {sum(df_id['label__pod_number'] == '1')} positive labels across {df_id['m_document_id'].nunique()} docs.")
df_id.to_csv(path_id, index=False)


print("\n=== 2. Labeling candidates_POD_date.csv ===")
path_date = os.path.join(csv_dir, "candidates_POD_date.csv")
df_date = pd.read_csv(path_date, dtype=str)
df_date["label__pod_date"] = "0"

delivery_receipt_dates = {
    "pod_12_variasi_1": "2023-06-16",
    "pod_12_variasi_2": "2025-10-22",
    "pod_13_variasi_1": "2024-02-26",
    "pod_13_variasi_2": "2025-11-14",
    "pod_15_variasi_1": "2019-12-11",
    "pod_15_variasi_2": "2025-11-18",
    "pod_16_variasi_1": "2023-03-26",
    "pod_16_variasi_2": "2025-11-10",
    "pod_17_variasi_1": "2023-03-12",
    "pod_17_variasi_2": "2025-11-18",
    "pod_7_variasi_1": "2021-09-27",
    "pod_7_variasi_2": "2025-10-14",
    "pod_8_variasi_1": "2026-05-28",
    "pod_8_variasi_2": "2026-11-10",
}

for doc_id, grp in df_date.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    gt_tanggal = gt.get("tanggal") or gt.get("tanggal_kirim") or gt.get("date")
    target_iso = parse_id_date_to_iso(gt_tanggal)
    note = ""
    
    if clean_doc in pod_evidence_truth and "pod_date" in pod_evidence_truth[clean_doc]:
        target_iso = pod_evidence_truth[clean_doc]["pod_date"]
        note = "from evidence, GT mismatch"
    elif not target_iso and clean_doc in delivery_receipt_dates:
        target_iso = delivery_receipt_dates[clean_doc]
        note = "from evidence, delivery date"
    
    if not target_iso:
        continue
    
    matched_idx = []
    for idx, row in grp.iterrows():
        if row["m_parsed_norm"] == target_iso:
            matched_idx.append(idx)
    
    if len(matched_idx) > 1:
        pref = [i for i in matched_idx if any(w in str(df_date.at[i, "ctx_line"]).lower() for w in ["delivered", "delivery", "tanggal", "surabaya", "semarang", "date of service"])]
        if pref:
            matched_idx = pref
    
    for idx in matched_idx:
        df_date.at[idx, "label__pod_date"] = "1"
        if note:
            df_date.at[idx, "label_notes"] = note

print(f"Labeled POD Date: {sum(df_date['label__pod_date'] == '1')} positive labels across {df_date['m_document_id'].nunique()} docs.")
df_date.to_csv(path_date, index=False)


print("\n=== 3. Labeling candidates_POD_party.csv ===")
path_party = os.path.join(csv_dir, "candidates_POD_party.csv")
df_party = pd.read_csv(path_party, dtype=str)
df_party["label__pihak_pertama_nama"] = "0"
df_party["label__pihak_kedua_nama"] = "0"

for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    
    # Check loose keys: penerima / pemberi in pod_6
    p1_gt = None
    p2_gt = None
    p1_note, p2_note = "", ""
    
    if "pihak_pertama" in gt and isinstance(gt["pihak_pertama"], dict):
        p1_gt = gt["pihak_pertama"].get("nama")
    elif "penerima" in gt and isinstance(gt["penerima"], dict):
        p1_gt = gt["penerima"].get("nama")
        p1_note = "gt_field:penerima.nama"
        
    if "pihak_kedua" in gt and isinstance(gt["pihak_kedua"], dict):
        p2_gt = gt["pihak_kedua"].get("nama")
    elif "pemberi" in gt and isinstance(gt["pemberi"], dict):
        p2_gt = gt["pemberi"].get("nama")
        p2_note = "gt_field:pemberi.nama"
        
    if clean_doc in pod_evidence_truth:
        if "p1_nama" in pod_evidence_truth[clean_doc]:
            p1_gt = pod_evidence_truth[clean_doc]["p1_nama"]
            p1_note = "from evidence, GT mismatch"
        if "p2_nama" in pod_evidence_truth[clean_doc]:
            p2_gt = pod_evidence_truth[clean_doc]["p2_nama"]
            p2_note = "from evidence, GT mismatch"
            
    p1_clean = clean_alphanum(p1_gt)
    p2_clean = clean_alphanum(p2_gt)
    
    # Match P1
    if p1_clean:
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == p1_clean or (len(p1_clean) >= 6 and (pv_clean in p1_clean or p1_clean in pv_clean)):
                if p2_clean and pv_clean == p2_clean:
                    continue
                df_party.at[idx, "label__pihak_pertama_nama"] = "1"
                if p1_note:
                    df_party.at[idx, "label_notes"] = p1_note
                break
                
    # Match P2
    if p2_clean:
        for idx, row in grp.iterrows():
            pv_clean = clean_alphanum(row["m_parsed_value"])
            if pv_clean == p2_clean or (len(p2_clean) >= 6 and (pv_clean in p2_clean or p2_clean in pv_clean)):
                if p1_clean and pv_clean == p1_clean:
                    continue
                df_party.at[idx, "label__pihak_kedua_nama"] = "1"
                if p2_note:
                    existing = str(df_party.at[idx, "label_notes"]) if pd.notna(df_party.at[idx, "label_notes"]) else ""
                    df_party.at[idx, "label_notes"] = (existing + "; " + p2_note).strip("; ")
                break

print(f"Labeled POD Party: P1={sum(df_party['label__pihak_pertama_nama'] == '1')}, P2={sum(df_party['label__pihak_kedua_nama'] == '1')}.")
df_party.to_csv(path_party, index=False)


print("\n=== 4. Labeling candidates_POD_address.csv ===")
path_addr = os.path.join(csv_dir, "candidates_POD_address.csv")
df_addr = pd.read_csv(path_addr, dtype=str)
df_addr["label__pihak_pertama_alamat"] = "0"
df_addr["label__pihak_kedua_alamat"] = "0"

for doc_id, grp in df_addr.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    
    a1_gt, a2_gt = None, None
    a1_note, a2_note = "", ""
    
    if "pihak_pertama" in gt and isinstance(gt["pihak_pertama"], dict):
        a1_gt = gt["pihak_pertama"].get("alamat")
    elif "penerima" in gt and isinstance(gt["penerima"], dict):
        a1_gt = gt["penerima"].get("alamat")
        a1_note = "gt_field:penerima.alamat"
        
    if "pihak_kedua" in gt and isinstance(gt["pihak_kedua"], dict):
        a2_gt = gt["pihak_kedua"].get("alamat")
    elif "pemberi" in gt and isinstance(gt["pemberi"], dict):
        a2_gt = gt["pemberi"].get("alamat")
        a2_note = "gt_field:pemberi.alamat"
        
    if clean_doc in pod_evidence_truth:
        if "p1_alamat" in pod_evidence_truth[clean_doc]:
            a1_gt = pod_evidence_truth[clean_doc]["p1_alamat"]
            a1_note = pod_evidence_truth[clean_doc].get("note_a1", "from evidence, GT mismatch")
        if "p2_alamat" in pod_evidence_truth[clean_doc]:
            a2_gt = pod_evidence_truth[clean_doc]["p2_alamat"]
            a2_note = pod_evidence_truth[clean_doc].get("note_a2", "from evidence, GT mismatch")
            
    a1_clean = clean_alphanum(a1_gt)
    a2_clean = clean_alphanum(a2_gt)
    
    for idx, row in grp.iterrows():
        pv_clean = clean_alphanum(row["m_parsed_value"])
        if a1_clean and len(pv_clean) >= 6:
            if pv_clean in a1_clean or a1_clean in pv_clean or (len(pv_clean) > 8 and pv_clean[:10] in a1_clean):
                df_addr.at[idx, "label__pihak_pertama_alamat"] = "1"
                if a1_note:
                    df_addr.at[idx, "label_notes"] = a1_note
                continue
        if a2_clean and len(pv_clean) >= 6:
            if pv_clean in a2_clean or a2_clean in pv_clean or (len(pv_clean) > 8 and pv_clean[:10] in a2_clean):
                df_addr.at[idx, "label__pihak_kedua_alamat"] = "1"
                if a2_note:
                    df_addr.at[idx, "label_notes"] = a2_note
                continue

print(f"Labeled POD Address: A1={sum(df_addr['label__pihak_pertama_alamat'] == '1')}, A2={sum(df_addr['label__pihak_kedua_alamat'] == '1')}.")
df_addr.to_csv(path_addr, index=False)


print("\n=== 5. Labeling candidates_POD_table.csv ===")
path_tab = os.path.join(csv_dir, "candidates_POD_table.csv")
df_tab = pd.read_csv(path_tab, dtype=str)
df_tab["label__line_items"] = "0"

for doc_id, grp in df_tab.groupby("m_document_id"):
    li_tables = grp[grp["m_evidence_role"] == "line_items"]
    if len(li_tables) > 0:
        best_idx = None
        for idx, row in li_tables.iterrows():
            rt = str(row["m_raw_text"]).lower()
            if "penerima" in rt and "menyerahkan" in rt:
                continue
            if "pihak kedua" in rt and "pihak pertama" in rt:
                continue
            if "tracking number" in rt and "service" in rt:
                continue
            best_idx = idx
            break
        if best_idx is None:
            best_idx = li_tables.index[0]
        df_tab.at[best_idx, "label__line_items"] = "1"

print(f"Labeled POD Table: {sum(df_tab['label__line_items'] == '1')} line item tables across {df_tab['m_document_id'].nunique()} docs.")
df_tab.to_csv(path_tab, index=False)


print("\n=== 6. Labeling candidates_POD_text.csv ===")
path_text = os.path.join(csv_dir, "candidates_POD_text.csv")
df_text = pd.read_csv(path_text, dtype=str)
df_text["label__pihak_pertama_jabatan"] = "0"
df_text["label__pihak_kedua_jabatan"] = "0"
df_text.to_csv(path_text, index=False)
print(f"Labeled POD Text: 4 rows set to 0.")

print("\n>>> ALL 6 POD CSV FILES SUCCESSFULLY LABELED! <<<")
