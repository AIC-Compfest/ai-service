import json, os, re
import pandas as pd

gt_path = r"c:\GitRepos\AIC-2026\ai-service\reports\Full_Dataset\pod\ground_truth_all_17_documents.json"
with open(gt_path, encoding="utf-8") as f:
    pod_gt = json.load(f)["documents"]

pod_map = {}
for doc in pod_gt:
    for v in doc.get("variations", []):
        vname = v["variant_name"].lower()
        pod_map[vname] = v["fields"]

csv_dir = r"c:\GitRepos\AIC-2026\ai-service\reports\out\all_types\candidates_v3"

def clean_str(s):
    if not s or pd.isna(s):
        return ""
    # remove punctuation and whitespace
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()

def clean_tokens(s):
    if not s or pd.isna(s):
        return set()
    return set(re.findall(r"\b\w+\b", str(s).lower()))

# Month mapping for Indonesian dates
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
    return ""

print("=== Testing POD ID ===")
df_id = pd.read_csv(os.path.join(csv_dir, "candidates_POD_id.csv"), dtype=str)
matched_id = 0
for doc_id, grp in df_id.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    gt_nomor = gt.get("nomor_berita_acara", "")
    gt_clean = clean_str(gt_nomor)
    matches = grp[grp["m_parsed_value"].apply(clean_str) == gt_clean]
    if len(matches) > 0:
        matched_id += 1
    else:
        # Check partial or raw_text
        alt = grp[grp["m_raw_text"].apply(clean_str).str.contains(gt_clean, na=False) | grp["m_parsed_value"].apply(lambda x: gt_clean in clean_str(x))]
        print(f"  Doc {doc_id}: GT '{gt_nomor}' (clean: {gt_clean}). Alt matches: {len(alt)}")
print(f"POD ID matched docs: {matched_id} / {df_id['m_document_id'].nunique()}")

print("\n=== Testing POD Date ===")
df_date = pd.read_csv(os.path.join(csv_dir, "candidates_POD_date.csv"), dtype=str)
matched_date = 0
for doc_id, grp in df_date.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    gt_tanggal = gt.get("tanggal", "")
    gt_iso = parse_id_date_to_iso(gt_tanggal)
    matches = grp[grp["m_parsed_norm"] == gt_iso]
    if len(matches) > 0:
        matched_date += 1
    else:
        print(f"  Doc {doc_id}: GT '{gt_tanggal}' -> ISO '{gt_iso}'. Parsed norm vals: {grp['m_parsed_norm'].tolist()}")
print(f"POD Date matched docs: {matched_date} / {df_date['m_document_id'].nunique()}")

print("\n=== Testing POD Party ===")
df_party = pd.read_csv(os.path.join(csv_dir, "candidates_POD_party.csv"), dtype=str)
matched_p1, matched_p2 = 0, 0
for doc_id, grp in df_party.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    p1_name = gt.get("pihak_pertama", {}).get("nama", "")
    p2_name = gt.get("pihak_kedua", {}).get("nama", "")
    
    # Matching
    m1 = grp[grp["m_parsed_value"].apply(clean_str) == clean_str(p1_name)]
    m2 = grp[grp["m_parsed_value"].apply(clean_str) == clean_str(p2_name)]
    if len(m1) > 0: matched_p1 += 1
    if len(m2) > 0: matched_p2 += 1
print(f"POD Party exact match: P1={matched_p1}, P2={matched_p2} / {df_party['m_document_id'].nunique()}")

print("\n=== Testing POD Address ===")
df_addr = pd.read_csv(os.path.join(csv_dir, "candidates_POD_address.csv"), dtype=str)
matched_a1, matched_a2 = 0, 0
for doc_id, grp in df_addr.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    a1 = gt.get("pihak_pertama", {}).get("alamat", "")
    a2 = gt.get("pihak_kedua", {}).get("alamat", "")
    m1 = grp[grp["m_parsed_value"].apply(clean_str).str.contains(clean_str(a1[:15]), na=False)]
    m2 = grp[grp["m_parsed_value"].apply(clean_str).str.contains(clean_str(a2[:15]), na=False)]
    if len(m1) > 0: matched_a1 += 1
    if len(m2) > 0: matched_a2 += 1
print(f"POD Address matches: A1={matched_a1}, A2={matched_a2} / {df_addr['m_document_id'].nunique()}")

print("\n=== Testing POD Text ===")
df_text = pd.read_csv(os.path.join(csv_dir, "candidates_POD_text.csv"), dtype=str)
for doc_id, grp in df_text.groupby("m_document_id"):
    clean_doc = doc_id.replace("pod--pdf--", "").lower()
    gt = pod_map.get(clean_doc, {})
    print(f"Text Doc {doc_id}: P1 jab='{gt.get('pihak_pertama', {}).get('jabatan')}', P2 jab='{gt.get('pihak_kedua', {}).get('jabatan')}'")
    print(grp[['m_parsed_value', 'ctx_line']])

print("\n=== Testing POD Table ===")
df_tab = pd.read_csv(os.path.join(csv_dir, "candidates_POD_table.csv"), dtype=str)
print("POD Table evidence roles:")
print(df_tab[["m_document_id", "m_table_id", "m_evidence_role", "m_parsed_value"]].head(10))
