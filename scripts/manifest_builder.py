'''
Manifest Builder

This script is used to build a manifest file for candidate generation. It scans the evidence from the specified source folder for PDF files, 
processes each document already generated using the Phase5DocumentPipeline, and compiles the results into a single manifest file.
'''

import json
from pathlib import Path

source_folder = Path("reports/out/all_types/evidence")
output_path = Path("reports/out/all_types/manifest.csv")

# store the manifest data
manifest_data = []

for source in source_folder.glob("*.json"):
    with open(source, 'r', encoding='utf-8') as f:
        evidence = json.load(f)
        # Here you can process the evidence as needed for your manifest
        # For example, you might want to extract certain fields or transform the data
        # Example manifest.csv: 
        # document_id,doc_type,evidence_json_path,vendor_group,template_group,is_base,base_document_id,page_start,page_end
        # invoice--a1,INVOICE,out\evidence\invoice--a1.json,UNKNOWN,invoice--a1,1,,,
        # invoice--a1-syn1,INVOICE,out\evidence\invoice--a1-syn1.json,UNKNOWN,invoice--a1,0,invoice--a1,,
        for segment in evidence.get("segments", []):
            manifest_data.append({
                "document_id": evidence.get("document_id", ""),
                "doc_type": segment.get("document_type", ""),
                "evidence_json_path": str(source),
                "vendor_group": "UNKNOWN",  
                "template_group": str(source.stem.lower().removesuffix("_variasi_1").removesuffix("_variasi_2")),
                "is_base": 1 if segment.get("segment_number", 0) == 1 else 0,
                "base_document_id": evidence.get("document_id", ""),
                "page_start": segment.get("start_page", 0),
                "page_end": segment.get("end_page", 0)
            })

# Write the manifest data to a CSV file
with open(output_path, 'w', encoding='utf-8') as f:
    # Write the header
    f.write("document_id,doc_type,evidence_json_path,vendor_group,template_group,is_base,base_document_id,page_start,page_end\n")
    for entry in manifest_data:
        f.write(f"{entry['document_id']},{entry['doc_type']},{entry['evidence_json_path']},{entry['vendor_group']},{entry['template_group']},{entry['is_base']},{entry['base_document_id']},{entry['page_start']},{entry['page_end']}\n")

# Print a message indicating that the manifest has been created
print(f"Manifest file created at {output_path}")