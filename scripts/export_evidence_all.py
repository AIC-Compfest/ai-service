import json
from pathlib import Path

from app.services.pipeline import Phase5DocumentPipeline

source_folder = Path("reports/Full_Dataset")
sources = list(source_folder.glob("**/*.pdf")) # Get all PDF files in the source folder and its subfolders

output_folder = Path("reports/out/all_types/evidence")
output_folder.parent.mkdir(parents=True, exist_ok=True)

# Generate evidence for each source PDF and save it as a JSON file in the output folder
for source in sources:
	output = output_folder / f"{source.stem}.json"
	output.parent.mkdir(parents=True, exist_ok=True)
	
	evidence = Phase5DocumentPipeline().process_document(
		source,
		document_id=f"evidence_{source.stem}",
	)
 
	output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Wrote {output}")
