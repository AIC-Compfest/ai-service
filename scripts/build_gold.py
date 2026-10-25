"""Build the dataset manifest, GT row-content gold, structural bootstrap gold and constructed fixtures.

    .\\.venv\\Scripts\\python.exe scripts\\build_gold.py [--dataset-root ..\\DATASET\\Dataset] [--out gold]

Outputs (all labelled synthetic, reviewed=false; none is human gold):
  gold/manifest.json                      path, sha256, type, template, variation, split, GT status
  gold/gt_field_inventory.json            GT table paths, key frequencies, canonical mapping
  gold/row_content/<id>.json              GT rows as scalar value lists (content only)
  gold/regression/<id>.json               structural gold, source_label=synthetic_grounded
  gold/bootstrap_report.json              grounding success/failure per document and table
  gold/constructed/<name>.json            exact gold of the constructed fixtures
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from app.tools.table_eval.grounding import ground_document  # noqa: E402
from app.tools.table_eval.manifest import build_manifest, inventory, load_gt, pdf_key, row_content_gold  # noqa: E402
from app.tools.table_eval.model import write_json  # noqa: E402
from app.tools.table_eval.synthetic import build_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(os.environ.get("VERIFLOW_DATASET_ROOT", AI_ROOT.parent / "DATASET" / "Dataset"))
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=AI_ROOT / "gold")
    args = parser.parse_args()
    root, out = args.dataset_root.resolve(), args.out

    gt = load_gt(root)
    manifest = build_manifest(root)
    write_json(out / "manifest.json", manifest)
    write_json(out / "gt_field_inventory.json", inventory(gt, AI_ROOT / "config" / "gold" / "gt_field_aliases.yaml"))
    for doc_id, content in row_content_gold(manifest, gt).items():
        write_json(out / "row_content" / f"{doc_id}.json", content)

    report, status = [], Counter()
    for entry in manifest["entries"]:
        if not entry["has_gt"] or entry["source_kind"] != "pdf":
            continue
        doc = gt[(entry["doc_type"], pdf_key(Path(entry["path"]).name))]
        pages, details = ground_document(root / entry["path"], doc.tables, doc.totals, entry["id"])
        for detail in details:
            status[detail["status"]] += 1
        report.append({"id": entry["id"], "doc_type": entry["doc_type"], "split": entry["split"], "tables": details})
        if not pages:
            continue
        write_json(
            out / "regression" / f"{entry['id']}.json",
            {
                "document_id": entry["id"],
                "source": {"relative_path": entry["path"], "sha256": entry["sha256"], "source_kind": "pdf"},
                "document_type": entry["doc_type"],
                "source_label": "synthetic_grounded",
                "reviewed": False,
                "review": {"schema_version": "1", "prepared_by": "table_eval.grounding", "prepared_at_utc": "", "approved_by": "", "approved_at_utc": "", "notes": "Bootstrapped from GT values and PDF words; not human reviewed."},
                "pages_exhaustive": False,
                "pages": [{"page_number": number, "tables": tables} for number, tables in sorted(pages.items())],
            },
        )
    write_json(out / "bootstrap_report.json", {"summary": dict(status), "documents": report})
    for fixture in build_all():
        write_json(out / "constructed" / f"{fixture.name}.json", fixture.gold)
    print("manifest:", manifest["summary"])
    print("grounding table status:", dict(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
