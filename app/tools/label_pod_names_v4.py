from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


EVIDENCE_NAME_OVERRIDES = {
    "pod_1_variasi_1": {"pihak_pertama_nama": "Andi Wijaya", "pihak_kedua_nama": "PT Nusantara Tech Solusindo"},
    "pod_1_variasi_2": {"pihak_pertama_nama": "Dr. Ratna Sari", "pihak_kedua_nama": "PT Medika Utama Indonesia"},
    "pod_2_variasi_1": {"pihak_pertama_nama": "YUS MULIANI, S.Kom", "pihak_kedua_nama": "KABUPATEN ACEH BARAT"},
    "pod_2_variasi_2": {"pihak_pertama_nama": "H. SURYANA, S.Pd"},
    "pod_3_variasi_1": {"pihak_pertama_nama": "ARIS SETIAWAN, S.T., M.T."},
    "pod_3_variasi_2": {"pihak_pertama_nama": "IR. H. KHAIRUL ANWAR, M.Si."},
}


def _clean(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def label(candidate_dir: Path, ground_truth: Path) -> dict[str, Any]:
    data = json.loads(ground_truth.read_text(encoding="utf-8"))
    fields: dict[str, dict[str, Any]] = {
        "pihak_pertama_nama": {"side": "pihak_pertama", "added": 0, "unmatched_gt": []},
        "pihak_kedua_nama": {"side": "pihak_kedua", "added": 0, "unmatched_gt": []},
    }
    variant_truth: dict[str, dict[str, Any]] = {}
    for document in data.get("documents", []):
        for variation in document.get("variations", []):
            key = str(variation.get("variant_name", "")).lower()
            variant_truth[key] = variation.get("fields", {})
    path = candidate_dir / "candidates_POD_party.csv"
    header, rows = _read(path)
    for field, config in fields.items():
        label_name = f"label__{field}"
        if label_name not in header:
            header.append(label_name)
        for variant, truth in variant_truth.items():
            document_id = f"pod--pdf--{variant}"
            expected = (truth.get(config["side"]) or {}).get("nama", "")
            override = EVIDENCE_NAME_OVERRIDES.get(variant, {}).get(field)
            target = override or expected
            if not target:
                continue
            document_rows = [row for row in rows if row.get("m_document_id", "").lower() == document_id]
            # V3 may already contain a human-confirmed correction based on evidence.
            if any(row.get(label_name) == "1" for row in document_rows):
                continue
            target_clean = _clean(target)
            matches = [row for row in document_rows if target_clean and (
                _clean(row.get("m_raw_text", "")) == target_clean
                or _clean(row.get("m_parsed_value", "")) == target_clean
                or (len(target_clean) >= 6 and target_clean in _clean(row.get("m_raw_text", "")))
            )]
            if not matches:
                config["unmatched_gt"].append({"document_id": document_id, "value": target, "source": "evidence_override" if override else "ground_truth"})
                continue
            # Prefer the line adjacent to the explicit Nama label; otherwise use the
            # earliest exact reading-order occurrence (usually the body name, not a signature repeat).
            matches.sort(key=lambda row: (-int(float(row.get("pt_near_name_label") or 0)), int(float(row.get("m_y1") or 0) * 10000), row.get("m_candidate_id", "")))
            selected = matches[0]
            selected[label_name] = "1"
            selected["label_notes"] = f"manual_ground_truth:{'evidence_override' if override else 'pod_ground_truth'}"
            config["added"] += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {field: {"added": value["added"], "unmatched_gt": value["unmatched_gt"]} for field, value in fields.items()}
    (candidate_dir / "pod_name_labeling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Add exact POD name labels to v4 candidates using ground truth and reviewed evidence overrides.")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    label(args.candidate_dir, args.ground_truth)


if __name__ == "__main__":
    main()
