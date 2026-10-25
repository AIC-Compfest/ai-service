"""Dataset manifest, ground-truth schema adapters, template-based split and field inventory.

The four GT files have four different shapes. Each adapter only knows its SHAPE (where the file name and the
table lists live). No adapter contains vendor, template or file-name specific rules beyond the documented
mapping ``invoice_variasi_N.pdf`` -> template ``contoh-invoice-<k>`` -> ``invoice_<k>_variasi_N.pdf``.

GT is SYNTHETIC (generated together with the PDFs). It is not human gold.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from .model import read_json

FOLDERS = {"invoice": "INVOICE", "PoD": "POD", "rate card": "RATE_AGREEMENT", "surat jalan": "SURAT_JALAN"}
SLUGS = {"invoice": "invoice", "PoD": "pod", "rate card": "rate_card", "surat jalan": "surat_jalan"}
TABLE_PATHS = {
    "INVOICE": ["line_items"],
    "POD": ["fields.rincian_barang", "fields.items", "fields.berkas", "fields.logistik_items", "fields.material"],
    "SURAT_JALAN": ["ground_truth.barang", "ground_truth.daftar_barang", "ground_truth.detail_order", "ground_truth.items", "ground_truth.produk"],
    "RATE_AGREEMENT": ["items", "fields.carrier_fees", "fields.charges", "fields.payment.items", "fields.rate_details", "fields.stops"],
}
TOTAL_KEY_RE = re.compile(r"(?i)(total|subtotal|sub_total|vat|tax|ppn|pph|grand|discount|net_)")
HOLDOUT_PERCENT, VAL_PERCENT = 20, 30  # dev gets the remaining 50 percent


def pdf_key(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def template_of(doc_type: str, key: str) -> tuple[str | None, int | None]:
    """Template id and variation from a normalized file name (``invoice_7_variasi_2.pdf``)."""
    match = re.match(r"^(?:invoice|pod|surat_jalan|rate_card)_(\d+)_variasi_(\d+)", key)
    if not match:
        return None, None
    return f"{doc_type}:{int(match.group(1))}", int(match.group(2))


def split_of(template: str | None) -> str:
    """Deterministic, template-level split so that variation 1 and 2 never straddle splits."""
    if template is None:
        return "smoke"
    bucket = int(hashlib.sha256(template.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < HOLDOUT_PERCENT:
        return "holdout"
    if bucket < HOLDOUT_PERCENT + VAL_PERCENT:
        return "val"
    return "dev"


@dataclass
class GtDoc:
    key: str
    doc_type: str
    body: dict
    tables: list[dict] = field(default_factory=list)  # [{"path": str, "rows": [[(key, value), ...]]}]
    totals: list[str] = field(default_factory=list)
    gt_name: str = ""


def _get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _scalar_pairs(row: dict) -> list[tuple[str, Any]]:
    return [(k, v) for k, v in row.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip() != ""]


def _collect_totals(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and TOTAL_KEY_RE.search(str(key)):
                out.append(str(value))
            elif isinstance(value, dict):
                _collect_totals(value, out)


def _make_gt(doc_type: str, key: str, body: dict, name: str) -> GtDoc:
    gt = GtDoc(key=key, doc_type=doc_type, body=body, gt_name=name)
    for path in TABLE_PATHS[doc_type]:
        value = _get(body, path)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            gt.tables.append({"path": path, "rows": [_scalar_pairs(item) for item in value]})
    _collect_totals(body, gt.totals)
    return gt


def load_gt(dataset_root: Path) -> dict[tuple[str, str], GtDoc]:
    """Map ``(doc_type, normalized_pdf_name)`` to its GT. Four shape adapters, one per GT family."""
    gt: dict[tuple[str, str], GtDoc] = {}
    # Invoice: documents[].file_name; 'invoice_variasi_N.pdf' is mapped through template_source.
    for path in sorted((dataset_root / "invoice").glob("ground_truth_variasi_*.json")):
        data = read_json(path)
        variation = data.get("variation")
        for doc in data["documents"]:
            key = pdf_key(doc["file_name"])
            if re.fullmatch(r"invoice_variasi_\d+\.pdf", key):
                match = re.search(r"(\d+)\s*$", doc.get("template_source", ""))
                if match:
                    key = f"invoice_{match.group(1)}_variasi_{variation}.pdf"
            gt[("INVOICE", key)] = _make_gt("INVOICE", key, doc, doc["file_name"])
    # PoD: documents[].variations[].file
    pod = read_json(dataset_root / "PoD" / "ground_truth_all_17_documents.json")
    for doc in pod["documents"]:
        for variation in doc["variations"]:
            key = pdf_key(variation["file"])
            gt[("POD", key)] = _make_gt("POD", key, variation, variation["file"])
    # Surat jalan: documents[].file
    sj = read_json(dataset_root / "surat jalan" / "ground_truth_surat_jalan.json")
    for doc in sj["documents"]:
        key = pdf_key(doc["file"])
        gt[("SURAT_JALAN", key)] = _make_gt("SURAT_JALAN", key, doc, doc["file"])
    # Rate card 1-7: {rate_card_N: {variasi_V: {...}}}; 8-15: documents[].file_name
    rc17 = read_json(dataset_root / "rate card" / "ground_truth_rate_cards_1_7.json")
    for card, variations in rc17.items():
        number = re.search(r"(\d+)$", card).group(1)
        for variation_key, body in variations.items():
            variation = re.search(r"(\d+)$", variation_key).group(1)
            key = f"rate_card_{number}_variasi_{variation}.pdf"
            gt[("RATE_AGREEMENT", key)] = _make_gt("RATE_AGREEMENT", key, body, f"{card}/{variation_key}")
    rc815 = read_json(dataset_root / "rate card" / "ground_truth_rate_card_8_15.json")
    for doc in rc815["documents"]:
        key = pdf_key(doc["file_name"])
        gt[("RATE_AGREEMENT", key)] = _make_gt("RATE_AGREEMENT", key, doc, doc["file_name"])
    return gt


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset_root: Path) -> dict:
    """Manifest of every canonical PDF/PNG: relative path, hash, type, template, variation, split, GT status."""
    gt = load_gt(dataset_root)
    entries = []
    gt_seen: set[tuple[str, str]] = set()
    for folder, doc_type in FOLDERS.items():
        slug = SLUGS[folder]
        for sub, kind in (("pdf", "pdf"), ("image", "image")):
            base = dataset_root / folder / sub
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".pdf", ".png", ".jpg", ".jpeg"):
                    continue
                relative = path.relative_to(dataset_root).as_posix()
                if "data_pod_testing" in relative.lower():
                    continue
                key = pdf_key(path.name)
                in_ori = "/ori/" in relative
                template, variation = (None, None) if (kind == "image" or in_ori) else template_of(doc_type, key)
                pages = len(pymupdf.open(path)) if kind == "pdf" else 1
                has_gt = kind == "pdf" and not in_ori and (doc_type, key) in gt
                if has_gt:
                    gt_seen.add((doc_type, key))
                split = "smoke" if (kind == "image" or in_ori) else (split_of(template) if has_gt else "nogt")
                entries.append(
                    {
                        "id": f"{slug}/{Path(key).stem}" if kind == "pdf" and not in_ori else f"{slug}/{'ori_' if in_ori else 'img_'}{Path(key).stem}",
                        "doc_type": doc_type,
                        "path": relative,
                        "sha256": sha256_of(path),
                        "source_kind": kind,
                        "pages": pages,
                        "template": template,
                        "variation": variation,
                        "split": split,
                        "has_gt": has_gt,
                        "gt_name": gt[(doc_type, key)].gt_name if has_gt else None,
                    }
                )
    orphan_gt = sorted(f"{t}:{k}" for (t, k) in gt if (t, k) not in gt_seen)
    return {
        "version": 1,
        "dataset_root_hint": "../DATASET/Dataset",
        "label": "synthetic ground truth, not human gold",
        "split_rule": f"sha256(template) % 100: <{HOLDOUT_PERCENT} holdout, <{HOLDOUT_PERCENT + VAL_PERCENT} val, else dev",
        "entries": entries,
        "gt_without_pdf": orphan_gt,
        "summary": {
            "entries": len(entries),
            "by_split": dict(Counter(e["split"] for e in entries)),
            "by_type_gt": dict(Counter(e["doc_type"] for e in entries if e["has_gt"])),
        },
    }


def row_content_gold(manifest: dict, gt: dict[tuple[str, str], GtDoc]) -> dict[str, dict]:
    """Per document: GT table rows as lists of scalar strings (content only, no structure)."""
    result = {}
    for entry in manifest["entries"]:
        if not entry["has_gt"]:
            continue
        doc_type = entry["doc_type"]
        key = pdf_key(Path(entry["path"]).name)
        doc = gt[(doc_type, key)]
        result[entry["id"]] = {
            "document_id": entry["id"],
            "doc_type": doc_type,
            "source_label": "synthetic_gt_row_content",
            "tables": [{"path": t["path"], "rows": [[str(v) for _, v in row] for row in t["rows"]]} for t in doc.tables],
            "totals": doc.totals,
        }
    return result


def inventory(gt: dict[tuple[str, str], GtDoc], alias_path: Path) -> dict:
    """GT table paths and key frequencies, plus a mapping of keys to canonical fields from YAML aliases."""
    aliases = yaml.safe_load(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    key_to_canonical: dict[str, list[str]] = defaultdict(list)
    for canonical, keys in (aliases.get("canonical_fields") or {}).items():
        for key in keys:
            key_to_canonical[key].append(canonical)
    paths: dict[str, dict] = {}
    for (doc_type, _), doc in gt.items():
        for table in doc.tables:
            record = paths.setdefault(f"{doc_type}:{table['path']}", {"tables": 0, "rows": 0, "keys": Counter()})
            record["tables"] += 1
            record["rows"] += len(table["rows"])
            for row in table["rows"]:
                record["keys"].update(key for key, _ in row)
    all_keys = Counter()
    for record in paths.values():
        all_keys.update(record["keys"])
    unmapped = sorted(key for key in all_keys if key not in key_to_canonical)
    return {
        "distinct_line_item_keys": len(all_keys),
        "paths": {p: {"tables": r["tables"], "rows": r["rows"], "keys": dict(r["keys"].most_common())} for p, r in sorted(paths.items())},
        "canonical_counts": {c: sum(n for k, n in all_keys.items() if c in key_to_canonical.get(k, [])) for c in (aliases.get("canonical_fields") or {})},
        "unmapped_keys": unmapped,
        "ambiguous_keys": sorted(k for k, v in key_to_canonical.items() if len(v) > 1),
    }
