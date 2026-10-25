"""Dataset-selection tests for the table benchmark harness (manifest driven, D-027)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tables.py"
    spec = spec_from_file_location("benchmark_tables", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(entry_id: str, path: str, kind: str, split: str, doc_type: str = "INVOICE") -> dict:
    return {"id": entry_id, "path": path, "source_kind": kind, "split": split, "doc_type": doc_type}


def test_discovery_follows_the_manifest_and_never_substitutes(tmp_path: Path, capsys):
    benchmark = _benchmark_module()
    (tmp_path / "invoice" / "pdf").mkdir(parents=True)
    present = tmp_path / "invoice" / "pdf" / "a.pdf"
    present.write_bytes(b"x")
    (tmp_path / "invoice" / "pdf" / "not_in_manifest.pdf").write_bytes(b"x")
    manifest = {
        "entries": [
            _entry("invoice/a", "invoice/pdf/a.pdf", "pdf", "dev"),
            _entry("invoice/missing", "invoice/pdf/missing.pdf", "pdf", "dev"),
            _entry("invoice/h", "invoice/pdf/h.pdf", "pdf", "holdout"),
            _entry("invoice/img", "invoice/image/i.png", "image", "smoke"),
        ]
    }
    sources = benchmark.discover_documents(manifest, tmp_path, splits={"dev", "val"}, kinds={"pdf"})
    assert sources == [(present, "INVOICE", "invoice/a", True, "dev")]
    assert "missing on disk" in capsys.readouterr().out


def test_aggregate_handles_missing_arithmetic_without_inventing_rates():
    benchmark = _benchmark_module()
    page = {"pct_estimated": 0.0, "has_table_region": False, "arith_rows_evaluated": 0, "arith_rows_passed": 0}
    doc = {"avg_confidence": 0.9, "avg_ocr_time_ms_per_page": 5.0, "has_table": True, "has_line_items": True, "pages": [page]}
    aggregate = benchmark.aggregate_metrics([doc])
    assert aggregate["arith_row_pass_rate"] is None
    assert aggregate["documents_with_table"] == 1
