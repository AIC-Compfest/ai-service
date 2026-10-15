"""Dataset-boundary tests for the table benchmark harness."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tables.py"
    spec = spec_from_file_location("benchmark_tables", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_uses_only_the_four_allow_listed_documents(tmp_path: Path):
    benchmark = _benchmark_module()
    expected = []
    for relative, doc_type in benchmark.CANONICAL_DATASET_DOCUMENTS:
        approved = tmp_path / relative
        approved.parent.mkdir(parents=True, exist_ok=True)
        approved.write_bytes(b"fixture")
        expected.append((approved, doc_type, approved.suffix.lower() == ".pdf"))

    # Valid-looking extras in a permitted directory and unknown paths cannot
    # leak into a run.
    extra = tmp_path / "invoice" / "image" / "unreviewed.jpg"
    extra.write_bytes(b"fixture")
    (tmp_path / "Dataset").mkdir()
    (tmp_path / "Dataset" / "excluded.png").write_bytes(b"fixture")

    sources = benchmark.discover_dataset_documents(tmp_path)

    assert sources == expected
