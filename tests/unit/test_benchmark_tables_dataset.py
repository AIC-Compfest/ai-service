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


def test_discovery_uses_only_immediate_canonical_media_files(tmp_path: Path):
    benchmark = _benchmark_module()
    expected = {
        "invoice": "INVOICE",
        "PoD": "POD",
        "rate card": "RATE_AGREEMENT",
        "surat jalan": "SURAT_JALAN",
    }
    for folder in expected:
        media = tmp_path / folder / "pdf"
        media.mkdir(parents=True)
        (media / "approved.pdf").write_bytes(b"%PDF")
        nested = media / "archive"
        nested.mkdir()
        (nested / "excluded.pdf").write_bytes(b"%PDF")

    # A valid-looking file outside the four canonical roots must not leak in.
    (tmp_path / "Dataset").mkdir()
    (tmp_path / "Dataset" / "excluded.png").write_bytes(b"not-an-image")

    sources = benchmark.discover_dataset_documents(tmp_path)

    assert {(path.parent.parent.name, doc_type) for path, doc_type, _ in sources} == set(expected.items())
    assert all(path.name == "approved.pdf" for path, _, _ in sources)
