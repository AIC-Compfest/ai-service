import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ocr.engine import OCREngine, to_evidence_json
from app.services.candidates.pipeline import CandidatePipeline

# Test files for comprehensive validation across all 4 categories in DATASET
DEFAULT_DATASET_DIR = Path(os.environ.get("VERIFLOW_DATASET_ROOT", Path(__file__).resolve().parents[2] / "DATASET" / "Dataset"))

TARGET_TESTS = [
    {
        "category": "INVOICE",
        "doc_type": "INVOICE",
        "rel_path": "invoice/image/Contoh-Template-Invoice-2.jpg"
    },
    {
        "category": "INVOICE",
        "doc_type": "INVOICE",
        "rel_path": "invoice/image/Contoh-Template-Invoice-1.jpg"
    },
    {
        "category": "POD",
        "doc_type": "POD",
        "rel_path": "PoD/image/contoh-surat-bukti-terima-1.png"
    },
    {
        "category": "RATE_AGREEMENT",
        "doc_type": "RATE_AGREEMENT",
        "rel_path": "rate card/image/contoh-rate-card-1.png"
    },
    {
        "category": "SURAT_JALAN",
        "doc_type": "SURAT_JALAN",
        "rel_path": "surat jalan/image/contoh-surat-jalan-10.png"
    }
]

def _benchmark_schema(benchmark_dir: Path, candidates: list[dict], doc_type: str) -> dict:
    """Compare generated row columns with the checked-in candidate benchmark schema."""
    results = {}
    for benchmark in benchmark_dir.glob(f"candidates_{doc_type}_*.csv"):
        with benchmark.open(encoding="utf-8", newline="") as handle:
            expected_columns = set((csv.DictReader(handle).fieldnames or []))
        candidate_type = benchmark.stem.rsplit("_", 1)[-1]
        type_rows = [row for row in candidates if row.get("m_cand_type") == candidate_type]
        if not type_rows:
            results[benchmark.name] = {
                "schema_parity": None,
                "status": "not_emitted",
                "missing_columns": [],
                "generated_rows_for_type": 0,
            }
            continue
        generated_type_columns = set().union(*(row.keys() for row in type_rows))
        missing = sorted(expected_columns - generated_type_columns)
        results[benchmark.name] = {
            "schema_parity": not missing,
            "status": "validated",
            "missing_columns": missing,
            "generated_rows_for_type": len(type_rows),
        }
    return results


def run_comprehensive_test(dataset_dir: Path, category: str | None, save_report: Path | None, benchmark_dir: Path | None):
    print("=" * 70)
    print("COMPREHENSIVE OCR & CANDIDATE PIPELINE INTEGRATION TEST")
    print("Target Dataset Root:", dataset_dir)
    print("=" * 70)

    engine = OCREngine()
    candidate_pipeline = CandidatePipeline(config_dir=Path("config/candidates"))

    summary_results = []

    for item in TARGET_TESTS:
        if category and item["category"].casefold() != category.casefold():
            continue
        file_path = dataset_dir / item["rel_path"]
        doc_type = item["doc_type"]
        doc_id = Path(item["rel_path"]).stem

        print(f"\n[TESTING] Category: {item['category']} | File: {item['rel_path']}")
        if not file_path.exists():
            print(f"  [ERROR] File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        # 1. OCR Pure Isolation
        page_evidence = engine.process_image(img_bytes, page_number=1, quality={})
        ocr_time = time.time() - t0

        evidence_payload = to_evidence_json([page_evidence])
        page_dict = evidence_payload["pages"][0]

        total_lines = len(page_dict.get("lines", []))
        total_tokens = sum(len(line.get("tokens", [])) for line in page_dict.get("lines", []))
        avg_conf = page_dict.get("average_confidence", 0.0)

        # 2. Check token contract conformance
        sample_token = None
        has_estimated_flag = True
        has_pixel_leak = False

        for line in page_dict.get("lines", []):
            for tok in line.get("tokens", []):
                if sample_token is None:
                    sample_token = tok
                if "_estimated" not in tok:
                    has_estimated_flag = False
                if "bbox_pixel" in tok:
                    has_pixel_leak = True

        # 3. Direct feeding into Candidate Generation Pipeline (Contract validation)
        t_cand_0 = time.time()
        try:
            candidates = candidate_pipeline.generate(
                document_id=doc_id,
                doc_type=doc_type,
                ocr_result=evidence_payload,
                run_id="test_comprehensive"
            )
            cand_time = time.time() - t_cand_0
            cand_types_count = {}
            for c in candidates:
                ct = c["m_cand_type"]
                cand_types_count[ct] = cand_types_count.get(ct, 0) + 1
            pipeline_status = "SUCCESS"
        except Exception as e:
            pipeline_status = f"FAILED: {e}"
            candidates = []
            cand_time = 0.0
            cand_types_count = {}

        result = {
            "category": item["category"],
            "file": file_path.name,
            "dimensions": f"{page_dict.get('width')}x{page_dict.get('height')}",
            "lines": total_lines,
            "tokens": total_tokens,
            "avg_conf": avg_conf,
            "contract_valid": has_estimated_flag and not has_pixel_leak,
            "pipeline_status": pipeline_status,
            "total_candidates": len(candidates),
            "candidates_breakdown": cand_types_count,
            "ocr_time_sec": round(ocr_time, 2),
            "sample_snippet": page_evidence.full_text()[:120].replace("\n", " ")
        }
        if benchmark_dir:
            result["benchmark"] = _benchmark_schema(benchmark_dir, candidates, doc_type)
        summary_results.append(result)

        print(f"  -> OCR Completed in {ocr_time:.2f}s | Lines: {total_lines} | Tokens: {total_tokens} | Conf: {avg_conf:.3f}")
        print(f"  -> Contract Valid: {'YES' if result['contract_valid'] else 'NO (Schema Violation)'}")
        print(f"  -> Candidate Pipeline: {pipeline_status} (Generated {len(candidates)} candidates in {cand_time:.2f}s)")
        print(f"  -> Candidates Breakdown: {cand_types_count}")

    print("\n" + "=" * 70)
    print("TEST SUMMARY REPORT")
    print("=" * 70)
    for r in summary_results:
        print(f"- [{r['category']}] {r['file']}: Lines={r['lines']}, Tokens={r['tokens']}, Candidates={r['total_candidates']}, Status={r['pipeline_status']}")

    if save_report:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(summary_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nReport written to: {save_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR-to-candidate validation against representative documents.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--category", help="Run only one category, e.g. INVOICE or POD.")
    parser.add_argument("--save-report", type=Path, help="Optional JSON path; no dataset files are overwritten by default.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_DATASET_DIR / "bundle_4docs")
    args = parser.parse_args()
    run_comprehensive_test(args.dataset_dir, args.category, args.save_report, args.benchmark_dir)

if __name__ == "__main__":
    main()
