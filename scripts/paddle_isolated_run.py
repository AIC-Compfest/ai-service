"""Run the raster OCR path on every dataset image, one child process per image, with a timeout.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\paddle_isolated_run.py [--timeout 240] [--limit N] [--out out/paddle] [--python PATH]
Writes <out>/<type>__<image>.json (evidence) and <out>/summary.json: exit code, seconds, lines, stderr tail. Nothing is
assumed about Paddle: a non-zero exit, a timeout, or an empty page is recorded as such. --python lets the same run be
repeated with another interpreter (e.g. .venv-exp) to compare environments.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("VERIFLOW_DATASET_ROOT", ROOT.parent / "DATASET" / "Dataset"))
CHILD = ROOT / "scripts" / "experiments" / "ocr_one.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "paddle")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", default="", help="substring of the image path")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in DATASET.glob("*/image/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and args.only in str(p))
    if args.limit:
        images = images[: args.limit]
    rows = []
    for image in images:
        target = args.out / f"{image.parent.parent.name.replace(' ', '_')}__{image.stem}.json"
        started = time.perf_counter()
        try:
            done = subprocess.run([args.python, str(CHILD), str(image), str(target)], capture_output=True, text=True, timeout=args.timeout, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
            code, status, stderr = done.returncode, "ok" if done.returncode == 0 else "crash", done.stderr[-400:]
        except subprocess.TimeoutExpired as error:
            code, status, stderr = None, "timeout", (error.stderr or b"")[-400:].decode("utf-8", "replace") if isinstance(error.stderr, bytes) else str(error.stderr or "")[-400:]
        row = {"image": f"{image.parent.parent.name}/{image.name}", "status": status, "exit_code": code, "wall_seconds": round(time.perf_counter() - started, 1)}
        if status == "ok" and target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
            row.update({k: result[k] for k in ("seconds", "lines", "tokens", "mean_confidence", "tables")})
            if result["lines"] == 0:
                row["status"] = "empty"
        else:
            row["stderr_tail"] = stderr
        rows.append(row)
        print(row["image"], row["status"], row.get("lines"), row["wall_seconds"], flush=True)
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    (args.out / "summary.json").write_text(json.dumps({"python": args.python, "timeout": args.timeout, "counts": counts, "images": rows}, indent=2), encoding="utf-8")
    print("SUMMARY", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
