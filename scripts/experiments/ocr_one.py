"""Child process: OCR one image with the production engine and write a small JSON result.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\experiments\\ocr_one.py <image> <out.json> [--no-tables]
Run through scripts/paddle_isolated_run.py so a crash or hang of the native Paddle runtime cannot take the caller down.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.ocr.engine import OCREngine  # noqa: E402

image, out = Path(sys.argv[1]), Path(sys.argv[2])
started = time.perf_counter()
page = OCREngine().process_image(image.read_bytes(), 1, {}, discover_tables=("--no-tables" not in sys.argv))
tables = [t.to_dict() for t in page.tables]
out.write_text(json.dumps({
    "image": image.name, "seconds": round(time.perf_counter() - started, 2), "engine": page.engine_used, "width": page.width, "height": page.height,
    "lines": len(page.lines), "tokens": sum(len(line.tokens or []) for line in page.lines), "mean_confidence": round(page.average_confidence(), 4),
    "tables": len(tables), "page": page.to_dict(),
}), encoding="utf-8")
print("OK", len(page.lines), "lines")
