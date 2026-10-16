"""Compare two folders written by paddle_isolated_run.py: speed and text agreement per image.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\compare_ocr_runs.py <reference_dir> <candidate_dir>
Agreement = share of reference lines whose text (casefolded, spaces collapsed) also appears in the candidate, and the
same for the numbers (digit runs) found in the page. Neither is accuracy: both runs can be wrong in the same way.
"""
import json
import re
import sys
from pathlib import Path


def texts(path: Path) -> list[str]:
    page = json.loads(path.read_text(encoding="utf-8"))["page"]
    return [re.sub(r"\s+", " ", line["text"]).strip().casefold() for line in page["lines"] if line.get("text")]


def numbers(lines: list[str]) -> set[str]:
    return {m for line in lines for m in re.findall(r"\d[\d.,]*\d|\d", line)}


reference, candidate = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for ref in sorted(reference.glob("*__*.json")):
    cand = candidate / ref.name
    if not cand.exists():
        continue
    a, b = texts(ref), texts(cand)
    na, nb = numbers(a), numbers(b)
    ra, rb = json.loads(ref.read_text(encoding="utf-8")), json.loads(cand.read_text(encoding="utf-8"))
    rows.append((ref.stem, ra["seconds"], rb["seconds"], len(a), len(b), sum(t in set(b) for t in a) / max(len(a), 1), len(na & nb) / max(len(na), 1)))
print("image | ref s | cand s | ref lines | cand lines | line agreement | number agreement")
for r in rows:
    print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.2f} | {r[6]:.2f}")
if rows:
    print("MEAN | %.1f | %.1f | | | %.2f | %.2f" % tuple(sum(r[i] for r in rows) / len(rows) for i in (1, 2, 5, 6)))
