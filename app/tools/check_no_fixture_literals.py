"""Reject reference-document literals from production source trees."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ROOT / "tests" / "fixtures" / "surat_jalan" / "forbidden_literals.txt"
SOURCE_SUFFIXES = {".py", ".go", ".ts", ".tsx", ".js", ".jsx"}
EXCLUDED_PARTS = {"tests", "test", "__fixtures__", "fixtures", "node_modules", ".git", ".next"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    literals = [line.strip() for line in FORBIDDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = []
    for root in args.repos:
        if not root.exists():
            raise SystemExit(f"Repository path does not exist: {root}")
        source_dirs = [root / name for name in ("app", "internal", "src") if (root / name).is_dir()]
        for source_dir in source_dirs:
            for path in source_dir.rglob("*"):
                if (
                    not path.is_file()
                    or path.suffix.lower() not in SOURCE_SUFFIXES
                    or path.name.endswith(("_test.go", ".test.ts", ".test.tsx"))
                    or EXCLUDED_PARTS.intersection(path.parts)
                ):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for literal in literals:
                    if literal in text:
                        failures.append(f"{path}: {literal}")
    if failures:
        print("Forbidden reference literals found:\n" + "\n".join(failures))
        return 1
    print("No reference-document literals found in production source trees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
