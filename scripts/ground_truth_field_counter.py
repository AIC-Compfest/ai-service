"""
Count how many times each field name is mentioned in the "documents" of a
ground truth JSON file.

- Plain fields (strings, numbers, null, arrays of plain values) are reported
  as a single integer: the number of times the field appears.
- Fields holding objects, or arrays of objects, are reported as:
      {"count": <times the field appears>,
       "fields": [{<sub-field>: <counts, same format recursively>}]}
  Sub-field counts include every object inside an array.

Usage:
    python ground_truth_field_counter.py path/to/ground_truth.json
    python ground_truth_field_counter.py path/to/ground_truth.json -o counts.json
"""

import argparse
import json
import sys
from pathlib import Path


def gather_children(objects) -> dict:
    """Collect, for every key found in the given objects, the list of its values."""
    children: dict = {}
    for obj in objects:
        if isinstance(obj, dict):
            for key, value in obj.items():
                children.setdefault(key, []).append(value)
    return children


def summarize_fields(children: dict) -> dict:
    """Turn {field: [values...]} into {field: count-or-structure}, most frequent first."""
    ordered = sorted(children.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return {key: summarize_key(values) for key, values in ordered}


def summarize_key(values: list):
    """
    values holds one entry per time the field appears.
    Returns an int for plain fields, or {"count", "fields"} for object/array-of-object fields.
    """
    count = len(values)

    # Flatten arrays so that objects inside them are counted too.
    items = []
    for value in values:
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)

    if not any(isinstance(item, dict) for item in items):
        return count

    return {"count": count, "fields": [summarize_fields(gather_children(items))]}


def build_report(data: dict, source_name: str) -> dict:
    documents = data.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError('"documents" must be a list in the ground truth file')

    return {
        "file": source_name,
        "total_documents": len(documents),
        "field_counts": summarize_fields(gather_children(documents)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count how many times each field is mentioned in the documents of a ground truth JSON file."
    )
    parser.add_argument("file", type=Path, help="Path to the ground truth JSON file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Optional path to write the JSON output to (default: print to stdout)",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    try:
        with args.file.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {args.file}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: top-level JSON value must be an object", file=sys.stderr)
        return 1

    try:
        report = build_report(data, args.file.name)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote results to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())