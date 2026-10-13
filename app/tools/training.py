from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.training.run import train_binary
from app.services.candidates.registry import FieldRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 training commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--candidate-dir", type=Path, required=True)
    run.add_argument("--out", type=Path, default=Path("models"))
    run.add_argument("--doc-type", default="POD")
    run.add_argument("--field", default="receiver_name")
    run.add_argument("--all-fields", action="store_true")
    run.add_argument("--config", type=Path, default=Path("config/training/training.yaml"))
    args = parser.parse_args()
    if args.command == "run":
        if args.all_fields:
            registry = FieldRegistry.load(Path("config/candidates/field_registry.yaml"))
            results = []
            for field in registry.for_doc(args.doc_type):
                try:
                    results.append(train_binary(args.candidate_dir, args.out, args.doc_type, field.field, config_path=args.config))
                except (FileNotFoundError, ValueError) as error:
                    results.append({"field": field.field, "skipped": str(error)})
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            print(json.dumps(train_binary(args.candidate_dir, args.out, args.doc_type, args.field, config_path=args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()