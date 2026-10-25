#!/usr/bin/env python
"""Bulk-train one LightGBM binary model per document type and field.

Point it at the folder that holds the candidate CSVs and it trains every field it finds,
writing one bundle per field to --out (``<out>/<DOC>.<field>.binary.v1/``), plus a
preflight table, a run log and a summary (JSON and CSV).

Example (PowerShell, from the repo root):

    conda run -n aic_compfest python -m app.tools.train_all `
        --candidate-dir data\\candidates_v3 --out models\\v3

Common options:
    --dry-run                    print the preflight table and stop
    --doc-types INVOICE POD     train only these document types
    --fields INVOICE.grand_total pihak_pertama_nama
    --set lgbm_defaults.n_estimators=200 --set t_unknown_floor=0.5
    --skip-existing              keep bundles already present in --out
    --workers 2                  train fields in parallel processes
    --fail-fast                  stop after the first failed field
    --no-progress                no progress bar (log and summary are still written)

Exit code: 0 when no field failed (skipped fields are normal), 1 when any field failed.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.services.candidates.registry import FieldRegistry
from app.services.training.run import train_binary

SCHEMA_FILE = "training_schema.json"
MANIFEST_FILE = "feature_manifest.json"
DEFAULT_REGISTRY = Path("config/candidates/field_registry.yaml")
DEFAULT_CONFIG = Path("config/training/training.yaml")

PREFLIGHT_COLUMNS = ["doc_type", "field", "candidate_file", "rows", "trainable", "positives", "negatives", "label_export"]
TRAINABLE_REVIEW = {"auto_accepted", "human_confirmed", "human_corrected"}  # same states the trainer accepts
SUMMARY_COLUMNS = [
    "doc_type", "field", "status", "model_id", "n_rows", "n_features", "positives", "negatives",
    "label_export", "oof_accepted", "oof_precision", "oof_coverage", "t_auto", "t_unknown",
    "seconds", "reason", "bundle",
]


def model_id(doc_type: str, field: str) -> str:
    return f"{doc_type}.{field}.binary.v1"


# ----------------------------------------------------------------------------- field discovery
def discover_fields(candidate_dir: Path, registry_path: Path, source: str) -> tuple[list[tuple[str, str]], str]:
    schema_path = candidate_dir / SCHEMA_FILE
    use_schema = source == "schema" or (source == "auto" and schema_path.exists())
    if use_schema:
        if not schema_path.exists():
            raise SystemExit(f"--field-source schema, but {schema_path} does not exist")
        keys = json.loads(schema_path.read_text(encoding="utf-8"))
        pairs = [tuple(key.split(".", 1)) for key in keys if "." in key]
        origin = str(schema_path)
    else:
        if not registry_path.exists():
            raise SystemExit(f"field registry not found: {registry_path}")
        registry = FieldRegistry.load(registry_path)
        doc_types = sorted({field.doc_type for field in registry.fields})
        pairs = [(doc, field.field) for doc in doc_types for field in registry.for_doc(doc)]
        origin = str(registry_path)
    return sorted(set(pairs)), origin


def filter_fields(pairs: list[tuple[str, str]], doc_types: list[str] | None, fields: list[str] | None) -> list[tuple[str, str]]:
    if doc_types:
        known = sorted({doc for doc, _ in pairs})
        unknown = sorted(set(doc_types) - set(known))
        if unknown:
            raise SystemExit(f"unknown --doc-types {unknown}; available: {known}")
        pairs = [(doc, field) for doc, field in pairs if doc in doc_types]
    if fields:
        matched = [(doc, field) for doc, field in pairs if f"{doc}.{field}" in fields or field in fields]
        names = {f"{doc}.{field}" for doc, field in matched} | {field for _, field in matched}
        missing = sorted(item for item in fields if item not in names)
        if missing:
            raise SystemExit(f"--fields not found: {missing}")
        pairs = matched
    return pairs


# ----------------------------------------------------------------------------- preflight
@functools.lru_cache(maxsize=None)
def read_csv(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(reader)
        header = tuple(reader.fieldnames or ())
    return header, rows


def inspect_field(candidate_dir: Path, doc_type: str, field: str) -> dict[str, Any]:
    """Report what the trainer will actually see for this field.

    If an export exists, counts come from it (only rows whose review state is trainable).
    Otherwise counts come from the raw candidate label column, which is informational only.
    """
    label = f"label__{field}"
    review = f"label_review__{field}"
    info: dict[str, Any] = {"doc_type": doc_type, "field": field, "candidate_file": "", "rows": 0, "trainable": 0,
                            "positives": 0, "negatives": 0, "label_export": "missing"}
    for path in sorted(candidate_dir.glob(f"candidates_{doc_type}_*.csv")):
        header, rows = read_csv(path)
        if label in header:
            info.update(candidate_file=path.name, rows=len(rows),
                        positives=sum(row[label] == "1" for row in rows),
                        negatives=sum(row[label] == "0" for row in rows))
            break
    for path in sorted(candidate_dir.glob(f"candidate_labels_{doc_type}_*.csv")):
        header, rows = read_csv(path)
        if label not in header:
            continue
        if review not in header:
            info["label_export"] = "no review column"
            break
        trainable = [row for row in rows if row[review] in TRAINABLE_REVIEW and row[label] in ("0", "1")]
        info.update(label_export="ok" if trainable else "no trainable labels", trainable=len(trainable),
                    positives=sum(row[label] == "1" for row in trainable),
                    negatives=sum(row[label] == "0" for row in trainable))
        break
    return info


def format_table(headers: list[str], rows: list[list[Any]], cap: int = 60) -> str:
    cells = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [min(cap, max([len(header)] + [len(row[i]) for row in cells])) for i, header in enumerate(headers)]

    def fmt(values: list[str]) -> str:
        return "  ".join(value[:width].ljust(width) for value, width in zip(values, widths))

    lines = [fmt(headers), "  ".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in cells)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- config overrides
def write_effective_config(base: Path, overrides: list[str], out: Path) -> Path:
    if not base.exists():
        raise SystemExit(f"config not found: {base}")
    config = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        node = config
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--set {key}: '{part}' is not a mapping in {base}")
        node[leaf] = yaml.safe_load(raw)
    path = out / "training_config_used.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- worker
def bundle_summary(bundle: Path) -> dict[str, Any]:
    meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    metrics = meta.get("metrics", {})
    thresholds = meta.get("thresholds", {})
    return {
        "bundle": str(bundle),
        "n_rows": meta.get("n_rows"),
        "n_features": len(meta.get("feature_columns", [])),
        "oof_accepted": metrics.get("accepted"),
        "oof_precision": metrics.get("precision"),
        "oof_coverage": metrics.get("coverage"),
        "t_auto": thresholds.get("t_auto"),
        "t_unknown": thresholds.get("t_unknown"),
    }


def train_job(job: dict[str, Any]) -> dict[str, Any]:
    """Runs in a worker process. Never raises: every outcome becomes a record."""
    doc_type, field = job["doc_type"], job["field"]
    started = time.time()
    record: dict[str, Any] = {"doc_type": doc_type, "field": field, "model_id": model_id(doc_type, field)}
    try:
        result = train_binary(job["candidate_dir"], job["out"], doc_type, field, config_path=job["config"])
        record.update(status="trained", reason="", **bundle_summary(Path(result["bundle"])))
    except (FileNotFoundError, ValueError) as error:  # missing labels, one class, no features
        record.update(status="skipped", reason=str(error))
    except Exception as error:  # one bad field must not stop the whole run
        record.update(status="failed", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    record["seconds"] = round(time.time() - started, 2)
    return record


def failed_record(job: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {"doc_type": job["doc_type"], "field": job["field"], "model_id": model_id(job["doc_type"], job["field"]),
            "status": "failed", "reason": f"worker process error: {error}", "seconds": 0}


# ----------------------------------------------------------------------------- progress and log
def hms(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))


def short(text: str, limit: int = 110) -> str:
    text = (text or "").splitlines()[0] if text else ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Progress:
    """tqdm bar when tqdm is installed, otherwise a dependency-free bar. Plain lines when not a terminal."""

    WIDTH = 30

    def __init__(self, total: int, enabled: bool):
        self.total = total
        self.enabled = enabled
        self.done = 0
        self.counts: Counter[str] = Counter()
        self.started = time.time()
        self.tty = sys.stderr.isatty()
        self.last = ""
        self.bar = None
        if enabled and self.tty:
            try:
                from tqdm import tqdm

                self.bar = tqdm(total=total, unit="model", file=sys.stderr, dynamic_ncols=True)
            except ImportError:
                self.bar = None

    def advance(self, record: dict[str, Any]) -> None:
        self.done += 1
        self.counts[record["status"]] += 1
        self.last = record["model_id"]
        if record["status"] == "failed":
            self.say(f"  FAILED {record['model_id']}: {short(record.get('reason', ''))}")
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix_str(f"trained={self.counts['trained']} skipped={self.counts['skipped']} "
                                     f"failed={self.counts['failed']}", refresh=True)
        elif self.tty:
            self._draw()
        else:
            print(f"[{self.done}/{self.total}] {record['status']:8} {record['model_id']}", file=sys.stderr, flush=True)

    def say(self, message: str) -> None:
        # Always visible (failures must never be silent), even with --no-progress.
        if self.bar is not None:
            self.bar.write(message)
        elif self.tty:
            sys.stderr.write("\r" + " " * 160 + "\r" + message + "\n")
            self._draw()
        else:
            print(message, file=sys.stderr, flush=True)

    def _draw(self) -> None:
        fraction = self.done / self.total if self.total else 1.0
        filled = int(self.WIDTH * fraction)
        elapsed = time.time() - self.started
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0.0
        line = (f"[{'#' * filled}{'.' * (self.WIDTH - filled)}] {self.done}/{self.total} {fraction:6.1%} "
                f"| trained {self.counts['trained']} skipped {self.counts['skipped']} failed {self.counts['failed']} "
                f"| elapsed {hms(elapsed)} eta {hms(eta)} | {self.last}")
        sys.stderr.write("\r" + line[:158].ljust(158))
        sys.stderr.flush()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        elif self.enabled and self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


class RunLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")


def write_csv(path: Path, columns: list[str], rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk-train LightGBM binary models for every field of every document type.")
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="folder with candidates_*.csv, feature_manifest.json, training_policy.yaml and training_schema.json")
    parser.add_argument("--out", type=Path, default=Path("models"), help="where bundles and reports are written (default: models)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"training config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="field registry, used when the field list does not come from training_schema.json")
    parser.add_argument("--field-source", choices=["auto", "schema", "registry"], default="auto",
                        help="auto = training_schema.json if it is in --candidate-dir, otherwise the registry")
    parser.add_argument("--doc-types", nargs="+", metavar="DOC", help="train only these document types, e.g. INVOICE POD")
    parser.add_argument("--fields", nargs="+", metavar="FIELD", help="train only these fields: DOC.field or a bare field name")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a training.yaml value for this run; dotted keys allowed, e.g. "
                             "--set lgbm_defaults.n_estimators=200 (repeatable)")
    parser.add_argument("--workers", type=int, default=1,
                        help="fields trained in parallel processes (default 1; each model also uses lgbm num_threads)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="do not retrain fields whose bundle already exists in --out (it is not checked against new candidates)")
    parser.add_argument("--fail-fast", action="store_true", help="stop scheduling new fields after the first failure")
    parser.add_argument("--dry-run", action="store_true", help="print the preflight table and exit without training")
    parser.add_argument("--no-progress", action="store_true", help="no progress bar; the log and summary are still written")
    parser.add_argument("--log", type=Path, help="log file (default: <out>/train_all.log)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not (args.candidate_dir / MANIFEST_FILE).exists():
        parser.error(f"{args.candidate_dir} has no {MANIFEST_FILE}; is this the candidate folder?")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = datetime.now()
    args.out.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.log or args.out / "train_all.log")

    config_path = write_effective_config(args.config, args.overrides, args.out)
    pairs, origin = discover_fields(args.candidate_dir, args.registry, args.field_source)
    pairs = filter_fields(pairs, args.doc_types, args.fields)
    if not pairs:
        raise SystemExit("no fields to train after filtering")

    preflight = {pair: inspect_field(args.candidate_dir, *pair) for pair in pairs}
    write_csv(args.out / "preflight.csv", PREFLIGHT_COLUMNS, preflight.values())
    doc_count = len({doc for doc, _ in pairs})
    print(f"Fields from {origin}: {len(pairs)} fields across {doc_count} document types")
    print(format_table(PREFLIGHT_COLUMNS, [[info[c] for c in PREFLIGHT_COLUMNS] for info in preflight.values()]))
    not_ready = [i for i in preflight.values() if i["candidate_file"] and i["label_export"] != "ok"]
    if not_ready:
        print(f"\nWARNING: {len(not_ready)} fields have no trainable labels (export missing, no review column, or all rows blank). "
              "They will be skipped. Finish grounding/review, then run `labeling export` and check this table again.")
    if args.dry_run:
        print("\nDry run: nothing trained.")
        return 0

    log.write(f"start candidate_dir={args.candidate_dir} out={args.out} config={config_path} "
              f"overrides={args.overrides} fields={len(pairs)} workers={args.workers}")
    progress = Progress(len(pairs), enabled=not args.no_progress)
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def record(rec: dict[str, Any]) -> None:
        key = (rec["doc_type"], rec["field"])
        info = preflight[key]
        rec.update(positives=info["positives"], negatives=info["negatives"], label_export=info["label_export"])
        records[key] = rec
        progress.advance(rec)
        log.write(f"{rec['status']:8} {rec['model_id']} {rec.get('reason', '')}".rstrip())
        if rec.get("traceback"):
            log.write(rec["traceback"].rstrip())

    jobs: list[dict[str, Any]] = []
    for pair in pairs:
        doc, field = pair
        info = preflight[pair]
        bundle_dir = args.out / model_id(doc, field)
        if not info["candidate_file"]:
            record({"doc_type": doc, "field": field, "model_id": model_id(doc, field), "status": "skipped",
                    "reason": f"no label__{field} column in candidates_{doc}_*.csv", "seconds": 0})
        elif args.skip_existing and (bundle_dir / "bundle.json").exists():
            record({"doc_type": doc, "field": field, "model_id": model_id(doc, field), "status": "existing",
                    "reason": "bundle already exists", "seconds": 0, **bundle_summary(bundle_dir)})
        else:
            jobs.append({"doc_type": doc, "field": field, "candidate_dir": str(args.candidate_dir),
                         "out": str(args.out), "config": str(config_path)})

    if args.workers == 1 or len(jobs) <= 1:
        for job in jobs:
            rec = train_job(job)
            record(rec)
            if rec["status"] == "failed" and args.fail_fast:
                break
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(train_job, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    rec = future.result()
                except Exception as error:  # worker process crashed
                    rec = failed_record(futures[future], error)
                record(rec)
                if rec["status"] == "failed" and args.fail_fast:
                    pool.shutdown(cancel_futures=True)
                    break

    for pair in pairs:
        if pair not in records:
            record({"doc_type": pair[0], "field": pair[1], "model_id": model_id(*pair), "status": "not_run",
                    "reason": "stopped by --fail-fast", "seconds": 0})
    progress.close()

    ordered = [records[pair] for pair in pairs]
    counts = Counter(rec["status"] for rec in ordered)
    finished_at = datetime.now()
    meta = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "candidate_dir": str(args.candidate_dir),
        "field_source": origin,
        "config_used": str(config_path),
        "overrides": args.overrides,
        "workers": args.workers,
        "skip_existing": args.skip_existing,
        "counts": dict(counts),
        "python": sys.version.split()[0],
    }
    (args.out / "training_summary.json").write_text(
        json.dumps({"meta": meta, "results": ordered}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_csv(args.out / "training_summary.csv", SUMMARY_COLUMNS, ordered)

    def show(value: Any, digits: int = 3) -> Any:
        return round(value, digits) if isinstance(value, float) else value

    print()
    print(format_table(
        ["doc_type", "field", "status", "n_rows", "oof_precision", "oof_coverage", "seconds", "reason"],
        [[rec["doc_type"], rec["field"], rec["status"], rec.get("n_rows", ""), show(rec.get("oof_precision")),
          show(rec.get("oof_coverage")), rec.get("seconds", ""), short(rec.get("reason", ""), 90)] for rec in ordered],
        cap=90))
    print(f"\nDone in {hms((finished_at - started_at).total_seconds())}: "
          + ", ".join(f"{status}={counts[status]}" for status in ("trained", "existing", "skipped", "failed", "not_run") if counts[status]))
    print(f"Bundles: {args.out}\nSummary: {args.out / 'training_summary.csv'} (details in training_summary.json)\nLog: {log.path}")
    log.write(f"finished counts={dict(counts)}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())