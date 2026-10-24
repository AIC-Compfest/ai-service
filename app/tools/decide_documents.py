#!/usr/bin/env python
"""Per-document decisions from candidate scores.

Two stages:

  score    candidate CSVs + trained bundles -> scored.csv (one row per candidate per field)
           --source oof         out-of-fold scores written at training time. Use this to evaluate
                                decisions, because each score came from a model that never saw
                                that template.
           --source in_sample   scores from the final model. Use this for new documents, where no
                                OOF scores exist. On training data these are optimistic.

  decide   scored.csv + decision policy -> decisions.csv and summary.csv.
           When labels are present (label__ in the scored rows), it also reports document-level
           accuracy and compares against plain top-1.

Policy (config/extraction/decision_policy.yaml), keyed by DOC.field, with an optional _default:
  mode: top1          accept the highest-scored candidate
  floor: 0.3          accept only if the top score is at least this, otherwise flag for review
  margin: 0.1         accept only if top score minus runner-up score is at least this, otherwise flag
  mode: filter_top1   keep candidates whose parsed value is in `allowed`, then apply floor/margin
  mode: amount_sum    a + b must equal c within `tolerance` (equation [a, b, c]).
                      strategy: confirm (default) accepts the top-1 candidates when they add up and
                      flags them for review when they do not.
                      strategy: search tries the top `--k` candidates of each field and picks the
                      best matching combination. Tested worse: coincidental sums add false matches.
  mode: skip          no automatic decision (rule or manual review)

Statuses: accepted (answer given), flagged (candidate exists but fails floor/margin, so review),
abstain (no candidate, or none allowed), skipped (policy says skip).

Examples (PowerShell, from the repo root):
    conda run -n aic_compfest python -m app.tools.decide_documents score `
        --candidate-dir data\\candidates_v3 --models-dir models\\all_v3 --source oof --out out\\decisions\\scored.csv
    conda run -n aic_compfest python -m app.tools.decide_documents decide `
        --scored out\\decisions\\scored.csv --policy config\\extraction\\decision_policy.yaml --out out\\decisions
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.services.training.bundle import load_bundle

TRAINABLE_REVIEW = {"auto_accepted", "human_confirmed", "human_corrected"}
SCORED_COLUMNS = ["doc_type", "field", "document_id", "candidate_id", "cand_type", "page", "value", "score", "label", "source"]
DECISION_COLUMNS = ["doc_type", "field", "document_id", "status", "candidate_id", "value", "score", "margin", "reason", "label", "correct"]
SUMMARY_COLUMNS = ["doc_type", "field", "mode", "documents", "answerable_docs", "accepted", "flagged", "abstain", "skipped",
                   "coverage", "accepted_precision", "doc_accuracy", "top1_accuracy", "top1_hit_answerable"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def encode(value: str, vocab: list[str]) -> float:
    """Same encoding as training (assemble.py) and inference (predictor._predict_value)."""
    if value in {None, ""}:
        return float("nan")
    if vocab:
        return float(vocab.index(value) if value in vocab else vocab.index("other") if "other" in vocab else -1)
    return float(value)


def make_record(doc: str, field: str, row: dict[str, str], score: float, label: str, source: str) -> dict[str, Any]:
    return {
        "doc_type": doc, "field": field, "document_id": row["m_document_id"], "candidate_id": row.get("m_candidate_id", ""),
        "cand_type": row.get("m_cand_type", ""), "page": row.get("m_page", ""),
        "value": row.get("m_parsed_value") or row.get("m_raw_text", ""), "score": f"{score:.6f}",
        "label": label or "", "source": source,
    }


def bundle_dirs(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.glob("*.binary.v1") if (path / "bundle.json").exists())


def score_in_sample(candidate_dir: Path, path: Path) -> list[dict[str, Any]]:
    model, meta = load_bundle(path)
    doc, field, cand_type = meta["doc_type"], meta["field"], meta["cand_type"]
    vocabs = meta.get("feature_vocabs", {})
    rows = [row for source in sorted(candidate_dir.glob(f"candidates_{doc}_*.csv"))
            for row in read_rows(source) if row.get("m_cand_type") == cand_type]
    if not rows:
        return []
    matrix = np.asarray([[encode(row.get(column, ""), vocabs.get(column, [])) for column in meta["feature_columns"]] for row in rows], dtype=float)
    probabilities = model.predict_proba(matrix)[:, 1]
    label = f"label__{field}"
    return [make_record(doc, field, row, float(p), row.get(label, ""), "in_sample") for row, p in zip(rows, probabilities)]


def score_oof(candidate_dir: Path, path: Path) -> list[dict[str, Any]]:
    """Attach OOF scores to the trainable rows. Row order and labels must match the training data exactly."""
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    doc, field = meta["doc_type"], meta["field"]
    label = f"label__{field}"
    oof_path = path / "oof_predictions.json"
    if not oof_path.exists():
        print(f"  skip {meta['model_id']}: no oof_predictions.json", file=sys.stderr)
        return []
    source = next((p for p in sorted(candidate_dir.glob(f"candidate_labels_{doc}_*.csv")) if label in read_header(p)), None)
    if source is None:
        print(f"  skip {meta['model_id']}: no candidate_labels export containing {label}", file=sys.stderr)
        return []
    rows = [row for row in read_rows(source) if row.get(f"label_review__{field}") in TRAINABLE_REVIEW]
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    if len(rows) != len(oof):
        raise SystemExit(f"{meta['model_id']}: {len(rows)} trainable rows but {len(oof)} OOF predictions. "
                         "The labels or candidates changed after training; retrain or re-export first.")
    records = []
    for row, prediction in zip(rows, oof):
        if row["m_document_id"] != prediction["document_id"] or int(row[label] or 0) != int(prediction["label"]):
            raise SystemExit(f"{meta['model_id']}: row/prediction mismatch at {row['m_document_id']}. Regenerate the OOF file.")
        records.append(make_record(doc, field, row, float(prediction["score"]), row[label], "oof"))
    return records


def cmd_score(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    for path in bundle_dirs(args.models_dir):
        if args.source == "oof":
            produced = score_oof(args.candidate_dir, path)
        else:
            produced = score_in_sample(args.candidate_dir, path)
        records.extend(produced)
        print(f"  {path.name:40} {len(produced):6d} candidates")
    if not records:
        print("Nothing scored. Check --source, --models-dir and the candidate folder.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, SCORED_COLUMNS, records)
    print(f"Wrote {len(records)} scored candidates to {args.out}")
    return 0


# ----------------------------------------------------------------------------- decisions
def parse_number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def decide_single(candidates: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("mode") == "skip":
        return {"status": "skipped", "choice": None, "margin": None, "reason": spec.get("note", "policy: skip")}
    pool = candidates
    if spec.get("mode") == "filter_top1":
        allowed = {str(v).strip().upper() for v in spec.get("allowed", [])}
        pool = [c for c in candidates if str(c["value"]).strip().upper() in allowed]
        if not pool:
            return {"status": "abstain", "choice": None, "margin": None, "reason": "no candidate with an allowed value"}
    if not pool:
        return {"status": "abstain", "choice": None, "margin": None, "reason": "no candidates"}
    top = pool[0]
    runner_up = pool[1]["score"] if len(pool) > 1 else 0.0
    margin = top["score"] - runner_up
    if spec.get("floor") is not None and top["score"] < spec["floor"]:
        return {"status": "flagged", "choice": top, "margin": margin, "reason": f"top score below floor {spec['floor']}"}
    if spec.get("margin") is not None and margin < spec["margin"]:
        return {"status": "flagged", "choice": top, "margin": margin, "reason": f"margin below {spec['margin']}"}
    return {"status": "accepted", "choice": top, "margin": margin, "reason": ""}


def decide_sum(fields: dict[str, list[dict[str, Any]]], equation: list[str], tolerance: float, k: int) -> tuple[float, dict[str, dict[str, Any]]] | None:
    a, b, c = equation
    options = {}
    for field in equation:
        parsed = [(candidate, parse_number(candidate["value"])) for candidate in fields.get(field, [])[:k]]
        options[field] = [(candidate, number) for candidate, number in parsed if number is not None]
        if not options[field]:
            return None
    best = None
    for (ca, va), (cb, vb), (cc, vc) in itertools.product(options[a], options[b], options[c]):
        if abs(va + vb - vc) <= tolerance:
            score = ca["score"] * cb["score"] * cc["score"]
            if best is None or score > best[0]:
                best = (score, {a: ca, b: cb, c: cc})
    return best


def decide_document(doc_type: str, fields: dict[str, list[dict[str, Any]]], all_fields: list[str],
                    policy: dict[str, Any], k: int) -> dict[str, dict[str, Any]]:
    for field in fields:
        fields[field].sort(key=lambda c: -c["score"])
    decisions: dict[str, dict[str, Any]] = {}
    specs = {field: policy.get(f"{doc_type}.{field}") or policy.get("_default") or {"mode": "top1"} for field in all_fields}
    handled_groups = set()
    for field in all_fields:
        spec = specs[field]
        if spec.get("mode") == "amount_sum":
            equation = spec["equation"]
            key = tuple(equation)
            if key in handled_groups:
                continue
            handled_groups.add(key)
            tolerance = float(spec.get("tolerance", 0.01))
            if spec.get("strategy", "confirm") == "search":
                match = decide_sum(fields, equation, tolerance, k)
            else:
                match = decide_sum(fields, equation, tolerance, 1)
            if match:
                for member, candidate in match[1].items():
                    decisions[member] = {"status": "accepted", "choice": candidate, "margin": None, "reason": "amount sum matched"}
                continue
            singles = {member: decide_single(fields.get(member, []), {"mode": "top1"}) for member in equation}
            if spec.get("strategy", "confirm") == "confirm":
                # The top-1 amounts do not add up, so at least one is wrong. Flag only the weakest.
                scored = {m: d["choice"]["score"] for m, d in singles.items() if d["status"] == "accepted"}
                weakest = min(scored, key=scored.get) if scored else None
                for member, single in singles.items():
                    if single["status"] == "accepted" and member == weakest:
                        single["status"] = "flagged"
                        single["reason"] = "top-1 amounts do not add up; weakest flagged"
                    elif single["status"] == "accepted":
                        single["reason"] = "top-1 amounts do not add up"
                    decisions[member] = single
            else:
                for member, single in singles.items():
                    if single["status"] == "accepted":
                        single["reason"] = "no sum match; top-1 used"
                    decisions[member] = single
            continue
        if field not in decisions:
            decisions[field] = decide_single(fields.get(field, []), spec)
    return decisions


def cmd_decide(args: argparse.Namespace) -> int:
    scored = read_rows(args.scored)
    if not scored:
        print("scored file is empty", file=sys.stderr)
        return 1
    policy: dict[str, Any] = {}
    if args.policy.exists():
        policy = yaml.safe_load(args.policy.read_text(encoding="utf-8")) or {}
    else:
        print(f"WARNING: {args.policy} not found; every field uses top1", file=sys.stderr)

    by_doc: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    fields_per_type: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        record = dict(row, score=float(row["score"]))
        by_doc[(row["doc_type"], row["document_id"])][row["field"]].append(record)
        fields_per_type[row["doc_type"]].add(row["field"])

    decision_rows: list[dict[str, Any]] = []
    field_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (doc_type, document_id), fields in sorted(by_doc.items()):
        all_fields = sorted(fields_per_type[doc_type])
        decisions = decide_document(doc_type, fields, all_fields, policy, args.k)
        for field in all_fields:
            candidates = fields.get(field, [])
            decision = decisions.get(field, {"status": "abstain", "choice": None, "margin": None, "reason": "no candidates"})
            labels = [c["label"] for c in candidates if c["label"] in ("0", "1")]
            answerable = any(label == "1" for label in labels) if labels else None
            choice = decision["choice"]
            choice_label = choice["label"] if choice and choice["label"] in ("0", "1") else ""
            if decision["status"] == "accepted":
                correct = (choice_label == "1") if choice_label else None
            elif decision["status"] in ("abstain", "flagged"):
                correct = (answerable is False) if answerable is not None else None
            else:
                correct = None
            top1 = candidates[0] if candidates else None
            top1_correct = None
            if answerable is not None:
                top1_correct = bool(top1 and top1["label"] == "1") if answerable else False
            decision_rows.append({
                "doc_type": doc_type, "field": field, "document_id": document_id, "status": decision["status"],
                "candidate_id": choice["candidate_id"] if choice else "", "value": choice["value"] if choice else "",
                "score": f"{choice['score']:.4f}" if choice else "", "margin": f"{decision['margin']:.4f}" if decision["margin"] is not None else "",
                "reason": decision["reason"], "label": choice_label, "correct": "" if correct is None else correct,
            })
            stats = field_stats[(doc_type, field)]
            stats["documents"] += 1
            stats[decision["status"]] += 1
            if answerable is not None and decision["status"] != "skipped":
                stats["answerable"] += int(answerable)
                stats["labeled_docs"] += 1
                stats["top1_correct"] += int(bool(top1_correct))
                stats["top1_answerable_hit"] += int(bool(answerable and top1_correct))
                if correct is not None:
                    stats["correct"] += int(correct)
                    stats["judged"] += 1
                if decision["status"] == "accepted" and choice_label:
                    stats["accepted_labeled"] += 1
                    stats["accepted_correct"] += int(choice_label == "1")

    summary_rows = []
    for (doc_type, field), stats in sorted(field_stats.items()):
        documents = stats["documents"]
        spec = policy.get(f"{doc_type}.{field}") or policy.get("_default") or {"mode": "top1"}
        labeled = stats["labeled_docs"]
        summary_rows.append({
            "doc_type": doc_type, "field": field, "mode": spec.get("mode", "top1"), "documents": documents,
            "answerable_docs": stats["answerable"], "accepted": stats["accepted"], "flagged": stats["flagged"],
            "abstain": stats["abstain"], "skipped": stats["skipped"],
            "coverage": round(stats["accepted"] / documents, 3) if documents else "",
            "accepted_precision": round(stats["accepted_correct"] / stats["accepted_labeled"], 3) if stats["accepted_labeled"] else "",
            "doc_accuracy": round(stats["correct"] / stats["judged"], 3) if stats["judged"] else "",
            "top1_accuracy": round(stats["top1_correct"] / labeled, 3) if labeled else "",
            "top1_hit_answerable": round(stats["top1_answerable_hit"] / stats["answerable"], 3) if stats["answerable"] else "",
        })

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "decisions.csv", DECISION_COLUMNS, decision_rows)
    write_csv(args.out / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    print(f"{'doc_type':14} {'field':24} {'mode':12} {'acc':>4} {'flag':>4} {'abst':>4} {'skip':>4} {'cov':>6} {'prec':>6} {'policy':>7} {'top1':>6} {'top1 hit*':>9}")
    for row in summary_rows:
        print(f"{row['doc_type']:14} {row['field']:24} {row['mode']:12} {row['accepted']:4d} {row['flagged']:4d} {row['abstain']:4d} "
              f"{row['skipped']:4d} {str(row['coverage']):>6} {str(row['accepted_precision']):>6} {str(row['doc_accuracy']):>7} "
              f"{str(row['top1_accuracy']):>6} {str(row['top1_hit_answerable']):>9}")
    print("\npolicy = document accuracy of this policy (correct accepts + correct abstains) over labeled documents")
    print("top1 = accuracy of always accepting the top candidate.  top1 hit* = top1 correct among documents that have a true candidate")
    print(f"Wrote {args.out / 'decisions.csv'} and {args.out / 'summary.csv'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-document decisions from candidate scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score candidates with trained bundles")
    score.add_argument("--candidate-dir", type=Path, required=True)
    score.add_argument("--models-dir", type=Path, required=True)
    score.add_argument("--source", choices=["oof", "in_sample"], default="oof")
    score.add_argument("--out", type=Path, required=True, help="scored CSV to write")

    decide = sub.add_parser("decide", help="apply the decision policy to scored candidates")
    decide.add_argument("--scored", type=Path, required=True)
    decide.add_argument("--policy", type=Path, default=Path("config/extraction/decision_policy.yaml"))
    decide.add_argument("--out", type=Path, help="output folder (default: folder of --scored)")
    decide.add_argument("--k", type=int, default=5, help="candidates per field tried in amount_sum combinations (default 5)")
    args = parser.parse_args(argv)
    if args.command == "decide":
        args.out = args.out or args.scored.parent
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return cmd_score(args) if args.command == "score" else cmd_decide(args)


if __name__ == "__main__":
    raise SystemExit(main())