from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .schema import FieldLabel, ReviewState

_LOCK = threading.RLock()


class LabelStore:
    """CSV-backed long-format store with atomic writes and append-only audit entries."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "field_labels.csv"
        self.audit_path = self.root / "audit_log.jsonl"

    def get(self, document_id: str, field: str) -> FieldLabel | None:
        return next((label for label in self.list() if label.key() == (document_id, field)), None)

    def list(self, **filters) -> list[FieldLabel]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8", newline="") as handle:
            values = [_field_label_from_row(row) for row in csv.DictReader(handle)]
        return [value for value in values if all(getattr(value, key) == expected for key, expected in filters.items())]

    def upsert(self, label: FieldLabel, reviewer: str | None = None, reason: str | None = None) -> None:
        with _LOCK:
            existing = self.get(label.document_id, label.field)
            values = [item for item in self.list() if item.key() != label.key()]
            values.append(label)
            self._atomic_write(values)
            if existing is None or _label_signature(existing) != _label_signature(label):
                self.root.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "reviewer": reviewer or label.reviewer, "document_id": label.document_id, "field": label.field, "old": asdict(existing) if existing else None, "new": asdict(label), "reason": reason or ""}, ensure_ascii=False, sort_keys=True) + "\n")

    def queue(self, mode: str = "needs_review", doc_type: str | None = None) -> list[FieldLabel]:
        values = self.list(doc_type=doc_type) if doc_type else self.list()
        if mode == "needs_review":
            return [value for value in values if value.review_state == ReviewState.PENDING]
        if mode == "base_all":
            return [value for value in values if value.is_base]
        if mode == "qa_sample":
            return [value for value in values if value.review_state == ReviewState.AUTO]
        return values

    def stats(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in self.list():
            result[value.status] = result.get(value.status, 0) + 1
            result[value.review_state] = result.get(value.review_state, 0) + 1
        return result

    def lock(self):
        return _LOCK

    def _atomic_write(self, values: Iterable[FieldLabel]) -> None:
        fields = list(asdict(next(iter(values), FieldLabel("", "", "", "", 0, "", "", "", ""))).keys())
        values = sorted(values, key=lambda item: (item.document_id, item.field))
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="field_labels.", suffix=".tmp", dir=self.root)
        os.close(fd)
        try:
            with open(temp_name, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(asdict(value) for value in values)
            for attempt in range(4):
                try:
                    os.replace(temp_name, self.path)
                    break
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def _label_signature(label: FieldLabel) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in asdict(label).items()}


def _field_label_from_row(row: dict[str, str]) -> FieldLabel:
    row = dict(row)
    row["is_base"] = int(row.get("is_base", "0") or 0)
    for key in ("confidence", "grounding_score", "transfer_score", "transfer_margin"):
        row[key] = float(row.get(key, "0") or 0)
    for key in ("page",):
        row[key] = int(row[key]) if row.get(key) else None
    return FieldLabel(**row)