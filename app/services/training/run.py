from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from app.core.config import settings

from .assemble import assemble
from .bundle import save_bundle
from .cv import make_group_folds
from .metrics import binary_metrics


def train_binary(candidate_dir: str | Path, output_dir: str | Path, doc_type: str, field: str, *, config_path: str | Path | None = None) -> dict:
    config_path = Path(config_path) if config_path is not None else Path(settings.config_dir) / "training" / "training.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = assemble(candidate_dir, doc_type, field, include_ablate=config.get("feature_set", "full") == "full")
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**config.get("lgbm_defaults", {}))
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LightGBM is required for Phase 2 training; install requirements.txt") from exc
    oof = np.zeros(len(data.y), dtype=float)
    for train_idx, valid_idx in make_group_folds(data.groups, seed=int(config.get("seed", 42))):
        if not train_idx or not valid_idx:
            continue
        model.fit(data.x[train_idx], data.y[train_idx], categorical_feature=[index for index, name in enumerate(data.feature_columns) if name in {"g_x_zone", "dt_fmt", "dt_month_lang", "tm_tz", "tm_sep", "id_shape_compact"}])
        oof[valid_idx] = model.predict_proba(data.x[valid_idx])[:, 1]
    model.fit(data.x, data.y)
    metrics = binary_metrics(data.y, oof, float(config.get("t_unknown_floor", 0.4)))
    thresholds = {"t_unknown": float(config.get("t_unknown_floor", 0.4)), "t_auto": float(config.get("t_auto_default", 0.9))}
    manifest = json.loads((Path(candidate_dir) / "feature_manifest.json").read_text(encoding="utf-8"))
    vocab = {name: spec.get("vocab", []) for name, spec in manifest.items() if spec.get("dtype") == "cat"}
    model_id = f"{doc_type}.{field}.binary.v1"
    bundle = save_bundle(output_dir, model_id, model, {"schema_version": "1.0.0", "doc_type": doc_type, "field": field, "cand_type": data.candidate_type, "strategy": "S1", "feature_columns": data.feature_columns, "feature_vocabs": vocab, "thresholds": thresholds, "metrics": metrics, "n_rows": len(data.y)})
    (bundle / "oof_predictions.json").write_text(json.dumps([{"document_id": document, "template_group": group, "score": float(score), "label": int(label)} for document, group, score, label in zip(data.documents, data.groups, oof, data.y)], indent=2) + "\n", encoding="utf-8")
    return {"model_id": model_id, "bundle": str(bundle), "metrics": metrics, "thresholds": thresholds}