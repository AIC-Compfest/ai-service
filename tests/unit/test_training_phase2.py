import json

import numpy as np

from app.services.training.bundle import load_bundle, save_bundle
from app.services.training.cv import make_group_folds


def test_group_folds_never_split_templates():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    for train, valid in make_group_folds(groups, n_splits=4, seed=42):
        assert set(groups[train]).isdisjoint(set(groups[valid]))


def test_bundle_hash_refuses_tampering(tmp_path):
    bundle = save_bundle(tmp_path, "D.field.v1", _Model(), {"feature_columns": ["g_cx"]})
    load_bundle(bundle)
    (bundle / "model.joblib").write_bytes(b"tampered")
    try:
        load_bundle(bundle)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered model was accepted")


class _Model:
    def predict_proba(self, values):
        return [[.5, .5] for _ in values]