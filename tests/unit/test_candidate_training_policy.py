import json
from pathlib import Path

from app.services.candidates.training_schema import classify_column, load_policy


def test_training_policy_drops_literal_and_context_columns():
    policy = load_policy(Path("config/candidates"))
    assert classify_column("m_raw_text", policy)[1] == "always_drop"
    assert classify_column("ctx_left", policy)[1] == "always_drop"
    assert classify_column("a_value", policy)[1] == "drop_default"
    assert classify_column("a_log10", policy)[1] == "ablate"
    assert classify_column("g_cx", policy)[1] == "keep"