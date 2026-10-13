from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .client import LLMResponse


class FakeLLMClient:
    """Canned, network-free client used by tests and local smoke runs."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        self.responses = responses or {}
        self.calls = 0

    def generate_json(self, system: str, parts: list[Any], json_schema: dict[str, Any], *, model: str, temperature: float = 0, max_tokens: int = 4096) -> LLMResponse:
        self.calls += 1
        document_id = next((part for part in parts if isinstance(part, str) and part in self.responses), "")
        parsed = self.responses.get(document_id, {"fields": {}, "tables": []})
        return LLMResponse(json.dumps(parsed, ensure_ascii=False), parsed, {"prompt_tokens": 0, "completion_tokens": 0}, f"fake-{self.calls}")