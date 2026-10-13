from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    raw_text: str
    parsed: dict[str, Any]
    usage: dict[str, Any]
    request_id: str = ""


class LLMClient(Protocol):
    def generate_json(self, system: str, parts: list[Any], json_schema: dict[str, Any], *, model: str, temperature: float = 0, max_tokens: int = 4096) -> LLMResponse: ...