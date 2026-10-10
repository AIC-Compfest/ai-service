from __future__ import annotations

from .schema import Candidate, Page, Token


def build_page(page_data: dict) -> Page:
    lines: list[list[Token]] = []
    all_tokens: list[Token] = []
    for line_idx, line in enumerate(page_data.get("lines", [])):
        raw_tokens = line.get("tokens") or []
        if not raw_tokens:
            words = str(line.get("text", "")).split()
            x1, y1, x2, y2 = _bbox(line.get("bbox_norm", [0, 0, 1, 1]))
            width = (x2 - x1) / max(len(words), 1)
            raw_tokens = [{"text": word, "bbox_norm": [x1 + i * width, y1, x1 + (i + 1) * width, y2], "confidence": line.get("confidence", 0), "_estimated": True} for i, word in enumerate(words)]
        tokens = [Token(str(item.get("text", "")), _bbox(item.get("bbox_norm", line.get("bbox_norm", [0, 0, 1, 1]))), float(item.get("confidence", line.get("confidence", 0))), line_idx, i, bool(item.get("_estimated", False))) for i, item in enumerate(raw_tokens) if str(item.get("text", "")).strip()]
        lines.append(tokens)
        all_tokens.extend(tokens)
    return Page(int(page_data.get("page_number", 1)), all_tokens, lines, str(page_data.get("engine_used", "")), page_data.get("quality") or {}, page_data.get("regions") or [])


def context_for(candidate: Candidate) -> dict[str, str]:
    tokens = candidate.page.tokens
    cy = (candidate.bbox[1] + candidate.bbox[3]) / 2
    height = candidate.bbox[3] - candidate.bbox[1]
    own = set(candidate.tokens)
    left = sorted((token for token in tokens if token not in own and token.bbox[2] <= candidate.bbox[0] + .005 and abs(token.cy - cy) <= .6 * max(height, token.bbox[3] - token.bbox[1]) and candidate.bbox[0] - token.bbox[2] <= .4), key=lambda token: (-token.bbox[2], token.tok_idx))[:6]
    right = sorted((token for token in tokens if token not in own and token.bbox[0] >= candidate.bbox[2] - .005 and abs(token.cy - cy) <= .6 * max(height, token.bbox[3] - token.bbox[1]) and token.bbox[0] - candidate.bbox[2] <= .4), key=lambda token: (token.bbox[0], token.tok_idx))[:6]
    same_line = [token for token in tokens if token not in own and abs(token.cy - cy) <= .5 * max(height, token.bbox[3] - token.bbox[1])]
    return {"ctx_left": " ".join(token.text for token in reversed(left)), "ctx_right": " ".join(token.text for token in right), "ctx_line": " ".join(token.text for token in sorted(same_line, key=lambda token: token.bbox[0]))}


def _bbox(value: list[float]) -> tuple[float, float, float, float]:
    values = [float(item) / 1000 if float(item) > 1 else float(item) for item in value[:4]]
    return tuple(max(0.0, min(1.0, value)) for value in values)  # type: ignore[return-value]
