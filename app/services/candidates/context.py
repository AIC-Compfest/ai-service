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
        tokens = [Token(str(item.get("text", "")), _bbox(item.get("bbox_norm", line.get("bbox_norm", [0, 0, 1, 1]))), float(item.get("confidence", line.get("confidence", 0))), line_idx, i, bool(item.get("_estimated", False)), int(item.get("row_idx", line.get("row_idx", line_idx)))) for i, item in enumerate(raw_tokens) if str(item.get("text", "")).strip()]
        lines.append(tokens)
        all_tokens.extend(tokens)
    return Page(int(page_data.get("page_number", 1)), all_tokens, lines, str(page_data.get("engine_used", "")), page_data.get("quality") or {}, page_data.get("regions") or [], page_data.get("_source_page"), _aspect(page_data), (page_data.get("page_image") or {}).get("path"))


def _aspect(page_data: dict) -> float | None:
    width, height = page_data.get("width"), page_data.get("height")
    try:
        return float(width) / float(height) if width and height else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def context_for(candidate: Candidate) -> dict[str, str]:
    tokens = candidate.page.tokens
    cx = (candidate.bbox[0] + candidate.bbox[2]) / 2
    cy = (candidate.bbox[1] + candidate.bbox[3]) / 2
    height = candidate.bbox[3] - candidate.bbox[1]
    own = set(candidate.tokens)
    left = sorted((token for token in tokens if token not in own and token.bbox[2] <= candidate.bbox[0] + .005 and abs(token.cy - cy) <= .6 * max(height, token.bbox[3] - token.bbox[1]) and candidate.bbox[0] - token.bbox[2] <= .4), key=lambda token: (-token.bbox[2], token.tok_idx))[:6]
    right = sorted((token for token in tokens if token not in own and token.bbox[0] >= candidate.bbox[2] - .005 and abs(token.cy - cy) <= .6 * max(height, token.bbox[3] - token.bbox[1]) and token.bbox[0] - candidate.bbox[2] <= .4), key=lambda token: (token.bbox[0], token.tok_idx))[:6]
    same_line = [token for token in tokens if token not in own and abs(token.cy - cy) <= .5 * max(height, token.bbox[3] - token.bbox[1])]
    above = sorted((token for token in tokens if token not in own and token.bbox[3] <= candidate.bbox[1] + .2 * height and candidate.bbox[1] - token.bbox[3] <= .07 and abs(token.cx - cx) <= .05 + (candidate.bbox[2] - candidate.bbox[0]) / 2), key=lambda token: (-token.bbox[3], token.bbox[0]))[:12]
    below = sorted((token for token in tokens if token not in own and token.bbox[1] >= candidate.bbox[3] - .2 * height and token.bbox[1] - candidate.bbox[3] <= .07 and abs(token.cx - cx) <= .05 + (candidate.bbox[2] - candidate.bbox[0]) / 2), key=lambda token: (token.bbox[1], token.bbox[0]))[:12]
    return {"ctx_left": " ".join(token.text for token in reversed(left)), "ctx_right": " ".join(token.text for token in right), "ctx_above": " ".join(token.text for token in reversed(above)), "ctx_below": " ".join(token.text for token in below), "ctx_line": " ".join(token.text for token in sorted(same_line, key=lambda token: token.bbox[0])), "ctx_sent": " ".join(token.text for token in sorted(same_line + above[:6] + below[:6], key=lambda token: (token.cy, token.bbox[0])))[:300]}


def _bbox(value: list[float]) -> tuple[float, float, float, float]:
    """Validate the canonical OCR coordinate scale: normalized floats [0.0, 1.0]."""
    if len(value) < 4:
        raise ValueError("bbox_norm must contain four normalized coordinates")
    values = [float(item) / 1000 if float(item) > 1 else float(item) for item in value[:4]]
    if any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"bbox_norm must use [0.0, 1.0] coordinates: {value}")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"bbox_norm must have positive area: {value}")
    return tuple(values)  # type: ignore[return-value]
