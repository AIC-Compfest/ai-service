"""
utils.py — Shared spatial & regex parsing helpers for Phase 6 Structured Extractors.

Design principles (PRD §18, §23, §49, §60):
- All helpers are vendor-agnostic: they use anchor keywords + spatial proximity,
  NEVER hardcoded pixel coordinates.
- Every returned value carries its source bbox_norm and confidence so callers
  can build a full EvidenceSource without extra lookups.
- When a pattern is not found, helpers return (None, 0.0, None) — callers mark
  the field is_unknown=True (No Hallucination Rule, PRD §60).
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.evidence import BoundingBox, EvidenceSource, ExtractedField, ExtractionMethod


# ---------------------------------------------------------------------------
# Low-level line helpers
# ---------------------------------------------------------------------------

def _bbox_norm_to_model(bbox_norm: list[float] | None) -> BoundingBox | None:
    """Convert [x1, y1, x2, y2] list from OCR evidence to BoundingBox model normalized to [0.0, 1.0]."""
    if bbox_norm and len(bbox_norm) == 4:
        x1, y1, x2, y2 = float(bbox_norm[0]), float(bbox_norm[1]), float(bbox_norm[2]), float(bbox_norm[3])
        if max(x1, y1, x2, y2) > 1.0:
            x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        return BoundingBox(
            x1=max(0.0, min(1.0, round(x1, 4))),
            y1=max(0.0, min(1.0, round(y1, 4))),
            x2=max(0.0, min(1.0, round(x2, 4))),
            y2=max(0.0, min(1.0, round(y2, 4))),
        )
    return None


def find_token_bbox_for_value(line: dict, value_substr: str) -> list[float] | None:
    """Extract precise token-level bounding box matching value substring from OCR rather than entire cluster."""
    tokens = line.get("tokens", [])
    if not tokens or not value_substr:
        return line.get("bbox_norm")

    val_clean = re.sub(r'[^a-zA-Z0-9]', '', str(value_substr)).lower()
    if not val_clean:
        return line.get("bbox_norm")

    for t in tokens:
        t_text = re.sub(r'[^a-zA-Z0-9]', '', t.get("text", "")).lower()
        if t_text and (val_clean in t_text or t_text in val_clean):
            return t.get("bbox_norm") or line.get("bbox_norm")
    return line.get("bbox_norm")


def to_unit_bbox(bbox) -> list[float]:
    """Return ``bbox`` on the canonical 0..1 scale.

    Production evidence is already 0..1; legacy fixtures use 0..1000 and are detected the same way as
    ``_bbox_norm_to_model`` does (any coordinate above 1.0).
    """
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    scale = 1000.0 if max(bbox) > 1.0 else 1.0
    return [float(value) / scale for value in bbox[:4]]


def _midpoint_y(bbox_norm: list[float]) -> float:
    return (bbox_norm[1] + bbox_norm[3]) / 2.0


def _midpoint_x(bbox_norm: list[float]) -> float:
    return (bbox_norm[0] + bbox_norm[2]) / 2.0


# ---------------------------------------------------------------------------
# Core spatial parsers
# ---------------------------------------------------------------------------

def find_key_value_horizontal(
    lines: list[dict],
    key_patterns: list[str],
    *,
    y_tolerance: float = 20.0,
    skip_tokens: int = 0,
) -> tuple[str | None, float, list[float] | None]:
    """
    Search for a keyword anchor and return value token(s) to its RIGHT
    on approximately the same Y-axis (either within the same OCRLine or on an
    adjacent OCRLine at the same vertical level).
    """
    key_patterns_lower = [p.lower() for p in key_patterns]

    for i, line in enumerate(lines):
        line_text = line.get("text", "")
        line_text_lower = line_text.lower()
        if not any(p in line_text_lower for p in key_patterns_lower):
            continue

        # Check if the line has content after the keyword on the same line
        for pattern in key_patterns_lower:
            if pattern in line_text_lower:
                idx = line_text_lower.index(pattern)
                after = line_text[idx + len(pattern):].strip()
                after = after.lstrip(":=- ").strip()
                if after:
                    return after, float(line.get("confidence", 0.90)), line.get("bbox_norm")

        # If line had only the keyword (or ends after keyword), check other lines on the same Y row to the right
        anchor_bbox = line.get("bbox_norm", [0, 0, 0, 0])
        anchor_y = _midpoint_y(anchor_bbox)
        anchor_x2 = anchor_bbox[2]

        right_lines = [
            l for l in lines
            if l != line
            and abs(_midpoint_y(l.get("bbox_norm", [0, 0, 0, 0])) - anchor_y) <= y_tolerance
            and l.get("bbox_norm", [0, 0, 0, 0])[0] >= anchor_x2 - 10.0
        ]
        if right_lines:
            # Sort by x1 (left to right)
            right_lines.sort(key=lambda l: l.get("bbox_norm", [0, 0, 0, 0])[0])
            val_text = " ".join(l.get("text", "").strip() for l in right_lines).strip()
            val_text = val_text.lstrip(":=- ").strip()
            if val_text:
                return val_text, float(right_lines[0].get("confidence", 0.90)), right_lines[0].get("bbox_norm")

    return None, 0.0, None


def find_key_value_vertical(
    lines: list[dict],
    key_patterns: list[str],
    *,
    x_tolerance: float = 120.0,
    max_y_gap: float = 80.0,
) -> tuple[str | None, float, list[float] | None]:
    """
    Search for a keyword anchor then return the value found on the line
    BELOW it, aligned on approximately the same X-axis.
    """
    key_patterns_lower = [p.lower() for p in key_patterns]

    for i, line in enumerate(lines):
        line_text = line.get("text", "")
        line_text_lower = line_text.lower()
        if not any(p in line_text_lower for p in key_patterns_lower):
            continue

        anchor_bbox = line.get("bbox_norm", [0, 0, 0, 0])
        anchor_x = _midpoint_x(anchor_bbox)
        anchor_y_end = anchor_bbox[3]

        # Look at subsequent lines
        for j in range(i + 1, min(i + 8, len(lines))):
            candidate = lines[j]
            cand_bbox = candidate.get("bbox_norm", [0, 0, 0, 0])
            cand_x = _midpoint_x(cand_bbox)
            cand_y_start = cand_bbox[1]

            if cand_y_start < anchor_y_end - 5.0:
                continue
            if cand_y_start - anchor_y_end > max_y_gap:
                break
            if abs(cand_x - anchor_x) <= x_tolerance or (cand_bbox[0] >= anchor_bbox[0] - 20.0 and cand_bbox[2] <= anchor_bbox[2] + 200.0):
                text = candidate.get("text", "").strip()
                if text and not any(p in text.lower() for p in key_patterns_lower):
                    return text, float(candidate.get("confidence", 0.90)), cand_bbox

    return None, 0.0, None


def find_amount_near_keyword(
    lines: list[dict],
    key_patterns: list[str],
    *,
    y_tolerance: float = 25.0,
) -> tuple[float | None, float, list[float] | None]:
    """
    Find a numeric monetary amount on the same line or right-aligned adjacent line
    at the same Y level as a keyword anchor. Returns (parsed_float, confidence, bbox).
    """
    key_patterns_lower = [p.lower() for p in key_patterns]
    SKIP_HEADER_PATTERNS = ["tax invoice", "faktur pajak", "pajak angkutan", "npwp", "page", "sr."]

    for line in lines:
        line_text = line.get("text", "")
        line_text_lower = line_text.lower()
        if any(shp in line_text_lower for shp in SKIP_HEADER_PATTERNS) and not any(kw in line_text_lower for kw in ["ppn", "total", "subtotal", "taxes (", "tax (", "tax:"]):
            continue
        if not any(p in line_text_lower for p in key_patterns_lower):
            continue

        anchor_bbox = line.get("bbox_norm", [0, 0, 0, 0])
        anchor_y = _midpoint_y(anchor_bbox)
        anchor_x2 = anchor_bbox[2]

        # Strip percentage expressions like (11%), 14 %, 6%, (11 %)
        line_clean = re.sub(r"\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?", "", line_text)
        
        # Check if amount exists on the same line (after stripping percentage)
        amt_match = parse_amount(line_clean)
        if amt_match and amt_match > 100:
            return amt_match, float(line.get("confidence", 0.90)), anchor_bbox

        # 2. Check other lines on the same Y row to the right (multi-column tables/totals)
        same_row_lines = [
            l for l in lines
            if l != line
            and abs(_midpoint_y(l.get("bbox_norm", [0, 0, 0, 0])) - anchor_y) <= 10.0
            and l.get("bbox_norm", [0, 0, 0, 0])[0] >= anchor_x2 - 10.0
        ]
        # Sort by closest Y level first, then left to right
        same_row_lines.sort(key=lambda l: (
            abs(_midpoint_y(l.get("bbox_norm", [0, 0, 0, 0])) - anchor_y),
            l.get("bbox_norm", [0, 0, 0, 0])[0]
        ))
        for rline in same_row_lines:
            rtext = rline.get("text", "")
            rtext_clean = re.sub(r"\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?", "", rtext)
            ramt = parse_amount(rtext_clean)
            if ramt and ramt > 100:
                return ramt, float(rline.get("confidence", 0.90)), rline.get("bbox_norm")

    return None, 0.0, None


_HEADER_TOKEN = re.compile(r"[a-z0-9]+")
_SHORT_ALIAS = 2  # aliases up to this many characters ("no", "kg") must match the whole header


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    return bool(needle) and any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _header_score(header: str, alias: str) -> int:
    """3 = identical tokens, 2 = alias is a whole-word run inside the header, 1 = header is a run inside the alias, 0 = no match.

    Matching is token based (never a raw substring) and empty headers never match, so ``"" in alias`` and
    ``"no" in "nomor"`` style accidents cannot happen (T-15).
    """
    h, a = _HEADER_TOKEN.findall(header.casefold()), _HEADER_TOKEN.findall(alias.casefold())
    if not h or not a:
        return 0
    if h == a:
        return 3
    if len("".join(a)) <= _SHORT_ALIAS:
        return 0
    if _contains_run(h, a):
        return 2
    if len("".join(h)) >= 3 and _contains_run(a, h):
        return 1
    return 0


def _map_headers(headers: list[str], column_mappings: dict[str, list[str]]) -> dict[str, int]:
    """Assign each column to at most one canonical field (best score, then longest alias) and each field to one column.

    Ties between columns keep the FIRST column; the previous behaviour let the last matching column overwrite.
    """
    best_per_column: dict[int, tuple[tuple[int, int], str]] = {}
    for col_i, header in enumerate(headers):
        for canonical, aliases in column_mappings.items():
            for alias in aliases:
                score = _header_score(header or "", alias)
                if score and ((score, len(alias)) > best_per_column.get(col_i, ((0, 0), ""))[0]):
                    best_per_column[col_i] = ((score, len(alias)), canonical)
    chosen: dict[str, tuple[tuple[int, int], int]] = {}
    for col_i in sorted(best_per_column):
        key, canonical = best_per_column[col_i]
        if canonical not in chosen or key > chosen[canonical][0]:
            chosen[canonical] = (key, col_i)
    return {canonical: col_i for canonical, (_, col_i) in chosen.items()}


def parse_table_columns(
    tables: list[dict],
    column_mappings: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """
    Convert Phase 5 table rows into structured dicts using fuzzy column-header matching.

    Args:
        tables:          List of table dicts from Phase 5 evidence pages.
        column_mappings: Dict of canonical_name -> [alias1, alias2, ...] for
                         matching against the actual header strings found in the table.

    Returns:
        List of row dicts keyed by canonical_name (missing cols are None).
    """
    results = []
    for table in tables:
        headers: list[str] = table.get("headers", [])
        rows: list[list[str]] = table.get("rows", [])
        if not headers or not rows:
            continue

        header_index = _map_headers(headers, column_mappings)

        for row in rows:
            row_dict: dict[str, Any] = {}
            for canonical in column_mappings:
                col_i = header_index.get(canonical)
                if col_i is not None and col_i < len(row):
                    row_dict[canonical] = row[col_i].strip() if row[col_i] else None
                else:
                    row_dict[canonical] = None
            results.append(row_dict)

    return results


def extract_regex_with_bbox(
    lines: list[dict],
    pattern: str,
    *,
    flags: int = re.IGNORECASE,
) -> tuple[str | None, float, list[float] | None]:
    """
    Search all lines for a regex pattern and return (matched_text, confidence, bbox_norm).
    Uses token-level matching when available for tighter bounding boxes.
    Follows No Hallucination Rule (PRD §60): returns (None, 0.0, None) if unmatched.
    """
    compiled = re.compile(pattern, flags)

    for line in lines:
        # First try full line text
        match = compiled.search(line.get("text", ""))
        if match:
            matched_str = match.group(0).strip()
            # Attempt to find the token that best contains the match
            for tok in line.get("tokens", []):
                if compiled.search(tok.get("text", "")):
                    return matched_str, float(tok.get("confidence", 0.90)), tok.get("bbox_norm")
            # Fallback to line bbox
            return matched_str, float(line.get("confidence", 0.90)), line.get("bbox_norm")

    return None, 0.0, None


# ---------------------------------------------------------------------------
# Value normalisation helpers
# ---------------------------------------------------------------------------

def parse_date_string(raw: str | None) -> str | None:
    """
    Parse date string into ISO format YYYY-MM-DD (PRD §24).
    Supports ISO (YYYY-MM-DD), Slash/Dash (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD,MM,YYYY),
    and Indonesian month names ("15 Juli 2026", "15-Jul-2026").
    """
    if not raw:
        return None
    raw = str(raw).strip()

    MONTH_MAP = {
        "januari": "01", "jan": "01",
        "februari": "02", "feb": "02",
        "maret": "03", "mar": "03",
        "april": "04", "apr": "04",
        "mei": "05", "may": "05",
        "juni": "06", "jun": "06",
        "juli": "07", "jul": "07",
        "agustus": "08", "agu": "08", "aug": "08",
        "september": "09", "sep": "09",
        "oktober": "10", "okt": "10", "oct": "10",
        "november": "11", "nov": "11",
        "desember": "12", "des": "12", "dec": "12",
    }

    # 1. ISO format: YYYY-MM-DD
    m = re.search(r"\b(\d{4})[/\-.,\s]+(\d{1,2})[/\-.,\s]+(\d{1,2})\b", raw)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # 2. DD/MM/YYYY format: DD/MM/YYYY with multiple or mixed delimiters (e.g. 10,/08/2026, 03.08/2026)
    m = re.search(r"\b(\d{1,2})[/\-.,\s]+(\d{1,2})[/\-.,\s]+(\d{4})\b", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    # 2b. Concatenated DDMM/YYYY format (e.g. 0407/2026)
    m = re.search(r"\b(\d{2})(\d{2})[/\-.,\s]+(\d{4})\b", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    # 3. Indonesian text date: DD Month YYYY
    m = re.search(
        r"\b(\d{1,2})[\s\-_]+([A-Za-z]+)[\s\-_]+(\d{4})\b",
        raw,
        re.IGNORECASE,
    )
    if m:
        day = m.group(1).zfill(2)
        month_str = m.group(2).lower()
        year = m.group(3)
        month = MONTH_MAP.get(month_str)
        if month:
            return f"{year}-{month}-{day}"

    return None


def parse_weight(raw: str | None) -> float | None:
    """
    Parse weight from a raw string. Supports "15 KG", "15.5kg", "15,5 Kgs".
    Returns float kg value or None (PRD §24 Weight normalisation).
    """
    if not raw:
        return None
    m = re.search(r"(\d+[.,]?\d*)", raw.replace(",", "."))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def parse_amount(raw: str | None) -> float | None:
    """
    Parse IDR / currency monetary amount from raw string:
    - "Rp 75.000", "IDR 75,000", "617,190.00", "IDR 617,190.00", "832.200", "Rp 67,891.00"
    Supports both Indonesian (dot thousand, comma decimal) and US (comma thousand, dot decimal) styles.
    Returns float or None.
    """
    if not raw:
        return None
    raw_str = str(raw).strip()
    # Remove currency symbols and word tokens
    cleaned = re.sub(r"[RpIDR$\s]", "", raw_str, flags=re.IGNORECASE).strip()
    if not cleaned:
        return None

    # Format 1: 617,190.00 or 617,190.50 (comma thousands, dot decimal cents)
    if re.search(r"^\d{1,3}(?:,\d{3})*\.\d{1,2}$", cleaned):
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            pass

    # Format 2: 617.190,00 or 617.190,50 (dot thousands, comma decimal cents)
    if re.search(r"^\d{1,3}(?:\.\d{3})*,\d{1,2}$", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            pass

    # Format 3: Standard dot thousands without decimal (e.g. 832.200, 1.234.567)
    if re.search(r"^\d{1,3}(?:\.\d{3})+$", cleaned):
        try:
            return float(cleaned.replace(".", ""))
        except ValueError:
            pass

    # Format 4: Standard comma thousands without decimal (e.g. 832,200, 1,234,567)
    if re.search(r"^\d{1,3}(?:,\d{3})+$", cleaned):
        try:
            return float(cleaned.replace(",", ""))
        except ValueError:
            pass

    # Generic fallback: strip all formatting except last decimal separator if any
    cleaned_num = re.sub(r"[^\d.,]", "", cleaned)
    if not cleaned_num:
        return None
    if "." in cleaned_num and "," in cleaned_num:
        if cleaned_num.rfind(".") > cleaned_num.rfind(","):
            cleaned_num = cleaned_num.replace(",", "")
        else:
            cleaned_num = cleaned_num.replace(".", "").replace(",", ".")
        try:
            return float(cleaned_num)
        except ValueError:
            pass

    if "." in cleaned_num:
        parts = cleaned_num.split(".")
        if len(parts[-1]) == 2:
            cleaned_num = "".join(parts[:-1]) + "." + parts[-1]
        else:
            cleaned_num = "".join(parts)
    elif "," in cleaned_num:
        parts = cleaned_num.split(",")
        if len(parts[-1]) == 2:
            cleaned_num = "".join(parts[:-1]) + "." + parts[-1]
        else:
            cleaned_num = "".join(parts)

    try:
        return float(cleaned_num)
    except ValueError:
        return None


def parse_integer(raw: str | None, *, reject_year: bool = False) -> int | None:
    """
    Parse integer from raw string (quantity, dispute_window_days, etc.).
    If reject_year=True, silently skips 4-digit numbers in range 2000-2099
    (avoids year tokens from date fields polluting quantity/count fields).
    """
    if not raw:
        return None
    for m in re.finditer(r"\d+", raw):
        val = int(m.group(0))
        if reject_year and 2000 <= val <= 2099:
            continue
        return val
    return None


# ---------------------------------------------------------------------------
# ExtractedField builder helper
# ---------------------------------------------------------------------------

def build_field(
    *,
    field: str,
    value: Any,
    confidence: float,
    bbox_norm: list[float] | None,
    document_id: str,
    page: int,
    method: ExtractionMethod = ExtractionMethod.OCR,
) -> ExtractedField:
    """
    Build an ExtractedField with full EvidenceSource (PRD §18).
    If value is None, automatically marks is_unknown=True and confidence=0.0.

    EvidenceSource is ALWAYS created (even when bbox is None) so that
    benchmark traceability checks do not falsely fail.
    """
    is_unknown = value is None
    return ExtractedField(
        field=field,
        value=value,
        confidence=0.0 if is_unknown else max(0.0, min(1.0, confidence)),
        source=EvidenceSource(
            document_id=document_id,
            page=page,
            bbox=_bbox_norm_to_model(bbox_norm),   # May be None — always create source
        ),
        method=method,
        is_unknown=is_unknown,
    )
