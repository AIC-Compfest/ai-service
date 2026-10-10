from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ..schema import NumberParse

_CURRENCY = re.compile(r"^(rp\.?|idr|usd|\$)\s*", re.I)


def parse_number(text: str) -> NumberParse:
    raw = (text or "").strip()
    negative = (raw.startswith("(") and raw.endswith(")")) or raw.startswith("-")
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    raw = raw.lstrip("-").strip()
    match = _CURRENCY.match(raw)
    sym = None
    if match:
        sym = match.group(1).lower().rstrip(".")
        raw = raw[match.end():].strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact or re.search(r"[^0-9.,]", compact):
        return NumberParse(None, "invalid", negative=negative, sym=sym)
    separators = set(c for c in compact if c in ".,")
    ambiguous = False
    value_alt = None
    try:
        if "." in separators and "," in separators:
            decimal_sep = "." if compact.rfind(".") > compact.rfind(",") else ","
            thousands_sep = "," if decimal_sep == "." else "."
            normalized = compact.replace(thousands_sep, "").replace(decimal_sep, ".")
            fmt = "en_comma_thousand_dot_decimal" if decimal_sep == "." else "id_dot_thousand_comma_decimal"
        elif separators:
            sep = next(iter(separators))
            if re.fullmatch(rf"\d{{1,3}}(?:{re.escape(sep)}\d{{3}})+", compact):
                normalized = compact.replace(sep, "")
                fmt = "id_dot_thousand" if sep == "." else "en_comma_thousand"
                ambiguous = compact.count(sep) == 1
                if ambiguous:
                    value_alt = Decimal(compact.replace(sep, "."))
            else:
                normalized = compact.replace(sep, ".")
                fmt = "dot_decimal" if sep == "." else "id_comma_decimal"
        else:
            normalized, fmt = compact, "plain"
        value = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return NumberParse(None, "invalid", negative=negative, sym=sym)
    if negative:
        value = -value
        if value_alt is not None:
            value_alt = -value_alt
    return NumberParse(value, fmt, ambiguous, value_alt, negative, sym)