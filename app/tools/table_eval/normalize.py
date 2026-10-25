"""Text and number normalization used by the table evaluator.

Money is always ``Decimal`` (D-003). Separator rules (documented and unit-tested):

* Currency markers (``Rp``, ``IDR``, ``USD``, ``US$``, ``$``), spaces and NBSP are ignored.
* Parentheses or a leading ``-`` mark a negative value.
* When both ``.`` and ``,`` appear, the separator that appears LAST is the decimal separator
  (``1.250,50`` and ``1,250.50`` both mean 1250.50).
* When one separator type appears several times it is a thousands separator (``1.250.000``).
* When one separator appears once: exactly three digits after it and a 1-3 digit lead without a
  leading zero means thousands (``1.250`` and ``1,250`` are 1250, flagged ambiguous); otherwise it is
  decimal (``20.5``, ``0,75``, ``12.50``, ``0.250``).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_CURRENCY = re.compile(r"(?i)\brp\b\.?|\bidr\b|\busd\b|us\$|\$")
_NUMERIC_BODY = re.compile(r"^\(?-?\d[\d.,]*\)?$")


@dataclass(frozen=True)
class ParsedNumber:
    value: Decimal | None
    ambiguous: bool = False


def clean_text(text: object) -> str:
    """NFKC, NBSP to space, collapse whitespace, strip. Case is preserved."""
    value = unicodedata.normalize("NFKC", "" if text is None else str(text))
    value = value.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", " ", value).strip()


def parse_number(text: object) -> ParsedNumber:
    """Parse an Indonesian or English formatted number. Returns ``value=None`` when not numeric."""
    raw = clean_text(text)
    if not raw:
        return ParsedNumber(None)
    stripped = _CURRENCY.sub("", raw).replace(" ", "")
    negative = False
    if stripped.startswith("(") and stripped.endswith(")"):
        negative, stripped = True, stripped[1:-1]
    if stripped.startswith("-"):
        negative, stripped = True, stripped[1:]
    if not stripped or not re.fullmatch(r"\d[\d.,]*", stripped):
        return ParsedNumber(None)
    if stripped.endswith((".", ",")):
        stripped = stripped[:-1]
    ambiguous = False
    has_dot, has_comma = "." in stripped, "," in stripped
    if has_dot and has_comma:
        decimal_sep = "." if stripped.rfind(".") > stripped.rfind(",") else ","
        thousands = "," if decimal_sep == "." else "."
        digits = stripped.replace(thousands, "").replace(decimal_sep, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        parts = stripped.split(sep)
        if len(parts) > 2:
            if not all(len(part) == 3 for part in parts[1:]) or len(parts[0]) > 3:
                return ParsedNumber(None)
            digits = "".join(parts)
        else:
            lead, tail = parts
            if len(tail) == 3 and 1 <= len(lead) <= 3 and not lead.startswith("0"):
                digits, ambiguous = lead + tail, True
            else:
                digits = f"{lead}.{tail}"
    else:
        digits = stripped
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return ParsedNumber(None)
    return ParsedNumber(-value if negative else value, ambiguous)


def is_numeric(text: object) -> bool:
    return parse_number(text).value is not None


def canonical_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text


def norm_cell(text: object) -> str:
    """Canonical comparison key: ``#<decimal>`` for numbers, case-folded collapsed text otherwise."""
    cleaned = clean_text(text)
    parsed = parse_number(cleaned)
    if parsed.value is not None and _NUMERIC_BODY.match(_CURRENCY.sub("", cleaned).replace(" ", "")):
        return "#" + canonical_decimal(parsed.value)
    return cleaned.casefold()


def norm_text(text: object) -> str:
    return clean_text(text).casefold()
