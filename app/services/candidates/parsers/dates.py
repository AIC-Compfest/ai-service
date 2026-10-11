from __future__ import annotations

import re
from datetime import date

from ..schema import DateParse

_MONTHS = {
    **dict(zip("january february march april may june july august september october november december".split(), range(1, 13))),
    **dict(zip("januari februari maret april mei juni juli agustus september oktober november desember".split(), range(1, 13))),
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5, "jun": 6, "jul": 7,
    "agu": 8, "agt": 8, "agust": 8, "aug": 8, "sep": 9, "sept": 9, "okt": 10,
    "oct": 10, "nov": 11, "des": 12, "dec": 12,
}


def parse_date(text: str) -> DateParse:
    raw = (text or "").strip()
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", raw)
    if match:
        day, month, year = map(int, match.groups())
        two = len(match.group(3)) == 2
        if two:
            year += 2000
        return _make(day, month, year, "dd_mm_yy" if two else "dd_mm_yyyy", "num", False, two, day <= 12 and month <= 12 and day != month, match.group(0)[2])
    match = re.fullmatch(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", raw)
    if match:
        year, month, day = map(int, match.groups())
        return _make(day, month, year, "yyyy_mm_dd", "num", False, False, False, match.group(0)[4])
    match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)[,\s-]+(\d{2,4})|([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{2,4})", raw)
    if not match:
        return DateParse(None)
    if match.group(1):
        day, month_text, year_text, fmt = int(match.group(1)), match.group(2), match.group(3), "dd_mon_yyyy"
    else:
        month_text, day, year_text, fmt = match.group(4), int(match.group(5)), match.group(6), "mon_dd_yyyy"
    normalized = month_text.lower()
    month = _MONTHS.get(normalized)
    fuzzy = False
    if month is None:
        choices = [(sum(a != b for a, b in zip(normalized, item)) + abs(len(normalized) - len(item)), number) for item, number in _MONTHS.items() if len(item) >= 4]
        if choices:
            distance, month = min(choices)
            fuzzy = distance <= 1
    if month is None:
        return DateParse(None)
    year = int(year_text) + (2000 if len(year_text) == 2 else 0)
    lang = "id" if normalized in {"januari", "maret", "mei", "juni", "juli", "agustus", "oktober", "desember"} else "en"
    return _make(day, month, year, fmt, lang, fuzzy, len(year_text) == 2, False, " ")


def _make(day: int, month: int, year: int, fmt: str, lang: str, fuzzy: bool, two: bool, ambiguous: bool, sep: str) -> DateParse:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return DateParse(None, "invalid")
    return DateParse(parsed.isoformat(), fmt, lang, fuzzy, two, ambiguous, sep)