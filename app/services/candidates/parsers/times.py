from __future__ import annotations

import re


def parse_time(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    match = re.fullmatch(r"(\d{1,2})[:.](\d{2})(?::(\d{2}))?\s*(AM|PM)?\s*(WIB|WITA|WIT)?", raw, re.I)
    if not match:
        return None
    hour, minute, second, ampm, tz = match.groups()
    hour = int(hour)
    if ampm and hour <= 12:
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    if hour > 23 or int(minute) > 59 or (second and int(second) > 59):
        return None
    return {"value": f"{hour:02d}:{minute}" + (f":{second}" if second else ""), "tz": (tz or "none").lower(), "sep": "dot" if "." in raw else "colon"}