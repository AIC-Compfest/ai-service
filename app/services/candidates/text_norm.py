from __future__ import annotations

import re
import unicodedata


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"[._:/\\|;,()\[\]{}<>#*]", " ", value)
    value = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def norm_alnum(value: str) -> str:
    return re.sub(r"\s+", "", norm_text(value))


def ocr_fold(value: str) -> str:
    replacements = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "|": "l"})

    def fold(match: re.Match[str]) -> str:
        word = match.group(0)
        if not re.search(r"[a-zA-Z]", word) or sum(c.isdigit() for c in word) > len(word) * 0.3:
            return word
        return word.translate(replacements)

    return re.sub(r"[a-zA-Z0-9|]+", fold, value)