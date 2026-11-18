"""Adversarial-text normalisation shared by PII redactor and injection shield.

Defeats: zero-width chars, Unicode homoglyphs (Cyrillic/Greek lookalikes, fullwidth), and
obfuscated e-mail forms (name[at]domain[dot]com).
"""
from __future__ import annotations

import re
import unicodedata

_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)

# common confusables -> ASCII (Cyrillic / Greek lookalikes)
_CONF = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ո": "n", "ս": "u", "ԛ": "q", "ɡ": "g", "ɑ": "a", "ο": "o",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "І": "I", "Ѕ": "S", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONF_TABLE = {ord(k): v for k, v in _CONF.items()}
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize(text: str) -> str:
    """Canonical form: NFKC, strip zero-width/format chars, fold homoglyphs."""
    t = unicodedata.normalize("NFKC", text).translate(_ZW)
    t = t.translate(_CONF_TABLE)
    t = "".join(c for c in t if unicodedata.category(c) not in ("Cf", "Cc") or c in "\n\t ")
    return t


def deobfuscate_symbols(text: str) -> str:
    """Turn 'name [at] dom [dot] com' / '(at)' / ' at ' / ' dot ' into real separators."""
    t = re.sub(r"\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*", "@", text, flags=re.I)
    t = re.sub(r"\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*", ".", t, flags=re.I)
    # mixed forms: 'name@domain dot org' (symbolic @, spelled dot)
    t = re.sub(r"@\s*([A-Za-z0-9\-]+)((?:\s+dot\s+[A-Za-z0-9\-]+)+)",
               lambda m: "@" + m.group(1) + re.sub(r"\s+dot\s+", ".", m.group(2), flags=re.I), t, flags=re.I)
    return t


def leet_fold(text: str) -> str:
    return text.lower().translate(_LEET)


def norm_query(q: str) -> str:
    """Cache-normal form of a query: lowercase, collapse whitespace, strip trailing punctuation."""
    q = normalize(q).lower()
    q = re.sub(r"[^\w\s\-\[\]]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def approx_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (word / punctuation pieces)."""
    return len(re.findall(r"\w+|[^\w\s]", text))
