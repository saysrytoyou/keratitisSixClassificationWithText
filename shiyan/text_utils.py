from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_clinical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()
