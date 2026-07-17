"""Lightweight tokenization helpers shared by sparse retrieval."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization suitable for biomedical BM25."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
