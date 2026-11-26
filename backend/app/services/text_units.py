"""Language-aware splitting of text into approximate token units.

Whitespace-delimited words are one unit each. CJK ideographs, kana and
Hangul syllables have no spaces between words, so every such character is
its own unit. This mirrors how the BERT basic tokenizer isolates CJK
characters before WordPiece runs, which keeps chunk sizes predictable for
English and Chinese text alike.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

CJK_CHARACTER_CLASS = (
    "぀-ヿ"  # Hiragana and Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "가-힯"  # Hangul syllables
    "豈-﫿"  # CJK Compatibility Ideographs
)

_UNIT_PATTERN = re.compile(rf"[{CJK_CHARACTER_CLASS}]|[^\s{CJK_CHARACTER_CLASS}]+")

_WORD_OR_CJK_RUN_PATTERN = re.compile(
    rf"[{CJK_CHARACTER_CLASS}]+|[^\W{CJK_CHARACTER_CLASS}]+"
)

_CJK_CHARACTER_PATTERN = re.compile(rf"([{CJK_CHARACTER_CLASS}])")


def iter_unit_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` character offsets of each unit in ``text``."""

    for match in _UNIT_PATTERN.finditer(text):
        yield match.span()


def iter_search_terms(text: str) -> Iterator[str]:
    """Yield words and contiguous CJK runs, without punctuation."""

    for match in _WORD_OR_CJK_RUN_PATTERN.finditer(text):
        yield match.group(0)


def space_cjk_characters(text: str) -> str:
    """Surround every CJK character with spaces.

    SQLite's ``unicode61`` tokenizer treats a run of CJK characters as one
    token. Spacing them out lets the full-text index match single
    characters and phrases inside Chinese, Japanese and Korean text.
    """

    return _CJK_CHARACTER_PATTERN.sub(r" \1 ", text)


def contains_cjk(text: str) -> bool:
    return _CJK_CHARACTER_PATTERN.search(text) is not None
