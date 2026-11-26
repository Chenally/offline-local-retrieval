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


# Function words ignored when measuring how much of a query a passage covers.
STOPWORDS = frozenset(
    """
    a an and are as at be by did do does for from how i in is it of on or
    that the this to was were what when where which who why will with you
    """.split()
)


def query_terms(query: str) -> list[str]:
    """Distinct lowercase terms of a query, without function words.

    If a query consists only of function words they are kept, so a query is
    never left without terms.
    """

    terms: list[str] = []

    for term in iter_search_terms(query.lower()):
        if term not in terms:
            terms.append(term)

    content = [term for term in terms if term not in STOPWORDS]
    return content or terms


def keyword_coverage(query: str, *texts: str) -> float:
    """Fraction of the query's content terms that appear in ``texts``.

    This is an absolute, explainable keyword score ("3 of 4 words found")
    that does not depend on which other documents were retrieved.
    """

    terms = query_terms(query)

    if not terms:
        return 0.0

    haystack = " ".join(texts).lower().replace("_", " ").replace("-", " ")
    words = set(iter_search_terms(haystack))
    compact = "".join(haystack.split())

    matched = sum(
        1
        for term in terms
        if (term in compact if contains_cjk(term) else term in words)
    )
    return matched / len(terms)
