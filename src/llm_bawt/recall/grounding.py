"""Map a model-proposed quote back to an exact span of the supplied evidence.

The model only *points* at evidence. Whatever it writes, the packet returns
source bytes: an exact substring of the bounded excerpt, or nothing. Three
formatting edits models routinely make are tolerated when locating the span:
collapsed/expanded whitespace runs (``\\n\\n`` -> ``\\n``), dropped or added
markdown emphasis/code markers at word edges, and ``...``/``…`` elisions inside
a quote. An elided quote grounds to the one contiguous source
span from its first to its last fragment, so omitted text is restored, never
invented. ``cap_span`` then shortens an over-long span to a prefix (still
exact source bytes) instead of discarding the pick. Each elided
fragment needs MIN_ELIDED_FRAGMENT_WORDS words so a common word cannot anchor it,
at most MAX_ELIDED_FRAGMENTS are accepted, and the tightest in-order span wins.
"""
from __future__ import annotations

import re

from .budgets import clip

_ELISION = re.compile(r"\.{3,}|…")
_WORDS_PER_FRAGMENT = 200
# An elided fragment must be distinctive enough to anchor the span: "yes ... done"
# would otherwise bridge from the first "yes" to any later "done".
MIN_ELIDED_FRAGMENT_WORDS = 3
MAX_ELIDED_FRAGMENTS = 4


# Markdown emphasis/code markers models drop or add around words ("**Why it
# fails:**" quoted as "Why it fails:"). Only word-edge runs are optional; a marker
# inside a word (maintenance_model) must still match literally.
_MARKERS = "*_`"
_OPTIONAL_MARKERS = r"[*_`]*"


def _pattern(fragment: str, min_words: int) -> str | None:
    words = [core for core in (word.strip(_MARKERS) for word in fragment.split()) if core]
    if len(words) < min_words or len(words) > _WORDS_PER_FRAGMENT:
        return None
    return r"\s+".join(f"{_OPTIONAL_MARKERS}{re.escape(word)}{_OPTIONAL_MARKERS}" for word in words)


def ground_quote(quote: str, text: str) -> str | None:
    """Exact span of ``text`` the quote refers to, or None if not grounded."""
    if not quote.strip():
        return None
    if quote in text:
        return quote
    # Whitespace drift alone first, so a literal "…" in the source is a word,
    # not an elision point.
    whole = _pattern(quote, 1)
    match = re.search(whole, text) if whole else None
    if match:
        return match.group(0)
    parts = _ELISION.split(quote)
    if len(parts) == 1:
        return None
    fragments = [part for part in parts if part.strip()]
    if not 0 < len(fragments) <= MAX_ELIDED_FRAGMENTS:
        return None
    patterns = [_pattern(part, MIN_ELIDED_FRAGMENT_WORDS) for part in fragments]
    if None in patterns:
        return None
    return _tightest_chain(text, [re.compile(pattern) for pattern in patterns])


def _tightest_chain(text: str, compiled: list[re.Pattern]) -> str | None:
    """Shortest span holding every fragment in order.

    Every fragment-1 occurrence (overlapping ones included) is a candidate
    start. Each later fragment's last match is remembered and reused while it
    still lies after the chain's current end, so text is scanned forward once
    per fragment rather than once per start. For a fixed start the earliest
    chained end is the tightest; a later start can only end later, so the
    first failed chain ends the scan.
    """
    best = None
    last: list[re.Match | None] = [None] * len(compiled)
    first = compiled[0].search(text)
    while first is not None:
        end = first.end()
        for index, pattern in enumerate(compiled[1:], start=1):
            found = last[index]
            if found is None or found.start() < end:
                found = pattern.search(text, end)
                if found is None:
                    return text[best[0]:best[1]] if best else None
                last[index] = found
            end = found.end()
        if best is None or end - first.start() < best[1] - best[0]:
            best = (first.start(), end)
        first = compiled[0].search(text, first.start() + 1)
    return text[best[0]:best[1]] if best else None


def cap_span(span: str, max_chars: int, max_tokens: int) -> str:
    """Prefix of a grounded span within both caps, ending on a natural boundary.

    A prefix of an exact source span is still exact source bytes, so a long
    pick is shortened rather than discarded. The caller must mark it: the cut
    can remove a trailing qualifier or correction.
    """
    capped = clip(span, max_tokens)[:max_chars]
    if capped == span:
        return span
    # Prefer a sentence/line end, then a word break, in the latter half.
    for pattern in (r"[.!?](?=\s)|\n", r"\s"):
        ends = [m.end() if not m.group(0).isspace() else m.start()
                for m in re.finditer(pattern, capped) if m.end() >= len(capped) // 2]
        if ends:
            return capped[:ends[-1]].rstrip()
    return capped.rstrip()
