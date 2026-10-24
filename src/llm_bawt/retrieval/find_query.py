"""``messages_find`` query parser and fill-only rung ladder (TASK-1323 §5.1).

Ported from the benchmarked r3 prototype (``find_proto2.parse/parts/ladder``),
which is the shape the §4 replay measured. Exact tokens are matched as
case-insensitive substrings (``ILIKE``, backed by the per-leaf trigram index),
so ``TASK-1281`` also finds ``TASK-1281's``; word rungs match the stored
``content_tsv`` and require a share of the query's lexemes.

The ladder is pure data: SQL lives in :mod:`llm_bawt.retrieval.find`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_QUERY_CHARS = 512
MAX_EXACT = 6
MAX_EXACT_CHARS = 64
MAX_WORDS = 12

# TASK-N ids, anything carrying a digit or _ . / : (paths, SHAs, versions,
# snake_case, dates), and camel/Pascal humps.
_IDENT = re.compile(
    r"[A-Za-z]+-\d+|[\w.:/-]*[\d_./:][\w.:/-]*|[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*"
)
_NAMESPACED_ID = re.compile(r"[A-Za-z]+-\d+")
_SHORT_NUMBER = re.compile(r"\d{1,3}")
_QUOTED = re.compile(r'"([^"]*)"')
_EDGES = "'`,;()[]{}"
_OR = re.compile(r"\bOR\b")
_BOOLEAN = re.compile(r"\b(?:OR|AND)\b")
_WORD = re.compile(r"[A-Za-z]{3,}")
_IDENT_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")


class InvalidQuery(ValueError):
    """The query is too long or not text."""


@dataclass(frozen=True)
class ParsedQuery:
    exact: tuple[str, ...]
    words: tuple[str, ...]
    explicit_or: bool

    @property
    def parts(self) -> tuple[str, ...]:
        """Words inside snake/camel/kebab/path identifiers, for the ``parts`` rungs.

        ``TASK-N``-style ids are skipped: their alphabetic half is a namespace,
        not a topic (the F1 failure that made ``TASK-1281`` mean "task").
        """
        out: list[str] = []
        for token in self.exact:
            if _NAMESPACED_ID.fullmatch(token):
                continue
            pieces = [p.lower() for p in _IDENT_PART.findall(token) if len(p) >= 3]
            if len(pieces) >= 2:
                out += [p for p in pieces if p not in out]
        return tuple(out)

    @property
    def terms(self) -> tuple[str, ...]:
        """Everything a snippet may centre on, most specific first."""
        return self.exact + self.words + self.parts


def parse(query: str | None) -> ParsedQuery:
    if query is None:
        query = ""
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS:
        raise InvalidQuery("invalid_query")
    exact = [q for q in _QUOTED.findall(query) if len(q.strip()) >= 3]
    rest = _QUOTED.sub(" ", query)
    kept: list[str] = []
    for raw in rest.split():
        # Sentence punctuation is not part of an identifier: "TASK-1281." is TASK-1281.
        token = raw.strip(_EDGES).rstrip(".:!?")
        if (len(token) >= 3 and _IDENT.fullmatch(token)
                and not _SHORT_NUMBER.fullmatch(token)):
            exact.append(token)
        else:
            kept.append(raw)
    rest = " ".join(kept)
    explicit_or = bool(_OR.search(rest))
    words = _WORD.findall(_BOOLEAN.sub(" ", rest))
    seen: dict[str, str] = {}
    for token in exact:
        seen.setdefault(token.lower(), token[:MAX_EXACT_CHARS])
    return ParsedQuery(tuple(seen.values())[:MAX_EXACT],
                       tuple(dict.fromkeys(words))[:MAX_WORDS], explicit_or)


@dataclass(frozen=True)
class Rung:
    """One match definition.

    ``exact`` tokens are ILIKE substrings joined by ``joiner`` (``all`` or
    ``any``). With no exact tokens the rung is a lexeme match on
    ``content_tsv``: a row must contain at least ``need`` of ``lexemes``.
    ``lexemes`` always feed ranking (coverage, ``ts_rank``).
    """

    name: str
    exact: tuple[str, ...]
    lexemes: tuple[str, ...]
    joiner: str
    need: int


def _half(n: int) -> int:
    return max(1, math.ceil(n / 2))


def _third(n: int) -> int:
    return max(1, math.ceil(n / 3))


def ladder(parsed: ParsedQuery, word_lexemes: tuple[str, ...],
           part_lexemes: tuple[str, ...]) -> tuple[Rung, ...]:
    """Rungs in fill order. ``*_lexemes`` are the english-config lexemes of
    ``parsed.words`` and of ``parsed.parts + parsed.words`` (computed in SQL
    with the same parser as ``content_tsv``)."""
    rungs: list[Rung] = []
    if parsed.exact:
        rungs += [Rung("exact_all", parsed.exact, word_lexemes, "all", 0),
                  Rung("exact_any", parsed.exact, word_lexemes, "any", 0)]
        if part_lexemes:
            n = len(part_lexemes)
            rungs.append(Rung("parts", (), part_lexemes, "", _half(n)))
            if _third(n) < _half(n):
                rungs.append(Rung("parts_loose", (), part_lexemes, "", _third(n)))
    elif word_lexemes:
        n = len(word_lexemes)
        rungs.append(Rung("words", (), word_lexemes, "", 1 if parsed.explicit_or else _half(n)))
        if not parsed.explicit_or and _third(n) < _half(n):
            rungs.append(Rung("words_loose", (), word_lexemes, "", _third(n)))
    return tuple(rungs)
