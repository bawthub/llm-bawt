"""Identifier-preserving full-text query for message search (TASK-1323, F1).

The old builder kept only ``[a-zA-Z]+`` runs, so ``TASK-1281`` searched for
every message containing "task" and ``messages_search_all`` or a commit SHA
vanished. :class:`FtsQuery` keeps those tokens exact and leaves plain words on
the old OR-of-stems path:

* Quoted text (``"stop button"``) and any token carrying an identifier signal
  (digit, ``_ . / : # @ -``, or a camelCase hump) is a phrase.
  ``phraseto_tsquery('english', …)`` parses it with the same parser the stored
  ``to_tsvector('english', content)`` used, so ``TASK-1281`` becomes
  ``'task' <-> '-1281'``: it matches ``TASK-1281`` and not ``TASK-12810``.
* A hex run of 7–40 chars with at least one digit and one letter is a commit
  prefix, ``to_tsquery('simple', '<hex>:*')``.
* Everything else contributes its alphabetic words, minus stop words.

The parts are OR-ed, as before. Identifiers are not decomposed into their
words: that is a recall fallback for the ``messages_find`` ladder, not the
default meaning of an exact token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Stop words for full-text message search queries (moved from memory.postgresql).
FTS_STOP_WORDS: frozenset[str] = frozenset({
    # Standard English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'we', 'you', 'i', 'he', 'she', 'it', 'they', 'them', 'their', 'our',
    'my', 'your', 'his', 'her', 'its', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'not', 'no', 'yes', 'so',
    'if', 'then', 'than', 'too', 'very', 'just', 'about', 'before', 'after',
    # Conversational meta-words (common in memory queries but not content)
    'remember', 'tell', 'told', 'said', 'say', 'know', 'think', 'thought',
    'talk', 'talked', 'talking', 'conversation', 'conversations', 'discussed',
    'discuss', 'discussion', 'mention', 'mentioned', 'anything', 'something',
    'everything', 'nothing', 'past', 'previous', 'earlier', 'last', 'time',
    'when', 'where', 'how', 'why', 'like', 'want', 'wanted', 'please',
})

MAX_EXACT_CHARS = 64
MAX_EXACT_TOKENS = 16
MAX_WORDS = 32

_QUOTED = re.compile(r'"([^"]*)"')
_EDGE_PUNCTUATION = ".,;:!?()[]{}<>'\"`*"
_HEX = re.compile(r"[0-9a-f]{7,40}")
_IDENTIFIER_SIGNAL = re.compile(r"[0-9_./:#@-]|[a-z][A-Z]")
_ALNUM = re.compile(r"[0-9A-Za-z]")
_WORD = re.compile(r"\b[a-z]+\b")


def _is_commit_prefix(token: str) -> bool:
    return (_HEX.fullmatch(token) is not None
            and any(c.isdigit() for c in token) and any(c.isalpha() for c in token))


@dataclass(frozen=True)
class FtsQuery:
    """Parsed search text; render with :meth:`tsquery_sql` and :attr:`params`."""

    words: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()

    @classmethod
    def parse(cls, query: str | None) -> FtsQuery | None:
        """Parse ``query``; ``None`` when nothing searchable remains."""
        if not query:
            return None
        phrases: list[str] = []
        prefixes: list[str] = []
        words: list[str] = []
        fallback: list[str] = []
        loose: list[str] = []
        cursor = 0
        for match in _QUOTED.finditer(query):
            loose.append(query[cursor:match.start()])
            quoted = " ".join(match.group(1).split())
            if len(quoted) <= MAX_EXACT_CHARS and _ALNUM.search(quoted):
                phrases.append(quoted)
            else:
                loose.append(quoted)
            cursor = match.end()
        loose.append(query[cursor:])

        for raw in " ".join(loose).split():
            token = raw.strip(_EDGE_PUNCTUATION)
            if not token or len(token) > MAX_EXACT_CHARS:
                continue
            lowered = token.lower()
            if _is_commit_prefix(lowered):
                prefixes.append(lowered)
            elif _IDENTIFIER_SIGNAL.search(token):
                # A lone digit or bare punctuation is noise, not an identifier.
                if len(_ALNUM.findall(token)) >= 2:
                    phrases.append(token)
            else:
                for word in _WORD.findall(lowered):
                    if len(word) > 2:
                        fallback.append(word)
                        if word not in FTS_STOP_WORDS:
                            words.append(word)

        # Case-insensitive dedupe, first spelling wins; phrases before prefixes.
        exact: dict[tuple[str, str], str] = {}
        for kind, values in (("phrase", phrases), ("prefix", prefixes)):
            for value in values:
                exact.setdefault((kind, value.lower()), value)
        kept = list(exact.items())[:MAX_EXACT_TOKENS]
        kept_phrases = tuple(v for (kind, _), v in kept if kind == "phrase")
        kept_prefixes = tuple(v for (kind, _), v in kept if kind == "prefix")
        kept_words = tuple(dict.fromkeys(words))[:MAX_WORDS]
        if not (kept_phrases or kept_prefixes or kept_words):
            # Old builder's fallback: a stop-word-only query still searches.
            kept_words = tuple(dict.fromkeys(fallback))[:3]
        if not (kept_phrases or kept_prefixes or kept_words):
            return None
        return cls(kept_words, kept_phrases, kept_prefixes)

    def tsquery_sql(self) -> str:
        """SQL ``tsquery`` expression; bind :attr:`params` alongside it."""
        parts = [f"phraseto_tsquery('english', :fts_p{i})" for i in range(len(self.phrases))]
        parts += [f"to_tsquery('simple', :fts_x{i})" for i in range(len(self.prefixes))]
        if self.words:
            parts.append("to_tsquery('english', :fts_w)")
        return "(" + " || ".join(parts) + ")"

    @property
    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {f"fts_p{i}": p for i, p in enumerate(self.phrases)}
        params.update({f"fts_x{i}": f"{x}:*" for i, x in enumerate(self.prefixes)})
        if self.words:
            params["fts_w"] = " | ".join(self.words)
        return params
