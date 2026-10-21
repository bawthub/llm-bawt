"""Query-relevant ORIGINAL contiguous windows, never stitched/generated snippets."""
from __future__ import annotations

import re

from .contracts import MAX_RECORD_CHARS


MAX_QUERY_TERMS = 16


def query_terms(query: str) -> list[str]:
    """Meaningful lowercase alphanumeric terms, in query order.

    Unlike the shared memory FTS builder, identifier parts survive: the
    `maintenance_model`, `query_bounded` and `TASK-1210` in a question yield
    maintenance/model, query/bounded and task/1210, matching how PostgreSQL's
    english parser splits those same identifiers in stored text.
    """
    from llm_bawt.memory.postgresql import _FTS_STOP_WORDS
    words = re.findall(r"[a-z0-9]+", query.lower())
    meaningful = [word for word in words if word not in _FTS_STOP_WORDS
                  and (len(word) > 2 or (word.isdigit() and len(word) > 1))]
    if not meaningful:
        meaningful = [word for word in words if len(word) > 2][:3]
    return list(dict.fromkeys(meaningful))[:MAX_QUERY_TERMS]


def fts_query(query: str) -> str | None:
    """OR tsquery of alphanumeric terms only; safe for to_tsquery('english', …)."""
    terms = query_terms(query)
    return " | ".join(terms) if terms else None


def terms_for(query: str) -> list[str]:
    return sorted(query_terms(query), key=lambda word: (-len(word), word))


def excerpt_window(content: str, query: str) -> tuple[str, int, bool, str]:
    """Python twin for bounded task API strings; PostgreSQL windows stay in DB."""
    folded = content.lower()
    position = folded.find(query.lower())
    if position < 0:
        for term in terms_for(query):
            position = folded.find(term)
            if position >= 0:
                break
    fallback = position < 0
    start = max(0, position - 80)
    excerpt = content[start:start + MAX_RECORD_CHARS]
    return excerpt, start, len(content) > len(excerpt), "head_fallback" if fallback else "matched"


def sql_window(column: str) -> str:
    """Trusted column literals only. Return excerpt + offsets without full bytes.

    Prefer the whole phrase, then the longest matching meaningful term. FTS can
    stem-match without any literal term (e.g. deploy/deployed); mark that bounded
    prefix fallback explicitly instead of pretending its head is a relevant hit.
    """
    hit = f"""COALESCE(NULLIF(strpos(lower({column}), :phrase), 0),
        (SELECT NULLIF(strpos(lower({column}), term), 0)
         FROM unnest(CAST(:terms AS text[])) term
         WHERE strpos(lower({column}), term) > 0
         ORDER BY length(term) DESC, term LIMIT 1))"""
    start = f"GREATEST(1, COALESCE({hit}, 1) - 80)"
    return f"""substring({column} FROM {start} FOR :chars) AS excerpt,
        ({start}) - 1 AS start_char,
        length({column}) > :chars AS truncated,
        CASE WHEN ({hit}) IS NULL THEN 'head_fallback' ELSE 'matched' END AS window_kind"""
