"""Query-relevant ORIGINAL contiguous windows, never stitched/generated snippets."""
from __future__ import annotations

import re

from .contracts import MAX_RECORD_CHARS


MAX_QUERY_TERMS = 16
# A window opens this many characters before the hit, for leading context.
WINDOW_LEAD_CHARS = 80
# The opening is moved back to the start of the word it lands in, so neither the
# excerpt nor a quote of its head begins mid-word ("efore the backfill").
# Bounded: a longer unbroken run (URL, hash, blob) keeps the raw offset.
MAX_WORD_SNAP_CHARS = 40


def _word_start(content: str, raw: int) -> int:
    """``raw`` moved back to the start of its word; mirrors ``sql_window``."""
    floor = max(0, raw - MAX_WORD_SNAP_CHARS)
    lead = len(re.search(r"\S*$", content[floor:raw]).group(0))
    return raw if lead == raw - floor and floor > 0 else raw - lead


def query_terms(query: str) -> list[str]:
    """Meaningful lowercase alphanumeric terms, in query order.

    Unlike :class:`llm_bawt.retrieval.FtsQuery`, identifiers are decomposed: the
    `maintenance_model`, `query_bounded` and `TASK-1210` in a question yield
    maintenance/model, query/bounded and task/1210, matching how PostgreSQL's
    english parser splits those same identifiers in stored text.
    """
    from llm_bawt.retrieval.fts_query import FTS_STOP_WORDS
    words = re.findall(r"[a-z0-9]+", query.lower())
    meaningful = [word for word in words if word not in FTS_STOP_WORDS
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
    start = _word_start(content, max(0, position - WINDOW_LEAD_CHARS))
    excerpt = content[start:start + MAX_RECORD_CHARS]
    return excerpt, start, len(content) > len(excerpt), "head_fallback" if fallback else "matched"


def term_coverage(content: str, query: str) -> int:
    """Distinct meaningful query terms present in ``content`` (case-insensitive)."""
    folded = content.lower()
    return sum(term in folded for term in query_terms(query))


def focused_window(content: str, query: str, width: int) -> tuple[str, int, bool, str]:
    """The ``width``-char original window covering the most distinct query terms.

    ``excerpt_window`` opens at the first hit, which in a long task part is often
    a passing mention in its first line. Recall hands each record to the model
    as a short prefix, so the window has to open where the answer is: at the
    candidate start (one per term occurrence, bounded) whose span holds the most
    distinct terms, earliest on ties. Same word snap and fallback as
    ``excerpt_window``.
    """
    folded = content.lower()
    terms = terms_for(query)
    phrase = folded.find(query.lower())
    starts = {phrase} if phrase >= 0 else set()
    for term in terms:
        position = folded.find(term)
        while position >= 0 and len(starts) < 64:
            starts.add(position)
            position = folded.find(term, position + 1)
    if not starts:
        excerpt = content[:width]
        return excerpt, 0, len(content) > len(excerpt), "head_fallback"
    best_start, best_score = 0, -1
    for position in sorted(starts):
        start = _word_start(content, max(0, position - WINDOW_LEAD_CHARS))
        span = folded[start:start + width]
        score = sum(term in span for term in terms) + (phrase >= 0 and query.lower() in span)
        if score > best_score:
            best_start, best_score = start, score
    excerpt = content[best_start:best_start + width]
    return excerpt, best_start, len(content) > len(excerpt), "matched"


def sql_window(column: str, alias: str = "win") -> tuple[str, str]:
    """Trusted column literals only. Return excerpt + offsets without full bytes.

    Returns ``(columns, lateral)``: select-list columns reading ``alias``, and the
    ``CROSS JOIN LATERAL`` clause that computes the window start once per row.
    Prefer the whole phrase, then the longest matching meaningful term. FTS can
    stem-match without any literal term (e.g. deploy/deployed); mark that bounded
    prefix fallback explicitly instead of pretending its head is a relevant hit.
    The start then moves back to its word start by the same rule as _word_start:
    unless the non-space run fills the whole bounded look-back and the text
    continues before it.
    """
    hit = f"""COALESCE(NULLIF(strpos(lower({column}), :phrase), 0),
        (SELECT NULLIF(strpos(lower({column}), term), 0)
         FROM unnest(CAST(:terms AS text[])) term
         WHERE strpos(lower({column}), term) > 0
         ORDER BY length(term) DESC, term LIMIT 1))"""
    lateral = f"""CROSS JOIN LATERAL (
        SELECT CASE WHEN lead = raw - floor AND floor > 1 THEN raw ELSE raw - lead END AS start,
               fallback
        FROM (SELECT raw, floor, fallback,
                     length(substring(substring({column} FROM floor FOR raw - floor) FROM '\\S*$')) AS lead
              FROM (SELECT raw, GREATEST(1, raw - {MAX_WORD_SNAP_CHARS}) AS floor, fallback
                    FROM (SELECT GREATEST(1, COALESCE(pos, 1) - {WINDOW_LEAD_CHARS}) AS raw,
                                 pos IS NULL AS fallback
                          FROM (SELECT {hit} AS pos) hit_position) raw_start) look_back) word_lead
    ) {alias}"""
    columns = f"""substring({column} FROM {alias}.start FOR :chars) AS excerpt,
        {alias}.start - 1 AS start_char,
        length({column}) > :chars AS truncated,
        CASE WHEN {alias}.fallback THEN 'head_fallback' ELSE 'matched' END AS window_kind"""
    return columns, lateral
