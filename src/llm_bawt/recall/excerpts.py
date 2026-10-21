"""Query-relevant ORIGINAL contiguous windows, never stitched/generated snippets."""
from __future__ import annotations

from .contracts import MAX_RECORD_CHARS


def terms_for(query: str) -> list[str]:
    from llm_bawt.memory.postgresql import build_fts_query
    query_text = build_fts_query(query)
    return sorted(set(query_text.split(" | ")), key=lambda word: (-len(word), word))[:16] if query_text else []


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
