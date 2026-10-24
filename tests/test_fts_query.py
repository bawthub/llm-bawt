"""TASK-1323 F1: the FTS query keeps identifiers exact (pure parsing, no DB)."""
from __future__ import annotations

import re

import pytest

from llm_bawt.retrieval import FTS_STOP_WORDS, FtsQuery
from llm_bawt.retrieval.fts_query import MAX_EXACT_TOKENS, MAX_WORDS


def parse(query):
    return FtsQuery.parse(query)


@pytest.mark.parametrize(("query", "phrases"), [
    ("TASK-1281", ("TASK-1281",)),
    ("where is messages_search_all", ("messages_search_all",)),
    ("searchAll handler", ("searchAll",)),
    ("src/llm_bawt/memory/postgresql.py", ("src/llm_bawt/memory/postgresql.py",)),
    ("v0.1.83 shipped", ("v0.1.83",)),
    ("2026-10-24", ("2026-10-24",)),
    ("PR #12", ("#12",)),
    ("ops@example.com", ("ops@example.com",)),
])
def test_identifier_tokens_are_exact_phrases(query, phrases):
    assert parse(query).phrases == phrases


def test_identifiers_are_not_decomposed_into_words():
    fts = parse("TASK-1281 messages_search_all")
    assert fts.words == ()
    assert fts.phrases == ("TASK-1281", "messages_search_all")


def test_edge_punctuation_is_stripped_before_classifying():
    fts = parse("(TASK-1281), `messages_search_all`; 'stop' button?")
    assert fts.phrases == ("TASK-1281", "messages_search_all")
    assert fts.words == ("stop", "button")


@pytest.mark.parametrize("sha", ["54386cc", "1d510f8afd511669790dfdfaed15e702700c64d2", "A1B2C3D"])
def test_hex_with_digit_and_letter_is_commit_prefix(sha):
    fts = parse(f"commit {sha}")
    assert fts.prefixes == (sha.lower(),)
    assert fts.params["fts_x0"] == f"{sha.lower()}:*"


@pytest.mark.parametrize(("token", "kind"), [
    ("deadbeef", "words"),           # no digit: an ordinary word
    ("1234567", "phrases"),          # no letter: a number, kept exact
    ("abc123", "phrases"),           # too short for a SHA prefix
    ("g1b2c3d4", "phrases"),         # not hex
])
def test_non_sha_hex_lookalikes(token, kind):
    fts = parse(token)
    assert fts.prefixes == ()
    assert getattr(fts, kind) == (token,)


def test_quoted_text_is_one_phrase_and_keeps_stop_words():
    fts = parse('the "stop button" broke')
    assert fts.phrases == ("stop button",)
    assert fts.words == ("broke",)


def test_quoted_text_over_the_cap_falls_back_to_tokens():
    long_quote = " ".join(["alpha"] * 13)  # 77 chars
    fts = parse(f'"{long_quote}"')
    assert fts.phrases == ()
    assert fts.words == ("alpha",)


def test_words_keep_the_old_or_of_stems_behaviour():
    fts = parse("what did we decide about the stop button")
    assert fts.words == ("decide", "stop", "button")
    assert fts.params == {"fts_w": "decide | stop | button"}
    assert fts.tsquery_sql() == "(to_tsquery('english', :fts_w))"


def test_stop_word_only_query_uses_the_old_fallback():
    fts = parse("what about that they said")
    assert fts.words == ("what", "about", "that")
    assert "what" in FTS_STOP_WORDS


@pytest.mark.parametrize("query", ["", None, "   ", "a I", "1 - ? ...", "a 7"])
def test_nothing_searchable_returns_none(query):
    assert parse(query) is None


def test_case_insensitive_dedupe_keeps_first_spelling():
    fts = parse("TASK-1281 task-1281 stop STOP 54386CC 54386cc")
    assert fts.phrases == ("TASK-1281",)
    assert fts.words == ("stop",)
    assert fts.prefixes == ("54386cc",)


def test_overlong_tokens_are_dropped():
    assert parse("x_" + "y" * 80) is None


def test_caps_bound_the_query():
    exact = " ".join(f"TASK-{n}" for n in range(MAX_EXACT_TOKENS + 5))
    words = " ".join(f"word{chr(97 + n // 26)}{chr(97 + n % 26)}x" for n in range(MAX_WORDS + 5))
    fts = parse(f"{exact} {words}")
    assert len(fts.phrases) == MAX_EXACT_TOKENS
    assert len(fts.words) == MAX_WORDS


def test_sql_or_combines_every_part_with_named_params():
    fts = parse('TASK-1281 "stop button" 54386cc broke')
    assert fts.tsquery_sql() == (
        "(phraseto_tsquery('english', :fts_p0) || phraseto_tsquery('english', :fts_p1)"
        " || to_tsquery('simple', :fts_x0) || to_tsquery('english', :fts_w))"
    )
    assert fts.params == {"fts_p0": "stop button", "fts_p1": "TASK-1281",
                          "fts_x0": "54386cc:*", "fts_w": "broke"}


def test_user_text_reaches_sql_only_as_bound_parameters():
    fts = parse("x'); DROP TABLE messages; -- TASK-1 \"a'b\" 54386cc")
    part = (r"(?:phraseto_tsquery\('english', :fts_p\d+\)|to_tsquery\('simple', :fts_x\d+\)"
            r"|to_tsquery\('english', :fts_w\))")
    assert re.fullmatch(rf"\({part}(?: \|\| {part})*\)", fts.tsquery_sql())
    assert fts.phrases == ("a'b", "TASK-1")
