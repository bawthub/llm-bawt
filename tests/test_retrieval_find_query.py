"""TASK-1323 §5.1: the ``messages_find`` parser and fill-only rung ladder."""
from __future__ import annotations

import pytest

from llm_bawt.retrieval.find_query import (
    MAX_EXACT, MAX_QUERY_CHARS, InvalidQuery, ladder, parse,
)


@pytest.mark.parametrize(("query", "exact"), [
    ("TASK-1281", ("TASK-1281",)),
    ("what about TASK-1281.", ("TASK-1281",)),
    ("(TASK-1281)", ("TASK-1281",)),
    ("messages_search_all", ("messages_search_all",)),
    ("54386cc0", ("54386cc0",)),
    ("src/llm_bawt/memory/postgresql.py", ("src/llm_bawt/memory/postgresql.py",)),
    ("searchAll", ("searchAll",)),
    ("SearchAllTools", ("SearchAllTools",)),
    ("v0.1.83", ("v0.1.83",)),
    ("2026-10-24", ("2026-10-24",)),
    ('"press the stop"', ("press the stop",)),
    ("stop button", ()),
    ("fix 12 bugs", ()),
])
def test_exact_tokens(query, exact):
    assert parse(query).exact == exact


def test_words_keep_order_dedupe_and_drop_booleans():
    parsed = parse("stop button OR stop the restart AND ok")
    assert parsed.words == ("stop", "button", "the", "restart")
    assert parsed.explicit_or


def test_exact_tokens_dedupe_case_insensitively_and_cap():
    assert parse("TASK-1 task-1").exact == ("TASK-1",)
    many = " ".join(f"TASK-{i}" for i in range(MAX_EXACT + 3))
    assert len(parse(many).exact) == MAX_EXACT
    assert len(parse("a_" + "b" * 100).exact[0]) == 64


def test_short_quoted_phrases_are_ignored():
    assert parse('"ok" then').exact == ()


def test_parts_skip_namespaced_ids():
    assert parse("TASK-1281").parts == ()
    assert parse("messages_search_all searchAll").parts == ("messages", "search", "all")
    assert parse("TASK-1 stop").terms == ("TASK-1", "stop")


@pytest.mark.parametrize("bad", ["x" * (MAX_QUERY_CHARS + 1), 42])
def test_invalid_query(bad):
    with pytest.raises(InvalidQuery):
        parse(bad)


def test_empty_query_parses_to_nothing():
    assert parse(None).exact == parse("").exact == () and parse("  ").words == ()


def test_ladder_with_exact_tokens():
    parsed = parse("TASK-1 messages_search_all")
    rungs = ladder(parsed, (), ("messag", "search", "all"))
    assert [(r.name, r.joiner, r.need) for r in rungs] == [
        ("exact_all", "all", 0), ("exact_any", "any", 0), ("parts", "", 2), ("parts_loose", "", 1)]
    assert rungs[0].exact == ("TASK-1", "messages_search_all")


def test_ladder_skips_parts_loose_when_it_equals_parts():
    rungs = ladder(parse("alpha_beta"), (), ("alpha", "beta"))
    assert [r.name for r in rungs] == ["exact_all", "exact_any", "parts"]


def test_ladder_words():
    assert [(r.name, r.need) for r in ladder(parse("a b c d e f"), tuple("abcdef"), ())] == [
        ("words", 3), ("words_loose", 2)]
    assert [(r.name, r.need) for r in ladder(parse("stop OR start"), ("stop", "start"), ())] == [
        ("words", 1)]
    assert ladder(parse("the and"), (), ()) == ()
