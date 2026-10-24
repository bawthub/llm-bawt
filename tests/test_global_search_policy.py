"""Aggregate cross-bot search always runs under a BotAccess and fails closed.

TASK-1323 (F2): the old exclusion-list path returned an empty exclusion set
when the bot policy couldn't be read (protected bots became searchable) and
turned search failures into ``[]``. Now an unbound caller gets a fresh
no-caller policy read, and every failure raises ``BotAccessError``.
"""
from __future__ import annotations

import asyncio

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.bot_access.context import bind_bot_access
from llm_bawt.mcp_server import message_search, storage_search
from llm_bawt.mcp_server.storage_search import StorageSearchMixin

ROWS = [{"slug": "public-fixture", "include_in_global_search": True},
        {"slug": "private_fixture", "include_in_global_search": False}]


class _Resolver:
    def __init__(self, _engine):
        pass

    async def resolve(self, caller):
        _Resolver.calls.append(caller)
        return BotPolicyResolver._compile(ROWS, caller)


class _FailingResolver:
    def __init__(self, _engine):
        pass

    async def resolve(self, caller):
        raise BotAccessError("policy_unavailable")


class _Searcher:
    def __init__(self, _engine, _hydrate):
        pass

    def search_fts(self, query, **kwargs):
        _Searcher.calls.append(kwargs)
        if _Searcher.fail:
            raise RuntimeError("backend details")
        return [{"id": p, "source": p} for p in sorted(kwargs["partitions"])]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Engine:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail = list(rows), fail

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                if engine.fail:
                    raise RuntimeError("connection refused")
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, _statement, params):
                engine.params = params
                return _Rows(engine.rows)

        return _Conn()


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    _Resolver.calls, _Searcher.calls, _Searcher.fail = [], [], False
    monkeypatch.setattr(storage_search, "BotPolicyResolver", _Resolver)
    monkeypatch.setattr(message_search, "CrossBotMessageSearcher", _Searcher)


def make_storage(engine=None):
    storage = StorageSearchMixin()
    storage._require_engine = lambda: engine if engine is not None else _Engine()
    storage.hydrate_message_authors = lambda rows, bot_id: rows
    return storage


def run(coro):
    return asyncio.run(coro)


def test_unbound_message_search_uses_a_fresh_no_caller_policy():
    rows = run(make_storage().search_all_messages("TASK-1281"))
    assert _Resolver.calls == [None]
    assert _Searcher.calls[0]["partitions"] == frozenset({"publicfixture"})
    # Sources come back as canonical slugs, not partition names.
    assert [row["source"] for row in rows] == ["public-fixture"]


def test_bound_access_is_used_and_policy_is_not_reread(monkeypatch):
    monkeypatch.setattr(storage_search, "BotPolicyResolver", _FailingResolver)
    access = BotPolicyResolver._compile(ROWS, TrustedCaller("private_fixture", "user_a", "service"))
    with bind_bot_access(access):
        rows = run(make_storage().search_all_messages("query"))
    assert _Searcher.calls[0]["partitions"] == frozenset({"publicfixture", "private_fixture"})
    assert {row["source"] for row in rows} == {"public-fixture", "private_fixture"}


@pytest.mark.parametrize("call", [
    lambda s: s.search_all_messages("query"),
    lambda s: s.search_all_messages("query", bot_id="public-fixture"),
    lambda s: s.list_memory_sources(),
    lambda s: s.search_all_memories("query"),
])
def test_policy_read_failure_raises_instead_of_searching_everything(monkeypatch, call):
    monkeypatch.setattr(storage_search, "BotPolicyResolver", _FailingResolver)
    with pytest.raises(BotAccessError, match="^policy_unavailable$"):
        run(call(make_storage()))
    assert _Searcher.calls == []


@pytest.mark.parametrize("bot_id", ["private_fixture", "privatefixture", "ghost"])
def test_protected_or_unknown_bot_filter_is_denied(bot_id):
    with pytest.raises(BotAccessError, match="^access_denied$"):
        run(make_storage().search_all_messages("query", bot_id=bot_id))
    assert _Searcher.calls == []


def test_bot_filter_narrows_to_one_partition():
    run(make_storage().search_all_messages("query", bot_id="Public-Fixture"))
    assert _Searcher.calls[0]["partitions"] == frozenset({"publicfixture"})


def test_search_failure_is_unavailable_not_an_empty_match():
    _Searcher.fail = True
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        run(make_storage().search_all_messages("query"))


def test_missing_engine_is_unavailable():
    storage = make_storage()

    def no_engine():
        raise RuntimeError("storage not initialized")

    storage._require_engine = no_engine
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        run(storage.search_all_messages("query"))


def test_unbound_memory_sources_are_the_public_scope():
    engine = _Engine(rows=[("publicfixture", 3)])
    assert run(make_storage(engine).list_memory_sources()) == [
        {"source": "public-fixture", "memory_count": 3},
    ]
    assert engine.params == {"partitions": ["publicfixture"]}


def test_memory_source_count_failure_is_unavailable():
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        run(make_storage(_Engine(fail=True)).list_memory_sources())


def test_unbound_memory_search_requires_a_user_before_embedding(monkeypatch):
    from llm_bawt.memory import embeddings
    monkeypatch.setattr(embeddings, "generate_embedding",
                        lambda *_a, **_k: pytest.fail("embedding reached"))
    with pytest.raises(BotAccessError, match="^caller_unavailable$"):
        run(make_storage().search_all_memories("query"))
