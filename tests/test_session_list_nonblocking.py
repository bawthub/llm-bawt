"""Session SQL and cold manager construction must not block the shared API loop."""

import asyncio
from contextvars import ContextVar
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.mcp_server import storage_sessions
from llm_bawt.mcp_server.storage_sessions import StorageSessionsMixin
from llm_bawt.bot_access import BotAccessError


@pytest.mark.parametrize("blocked_stage", ["manager", "query"])
def test_session_list_allows_other_loop_work_and_keeps_context(monkeypatch, blocked_stage):
    monkeypatch.setattr(storage_sessions, "current_bot_access", lambda: None)
    marker = ContextVar("session_list_test_context", default=None)
    result = [{"id": "session_fixture"}]
    release = Event()

    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = get_ident()
        entered = asyncio.Event()
        observed = []

        def block(stage):
            observed.append(stage)
            # An immediate, deterministic regression failure before any wait:
            # the old implementation executes these functions on loop_thread.
            assert get_ident() != loop_thread
            assert marker.get() == "request_context"
            if stage == blocked_stage:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10), "test did not release worker"

        def query(**kwargs):
            block("query")
            assert kwargs == {
                "bot_id": "public_fixture", "since": 123.0, "status": "archived",
                "limit": 200, "user_id": "owner_fixture",
                "include_deleted": True, "query": "needle",
            }
            return result

        class Storage(StorageSessionsMixin):
            def get_short_term_manager(self, bot_id):
                block("manager")
                assert bot_id == "public_fixture"
                return SimpleNamespace(list_sessions=query)

        token = marker.set("request_context")
        pending = asyncio.create_task(Storage().list_sessions(
            "public_fixture", since=123.0, status="archived", limit=200,
            user_id="owner_fixture", include_deleted=True, query="needle",
        ))
        try:
            # If the worker fails before announcing entry, propagate that
            # exception immediately instead of waiting for a timeout.
            waiter = asyncio.create_task(entered.wait())
            done, _ = await asyncio.wait(
                {pending, waiter}, timeout=10, return_when=asyncio.FIRST_COMPLETED,
            )
            if pending in done:
                await pending
                pytest.fail("session read finished without entering blocked stage")
            assert waiter in done
            assert not pending.done()
            unrelated = loop.create_future()
            loop.call_soon(unrelated.set_result, "healthy")
            assert await unrelated == "healthy"
        finally:
            release.set()
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            marker.reset(token)
        assert await pending is result
        assert observed == ["manager", "query"]

    asyncio.run(run())


def test_session_list_propagates_worker_exception(monkeypatch):
    monkeypatch.setattr(storage_sessions, "current_bot_access", lambda: None)
    storage = StorageSessionsMixin()
    storage.get_short_term_manager = Mock(side_effect=RuntimeError("manager failed"))
    with pytest.raises(RuntimeError, match="manager failed"):
        asyncio.run(storage.list_sessions(bot_id="public_fixture"))


def test_session_list_empty_access_scope_does_not_construct_manager(monkeypatch):
    access = SimpleNamespace(
        caller=None,
        aggregate_scope=lambda bot: SimpleNamespace(partitions=[]),
    )
    monkeypatch.setattr(storage_sessions, "current_bot_access", lambda: access)
    monkeypatch.setattr(storage_sessions, "is_service_self", lambda access, bot: False)
    storage = StorageSessionsMixin()
    storage.get_short_term_manager = Mock(side_effect=AssertionError("must not run"))
    assert asyncio.run(storage.list_sessions(bot_id="public_fixture")) == []
    storage.get_short_term_manager.assert_not_called()


def test_session_list_preserves_scoped_owner_and_partitions_in_worker(monkeypatch):
    access = SimpleNamespace(
        caller=SimpleNamespace(user_id="owner_fixture"),
        aggregate_scope=lambda bot: SimpleNamespace(
            partitions=[SimpleNamespace(bot_id="public_fixture", authority="caller")],
            user_id="owner_fixture",
        ),
    )
    monkeypatch.setattr(storage_sessions, "current_bot_access", lambda: access)
    monkeypatch.setattr(storage_sessions, "is_service_self", lambda access, bot: False)
    manager = SimpleNamespace(list_sessions=Mock(return_value=[{"id": "owned_session"}]))
    storage = StorageSessionsMixin()
    storage.get_short_term_manager = Mock(return_value=manager)
    assert asyncio.run(storage.list_sessions(bot_id="public_fixture")) == [{"id": "owned_session"}]
    storage.get_short_term_manager.assert_called_once_with("public_fixture")
    manager.list_sessions.assert_called_once_with(
        bot_id="public_fixture", since=None, status=None, limit=50,
        user_id="owner_fixture", include_deleted=False, query=None,
        allowed_bot_ids=frozenset({"public_fixture"}),
    )

    storage.get_short_term_manager.reset_mock()
    with pytest.raises(BotAccessError, match="access_denied"):
        asyncio.run(storage.list_sessions(bot_id="public_fixture", user_id="other_fixture"))
    storage.get_short_term_manager.assert_not_called()
