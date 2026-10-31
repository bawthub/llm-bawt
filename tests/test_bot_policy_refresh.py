"""Bounded snapshot refresh and real MCP authorization stages; fixture identities."""
from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeout

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.refresh import PolicyRefresher, failure_reason
from llm_bawt.bot_access.snapshot import SnapshotUnavailable, elapsed_clock
from llm_bawt.bot_access.service import ServicePrincipal
from llm_bawt.bot_policy_events import POLICY_CHANNEL
from llm_bawt.mcp_server.bot_access import ToolBotAccess, invocation_context
from llm_bawt.mcp_server.bot_access_identity import _service_capability
from tests.test_bot_access_mcp import identity, signer  # noqa: F401 - fixture
from llm_bawt.task_turn_context import mint_task_turn_context

pytestmark = pytest.mark.usefixtures("signer")

ROWS = [{"slug": "fixture_owner", "include_in_global_search": False},
        {"slug": "fixture_peer", "include_in_global_search": True}]


def loaded(monkeypatch):
    engine = Mock()
    owner = PolicyRefresher(engine)
    monkeypatch.setattr("llm_bawt.bot_access.refresh.read_snapshot_rows", lambda _: ROWS)
    assert owner.refresh()
    return owner, BotPolicyResolver(engine)


def token(session, user="user_a"):
    return mint_task_turn_context(session_id=session, turn_id="turn-" + uuid.uuid4().hex,
        trigger_message_id=str(uuid.uuid4()), bot_id="fixture_owner", user_id=user)


def test_missing_engine_is_sanitized_cold_policy_not_constructor_error():
    with pytest.raises(BotAccessError, match="^policy_unavailable$"):
        BotPolicyResolver(None).resolve_sync()


@pytest.mark.asyncio
async def test_cold_resolver_never_lazy_reads_or_starts_worker():
    engine = Mock()
    resolver = BotPolicyResolver(engine)
    for _ in range(8):
        with pytest.raises(BotAccessError, match="policy_unavailable"):
            await resolver.resolve()
    assert engine.mock_calls == []


@pytest.mark.asyncio
async def test_exact_verified_conversation_zero_owner_reads_at_both_stages(monkeypatch):
    owner, resolver = loaded(monkeypatch)
    guard = ToolBotAccess(resolver)
    lookup = Mock(side_effect=AssertionError("unexpected ownership SELECT"))
    monkeypatch.setattr(guard, "_session_owner", lookup)
    received = []

    async def write(bot_id, session_id=None, user_id=None):
        received.append((bot_id, session_id, user_id))

    guarded = guard.register("messages_add", write)
    session = str(uuid.uuid4())
    args = {"bot_id": "fixture_owner", "session_id": session}
    with identity(token(session)):
        await guard.preflight("messages_add", args)
        await guarded(**args)
    assert received == [("fixture_owner", session, "user_a")]
    lookup.assert_not_called()
    assert owner.engine.mock_calls == []


@pytest.mark.asyncio
async def test_original_runtime_service_caller_without_explicit_session_uses_memory(monkeypatch):
    owner, resolver = loaded(monkeypatch)
    guard = ToolBotAccess(resolver)
    lookup = Mock(side_effect=AssertionError("unexpected ownership SELECT"))
    monkeypatch.setattr(guard, "_session_owner", lookup)
    writes = []

    async def write(bot_id, session_id=None, user_id=None):
        writes.append((bot_id, session_id, user_id))

    guarded = guard.register("messages_add", write)
    args = {"bot_id": "fixture_owner", "user_id": "user_a"}
    envelope = ServicePrincipal("fixture_owner", "user_a").mint("messages_add", args)
    binding = _service_capability.set(envelope)
    try:
        with invocation_context("messages_add", args):
            await asyncio.gather(*(guard.preflight("messages_add", args) for _ in range(64)))
            await asyncio.gather(*(guarded(**args) for _ in range(64)))
    finally:
        _service_capability.reset(binding)
    assert writes == [("fixture_owner", None, "user_a")] * 64
    lookup.assert_not_called()
    assert owner.engine.mock_calls == []


@pytest.mark.asyncio
async def test_same_bot_other_user_session_still_denied_not_claimed_zero_read(monkeypatch):
    _, resolver = loaded(monkeypatch)
    guard = ToolBotAccess(resolver)
    lookup = Mock(return_value=("fixture_owner", "user_b"))
    monkeypatch.setattr(guard, "_session_owner", lookup)

    async def write(bot_id, session_id=None, user_id=None):
        pytest.fail("spoof reached writer")

    guarded = guard.register("messages_add", write)
    own, other = str(uuid.uuid4()), str(uuid.uuid4())
    args = {"bot_id": "fixture_owner", "session_id": other}
    with identity(token(own)):
        with pytest.raises(BotAccessError, match="access_denied"):
            await guard.preflight("messages_add", args)
        with pytest.raises(BotAccessError, match="access_denied"):
            await guarded(**args)
    assert lookup.call_count == 2  # unchanged/unapproved fallback, disclosed


@pytest.mark.asyncio
async def test_expiry_after_async_ownership_work_refuses_admission(monkeypatch):
    from llm_bawt.mcp_server import bot_access as module

    owner, resolver = loaded(monkeypatch)
    guard = ToolBotAccess(resolver)

    class Worker:
        async def run(self, *_):
            owner.snapshot.invalidate("snapshot_revoked")
            return "fixture_peer", "user_a"

    monkeypatch.setattr(module, "_METADATA_WORKER", Worker())

    async def write(bot_id, session_id=None):
        pytest.fail("invalidated authority reached writer")

    guarded = guard.register("messages_add", write)
    with pytest.raises(BotAccessError, match="policy_unavailable"):
        await guarded(bot_id="fixture_peer", session_id="fixture-session")


def test_failed_refresh_sanitizes_and_recovers_without_request_reads(monkeypatch, caplog):
    owner, resolver = loaded(monkeypatch)

    def fail(_):
        raise RuntimeError("sensitive database detail")

    monkeypatch.setattr("llm_bawt.bot_access.refresh.read_snapshot_rows", fail)
    assert not owner.refresh()
    assert owner.snapshot.diagnostics()["state"] == "snapshot_database"
    with pytest.raises(BotAccessError, match="policy_unavailable"):
        resolver.resolve_sync()
    assert "sensitive database detail" not in caplog.text
    monkeypatch.setattr("llm_bawt.bot_access.refresh.read_snapshot_rows", lambda _: ROWS)
    assert owner.refresh()
    assert resolver.resolve_sync().require_target("fixture_peer") == "fixture_peer"


@pytest.mark.parametrize("elapsed", [5, 6, 600])
def test_read_latency_and_process_suspension_count_toward_five_second_bound(monkeypatch, elapsed):
    now = [0.0]
    owner = PolicyRefresher(Mock(), clock=lambda: now[0])
    owner.snapshot._clock = lambda: now[0]

    def read(_):
        now[0] += elapsed
        return ROWS

    monkeypatch.setattr("llm_bawt.bot_access.refresh.read_snapshot_rows", read)
    assert not owner.refresh()
    with pytest.raises(SnapshotUnavailable):
        owner.snapshot.get()


def test_wall_clock_changes_cannot_extend_elapsed_deadline(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: -10**10)
    before = elapsed_clock()
    monkeypatch.setattr(time, "time", lambda: 10**10)
    after = elapsed_clock()
    assert after >= before
    assert time.CLOCK_BOOTTIME != time.CLOCK_REALTIME


def test_notification_racing_select_discards_candidate(monkeypatch):
    owner, resolver = loaded(monkeypatch)
    raw = SimpleNamespace(poll=lambda: None, notifies=[
        SimpleNamespace(channel=POLICY_CHANNEL, payload="fixture_schema")])
    assert not owner.refresh(raw, "fixture_schema")
    with pytest.raises(BotAccessError, match="policy_unavailable"):
        resolver.resolve_sync()
    assert owner.refresh(raw, "fixture_schema")


@pytest.mark.parametrize("exc,expected", [
    (PoolTimeout("sensitive"), "snapshot_busy"),
    (TimeoutError("sensitive"), "snapshot_timeout"),
    (ValueError("sensitive"), "snapshot_validation"),
    (RuntimeError("sensitive"), "snapshot_database"),
])
def test_internal_failure_reasons_are_distinct_and_sanitized(exc, expected):
    assert failure_reason(exc) == expected


@pytest.mark.asyncio
async def test_approved_retry_reauthorizes_against_real_memory_snapshot(monkeypatch):
    from agent_bridge.mcp_call_context import canonical_invocation_hash
    from llm_bawt.mcp_server.approval_interceptor import ApprovedCallerContext
    from llm_bawt.mcp_server.catalog import CatalogFastMCP

    owner, resolver = loaded(monkeypatch)
    server = CatalogFastMCP("fixture", bot_policy_resolver=resolver)
    writes = []

    async def write(bot_id):
        writes.append(bot_id)
        return {"ok": True}

    server.add_tool(write, name="messages_add")
    server.mark_tools_registered()
    args = {"bot_id": "fixture_peer"}
    context = ApprovedCallerContext(bot_id="fixture_owner", user_id="user_a")
    options = {"expected_invocation_hash": canonical_invocation_hash("messages_add", args),
               "caller_context": context}
    await server.call_approved_tool("messages_add", args, **options)
    rows = [dict(row, include_in_global_search=False) for row in ROWS]
    owner.snapshot.invalidate("snapshot_revoked")
    owner.snapshot.install(BotPolicyResolver._compile(rows, None), 2,
                           elapsed_clock() + 5, owner.snapshot.generation())
    with pytest.raises(Exception, match="access_denied"):
        await server.call_approved_tool("messages_add", args, **options)
    assert writes == ["fixture_peer"]
    assert owner.engine.mock_calls == []


def test_event_refresh_occurs_without_waiting_for_reconciliation(monkeypatch):
    from llm_bawt.bot_access import refresh as module

    engine = MagicMock()
    listener = engine.connect.return_value.__enter__.return_value
    listener.execution_options.return_value = listener
    # Context manager is entered after execution_options().
    engine.connect.return_value.execution_options.return_value.__enter__.return_value = listener
    listener.execute.return_value.scalar_one.return_value = "fixture_schema"
    raw = listener.connection.driver_connection
    owner = PolicyRefresher(engine, clock=lambda: 0.0)
    owner.snapshot._clock = lambda: 0.0
    reads = []
    monkeypatch.setattr(module, "read_snapshot_rows", lambda _: reads.append(True) or ROWS)
    drains = iter([False, True, False])  # initial read; event; event-triggered read
    monkeypatch.setattr(owner, "_drain", lambda *_: next(drains))
    wakes = [0]

    def wake(*_):
        wakes[0] += 1
        if wakes[0] == 1:
            return [raw], [], []
        owner._stop.set()
        return [], [], []

    monkeypatch.setattr(module.select, "select", wake)
    owner._run()
    assert len(reads) == 2  # clock stayed zero; 2s reconciliation never fired
