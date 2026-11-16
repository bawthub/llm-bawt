"""Exact-entry transport receipts use real Lua, never command execution."""
import asyncio
import hashlib
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from fakeredis.commands_mixins.scripting_mixin import ScriptingCommandsMixin
from redis.exceptions import ResponseError

from agent_bridge.command_ack import CommandAckOwner
from agent_bridge.command_ack_scripts import MINT_ACK

STREAM, GROUP, BACKEND = "test:commands", "claude-code-bridge", "claude-code"


@pytest.fixture
async def transport(monkeypatch):
    # fakeredis executes the actual Lua but omits Redis's sha1hex builtin.
    # Supply only that primitive; source parsing/binding/ACK logic stays Lua.
    get_runtime = ScriptingCommandsMixin._get_server_runtime

    def runtime_with_sha1(self, server):
        runtime = get_runtime(self, server)
        runtime.globals()[b"redis"][b"sha1hex"] = lambda value: hashlib.sha1(value).hexdigest().encode()
        return runtime

    monkeypatch.setattr(ScriptingCommandsMixin, "_get_server_runtime", runtime_with_sha1)
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    try:
        yield client
    finally:
        await client.aclose()


def owner(client, **kwargs):
    return CommandAckOwner(
        client, stream=STREAM, group=GROUP, backend=BACKEND, boot_id="original", **kwargs,
    )


async def pending(client, **extra):
    fields = {
        "action": "chat.send", "backend": BACKEND, "request_id": "request",
        "session_key": "opaque/session", "message": "input café 🎈", **extra,
    }
    entry_id = await client.xadd(STREAM, fields)
    await client.xreadgroup(GROUP, "worker-0", {STREAM: ">"}, count=1)
    return entry_id, fields


async def mint(handler, context):
    assert await handler._eval(MINT_ACK, context) == 1


@pytest.mark.anyio
async def test_original_mint_settle_and_late_duplicate_are_idempotent(transport):
    entry_id, fields = await pending(transport, house_epoch="old-epoch")
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    await handler.acknowledge(context)
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 0
    assert await transport.get(context.key) is None
    assert await handler.settle(context)
    await handler.acknowledge(context)


@pytest.mark.anyio
async def test_original_handler_acks_entry_trimmed_during_run_without_receipt(transport):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    await transport.xdel(STREAM, entry_id)
    assert not await handler.settle(context)
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 1
    await handler.acknowledge(context)
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 0
    assert await transport.get(context.key) is None


@pytest.mark.anyio
async def test_restart_and_concurrent_settlers_use_original_boot_and_epoch(transport):
    entry_id, fields = await pending(transport, house_epoch="old-epoch")
    context = owner(transport).capture(fields, entry_id)
    await mint(owner(transport), context)
    assert await transport.ttl(context.key) == -1
    replacement = replace(context, evidence=context.evidence.replace('"original"', '"replacement"'))
    assert not await owner(transport).settle(replacement)
    recovery = CommandAckOwner(
        transport, stream=STREAM, group=GROUP, backend=BACKEND, boot_id="replacement",
    )
    assert all(await asyncio.gather(recovery.settle(context), recovery.settle(context)))
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["cancel", "lost_response"])
async def test_committed_mint_survives_cancelled_or_lost_response(transport, failure):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    evaluate = handler._eval

    async def fail_after_commit(script, bound):
        result = await evaluate(script, bound)
        assert result == 1
        if failure == "cancel":
            raise asyncio.CancelledError
        raise ConnectionError("response lost")

    handler._eval = fail_after_commit
    with pytest.raises((asyncio.CancelledError, ConnectionError)):
        await handler.acknowledge(context)
    counts = await owner(transport).scan()
    assert counts == {"examined": 1, "settled": 1, "retained": 0}
    assert await transport.get(context.key) is None


@pytest.mark.anyio
async def test_unknown_heads_do_not_starve_tail_or_behind_cursor_mint(transport):
    first_id, first_fields = await pending(transport)
    last_id, last_fields = await pending(transport, request_id="tail")
    recovery = owner(transport, page_size=1)
    assert (await recovery.scan())["retained"] == 1
    # A valid receipt appears behind the scan cursor.
    await mint(recovery, recovery.capture(first_fields, first_id))
    await mint(recovery, recovery.capture(last_fields, last_id))
    # Fresh traffic cannot keep extending this cycle and starve the old mint.
    await pending(transport, request_id="newer")
    assert (await recovery.scan())["settled"] == 1
    assert (await recovery.scan())["examined"] == 0
    assert (await recovery.scan())["settled"] == 1


@pytest.mark.anyio
async def test_unknown_pending_including_live_runs_is_retained_without_dispatch(transport):
    for _ in range(3):
        await pending(transport)
    recovery = owner(transport)
    assert await recovery.scan() == {"examined": 3, "settled": 0, "retained": 3}
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 3
    assert await transport.keys("agent:command_ack:*") == []


@pytest.mark.anyio
@pytest.mark.parametrize("field", [
    "action", "backend", "request_id", "session_key", "house_epoch", "message",
])
async def test_mint_checks_all_original_fields_before_committing(transport, field):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    changed = {**fields, field: "different"}
    if field in {"action", "backend"}:
        with pytest.raises(ValueError):
            handler.capture(changed, entry_id)
    else:
        context = handler.capture(changed, entry_id)
        with pytest.raises(RuntimeError, match="conflicted"):
            await handler.acknowledge(context)
        assert await transport.get(context.key) is None
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 1


@pytest.mark.anyio
async def test_different_owner_cannot_overwrite_original_receipt(transport):
    entry_id, fields = await pending(transport)
    context = owner(transport).capture(fields, entry_id)
    await mint(owner(transport), context)
    other = CommandAckOwner(
        transport, stream=STREAM, group=GROUP, backend=BACKEND, boot_id="other",
    )
    with pytest.raises(RuntimeError, match="conflicted"):
        await other.acknowledge(other.capture(fields, entry_id))
    assert await transport.get(context.key) == context.evidence


@pytest.mark.anyio
async def test_reused_request_id_or_missing_receipt_never_settles_another_entry(transport):
    first_id, fields = await pending(transport)
    handler = owner(transport)
    await mint(handler, handler.capture(fields, first_id))
    second_id, _ = await pending(transport)  # Same request ID, different publication.
    assert not await handler.settle(handler.capture(fields, second_id))
    assert (await handler.scan())["settled"] == 1
    remaining = await transport.xpending_range(STREAM, GROUP, "-", "+", 10)
    assert [row["message_id"] for row in remaining] == [second_id]


@pytest.mark.anyio
@pytest.mark.parametrize("loss", ["delete", "trim", "group"])
async def test_destructive_transport_loss_is_retained_not_recreated(transport, loss):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    await mint(handler, context)
    if loss == "group":
        await transport.xgroup_destroy(STREAM, GROUP)
        with pytest.raises(ResponseError, match="NOGROUP"):
            await handler.scan()
        with pytest.raises(ResponseError, match="NOGROUP"):
            await handler.settle(context)
        assert await transport.xinfo_groups(STREAM) == []
    else:
        if loss == "delete":
            await transport.xdel(STREAM, entry_id)
        else:
            await transport.xtrim(STREAM, maxlen=0)
        assert (await handler.scan())["retained"] == 1
        assert not await handler.settle(context)
    assert await transport.get(context.key) == context.evidence
    assert await transport.ttl(context.key) == -1


@pytest.mark.anyio
@pytest.mark.parametrize("field,value", [
    ("version", True), ("action", "chat.steer"), ("backend", "other"),
    ("request_id", ""), ("stream", "other"), ("group", "other"),
    ("entry_id", "99-0"), ("boot_id", ""), ("house_epoch", 2),
    ("source_digest", "0" * 40), ("reason", "terminal_lookup"),
])
async def test_invalid_receipts_fail_closed(transport, field, value):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    data = {**json.loads(context.evidence), field: value}
    evidence = json.dumps(data)
    await transport.set(context.key, evidence)
    assert (await handler.scan())["retained"] == 1
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 1
    assert await transport.get(context.key) == evidence


@pytest.mark.anyio
async def test_error_and_cancellation_before_mint_do_not_ack_unknown(transport):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    handler._eval = AsyncMock(side_effect=ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        await handler.acknowledge(handler.capture(fields, entry_id))
    assert (await owner(transport).scan())["retained"] == 1


@pytest.mark.anyio
async def test_worker_paces_success_and_errors_and_stops(monkeypatch, transport):
    recovery = owner(transport)
    recovery.scan = AsyncMock(side_effect=[{"examined": 0}, ConnectionError("offline"), {"examined": 0}])
    sleeping = asyncio.Event()
    delays = []

    async def pause(delay):
        delays.append(delay)
        if len(delays) == 3:
            sleeping.set()
            await asyncio.Event().wait()

    monkeypatch.setattr("agent_bridge.command_ack.asyncio.sleep", pause)
    recovery.start()
    await sleeping.wait()
    await recovery.stop()
    assert delays == [5, 10, 5]
    assert recovery.task.cancelled()


@pytest.mark.anyio
async def test_blocked_redis_operation_is_bounded_and_cancellable(transport):
    recovery = owner(transport, timeout=0)
    recovery.redis = type("Blocked", (), {"xpending": AsyncMock()})()

    async def blocked(*args):
        await asyncio.Event().wait()

    recovery.redis.xpending.side_effect = blocked
    # Cancellation at the timeout boundary proves no infinite socket read.
    with pytest.raises(TimeoutError):
        await recovery.scan()


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_actual_original_final_ack_preserves_admission_and_outer_terminal(
    monkeypatch, transport, outcome,
):
    from agent_bridge.session_queue import RequestPhase
    from tests.test_claude_code_send_boundaries import MODEL, SESSION, _SDKStream, _SendHarness

    entry_id, fields = await pending(
        transport, session_key=SESSION, model=MODEL, thread_resume_id="sdk-session",
        thread_session_id="thread", trigger_message_id="trigger",
    )
    harness, sdk = _SendHarness(), _SDKStream(result=True)
    handler = owner(transport)
    harness._command_ack_owner = handler
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
    capture, acknowledge = handler.capture, handler.acknowledge
    calls = []

    def capture_after_admission(raw, msg_id):
        assert harness._session_queue.request_phase(SESSION, "request") is RequestPhase.PREFLIGHT
        return capture(raw, msg_id)

    async def original_disposition(context):
        assert not harness._publisher.publish_run_done.called
        calls.append(context)
        await acknowledge(context)

    handler.capture, handler.acknowledge = capture_after_admission, original_disposition
    if outcome == "error":
        sdk.connect = AsyncMock(side_effect=ValueError("fixture SDK failure"))
    send = asyncio.create_task(harness._handle_send(fields, entry_id, transport))
    if outcome != "error":
        await sdk.waiting.wait()
        if outcome == "cancel":
            send.cancel()
        else:
            sdk.release_stream.set()
    result, = await asyncio.gather(send, return_exceptions=True)
    if outcome == "cancel":
        assert isinstance(result, asyncio.CancelledError), repr(result)
    else:
        assert result is None, repr(result)
    assert len(calls) == 1
    assert "input café" not in calls[0].evidence
    harness._publisher.publish_run_done.assert_called_once()
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 0
    assert harness._session_queue.request_phase(SESSION, "request") is None


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["refused", "preflight", "duplicate"])
async def test_other_original_disposition_paths_do_not_mint(monkeypatch, transport, path):
    from tests.test_claude_code_send_boundaries import MODEL, SESSION, _SendHarness

    entry_id, fields = await pending(transport, session_key=SESSION, model=MODEL)
    harness = _SendHarness()
    handler = owner(transport)
    harness._command_ack_owner = handler
    handler.acknowledge = AsyncMock()
    if path == "refused":
        harness._session_queue.tombstone(SESSION, "request")
    elif path == "duplicate":
        harness._session_queue.accept(SESSION, "request")
    else:
        harness._preprocess_new_command = AsyncMock(return_value=None)
    # These paths keep their existing ACK/terminal handling; use its small
    # transport fake, since only the selected normal-finally path uses receipts.
    await harness._handle_send(fields, entry_id, harness.redis)
    handler.acknowledge.assert_not_awaited()
    assert await transport.keys("agent:command_ack:*") == []


@pytest.mark.anyio
async def test_lost_settlement_response_retries_only_ack(transport):
    entry_id, fields = await pending(transport)
    handler = owner(transport)
    context = handler.capture(fields, entry_id)
    evaluate = handler._eval

    async def lose_settlement_response(script, bound):
        result = await evaluate(script, bound)
        if script != MINT_ACK:
            raise ConnectionError("settlement response lost")
        return result

    handler._eval = lose_settlement_response
    with pytest.raises(ConnectionError):
        await handler.acknowledge(context)
    assert (await transport.xpending(STREAM, GROUP))["pending"] == 0
    assert await transport.get(context.key) is None
    assert await owner(transport).settle(context)


@pytest.mark.anyio
async def test_stop_on_actual_post_mint_boundary_repairs_without_sdk_replay(monkeypatch, transport):
    from tests.test_claude_code_send_boundaries import MODEL, SESSION, _SDKStream, _SendHarness

    entry_id, fields = await pending(
        transport, session_key=SESSION, model=MODEL, thread_resume_id="sdk-session",
        thread_session_id="thread",
    )
    harness, sdk, boundary = _SendHarness(), _SDKStream(result=True), asyncio.Event()
    handler = owner(transport)
    harness._command_ack_owner = handler
    sdk.connect = AsyncMock()
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)

    async def interrupted_settle(context):
        assert await transport.get(context.key) == context.evidence
        boundary.set()
        await asyncio.Event().wait()

    handler.settle = interrupted_settle
    sdk.release_stream.set()
    send = asyncio.create_task(harness._handle_send(fields, entry_id, transport))
    await boundary.wait()
    send.cancel()
    await asyncio.gather(send, return_exceptions=True)
    harness._publisher.publish_run_done.assert_called_once()
    assert (await owner(transport).scan())["settled"] == 1
    sdk.connect.assert_awaited_once()
    harness._publisher.publish_run_done.assert_called_once()  # Recovery never republishes.
