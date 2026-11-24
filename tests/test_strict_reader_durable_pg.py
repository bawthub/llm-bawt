"""TASK-2590: a strict-reader turn through the real durable agent path.

Synthetic identities on private PostgreSQL. The strict bot is an agent on the
Claude Code backend, so the HTTP stream admits a durable ordinary start and
returns a queued receipt. The real dispatcher then drains the target, claims
the row, validates the claim and re-enters the real streaming path, which
builds the strict instance with the real agent client and publishes the turn
command through the real command publisher. Only the Redis write client under
that publisher and the bridge's returned run events are faked. The shared bot
registry is cold and every shared read is fatal for the whole flow.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.command_publisher import CommandPublisherMixin
from agent_bridge.events import AgentEvent, AgentEventKind
from llm_bawt.service import strict_reader_execution as strict
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_message_native_upsert import capture_sql
from tests.test_message_native_upsert_pg import http_runtime, real_storage  # noqa: F401
from tests.test_native_selected_owner_pg import native_runtime  # noqa: F401
from tests.test_strict_reader_stream_pg import (  # noqa: F401
    APPROVED, FOREIGN, _body, _request, _stream, strict_runtime,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("private_app_pg", "persisted_master_key"),
]

_REAL_CLIENT = strict.StrictReaderLLMBawt._initialize_client


class _RecordingRedis:
    """The publisher's Redis write client: records each published command."""

    #: Live house leases, ``lease_key -> epoch``; set by a test.
    leases: dict = {}
    #: The command stream each published command went to.
    streams: list = []

    def __init__(self, published):
        self.published = published

    async def get(self, key):
        return self.leases.get(key)

    async def set(self, *_a, **_k):
        return True

    async def eval(self, script, numkeys, *args):
        keys, argv = args[:numkeys], args[numkeys:]
        self.streams.append(keys[1])
        self.published.append(argv)
        return 1, "1-0"


class _FakeBridge(CommandPublisherMixin):
    """Real ``send_command``; Redis and the bridge's run events are faked."""

    published: list = []

    def __init__(self, _url):
        self._pub_redis = _RecordingRedis(self.published)

    async def connect(self):
        return None

    async def close(self):
        return None

    async def subscribe_run(self, request_id, **_kwargs):
        for kind, text in ((AgentEventKind.ASSISTANT_DELTA, "strict answer"),
                           (AgentEventKind.ASSISTANT_DONE, "strict answer")):
            yield AgentEvent(event_id=kind.value, session_key="s", run_id=request_id,
                             kind=kind, origin="assistant", text=text)


def _command_fields(argv) -> dict:
    """Decode the published ``chat.send`` field/value pairs from ARGV."""
    for item in argv:
        if isinstance(item, str) and item.startswith("{") and '"chat.send"' in item:
            import json
            return json.loads(item)
    flat = [a for a in argv if isinstance(a, (str, bytes))]
    pairs = dict(zip(flat[::2], flat[1::2]))
    if pairs.get("action") == "chat.send":
        return pairs
    raise AssertionError(f"no chat.send command in {argv!r}")


@pytest.fixture
def durable(strict_runtime, rig, monkeypatch):  # noqa: F811
    from llm_bawt import model_catalog
    from llm_bawt.agent_backends import agent_bridge as bridge_module
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
    from llm_bawt.service.turn_logs import TurnLogStore

    rt = strict_runtime.rt
    # The real agent client, not the native recording client.
    built = strict_runtime.built

    def real_client(self):
        built.append(self)
        return _REAL_CLIENT(self)

    monkeypatch.setattr(strict.StrictReaderLLMBawt, "_initialize_client", real_client)
    # Task attribution is a remote HTTP lookup unrelated to this boundary.
    monkeypatch.setattr("llm_bawt.service.turn_task_attribution.stamp_turn_task",
                        AsyncMock(return_value=None))
    real_resolve = model_catalog.resolve_model_config

    def resolve_model(config, alias, *a, **k):
        if alias == "fixture-native":
            return {"type": "claude-code", "backend": "claude-code", "model_id": "fixture-native"}
        return real_resolve(config, alias, *a, **k)

    monkeypatch.setattr(model_catalog, "resolve_model_config", resolve_model)
    published: list = []
    monkeypatch.setattr(_FakeBridge, "published", published)
    from agent_bridge.houses import lease_key
    from tests.test_strict_reader_stream_pg import HOUSE
    monkeypatch.setattr(_RecordingRedis, "leases", {lease_key(HOUSE, "claude-code"): "7"})
    monkeypatch.setattr(_RecordingRedis, "streams", [])
    root = SimpleNamespace(_redis_url="redis://fixture.invalid/0")
    monkeypatch.setattr(bridge_module, "get_agent_subscriber", lambda: root)
    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", _FakeBridge)

    turns = TurnLogStore(rig.config)
    deliveries = InterBotDeliveryStore(rig.config)
    rt.service._turn_log_store = turns
    # Real turn persistence: the claimed reservation must become the open turn.
    from llm_bawt.service.background_service import BackgroundService
    for name in ("_persist_turn_log", "_update_turn_log", "_bind_agent_thread",
                 "_resolve_active_thread_binding"):
        setattr(rt.service, name, getattr(BackgroundService, name).__get__(rt.service))
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = rt.service
    dispatcher.store = deliveries
    dispatcher.claim_owner = "fixture-dispatcher"
    dispatcher._dispatch_tasks = set()
    dispatcher._wake_event = asyncio.Event()
    dispatcher._loop = None
    dispatcher._emit = AsyncMock()
    dispatcher._announce = lambda record, wake=False: None
    dispatcher._heartbeat = AsyncMock()
    rt.service._inter_bot_dispatcher = dispatcher
    yield SimpleNamespace(rt=rt, dispatcher=dispatcher, deliveries=deliveries, turns=turns,
                          published=published, built=strict_runtime.built,
                          foreign=strict_runtime.foreign)
    from sqlalchemy import text
    with deliveries.engine.begin() as conn:
        conn.execute(text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"),
                     {"bot": rig.bot})


def _drain(dispatcher, bot):
    async def run():
        drained = await dispatcher._drain_target(bot)
        await asyncio.gather(*list(dispatcher._dispatch_tasks), return_exceptions=False)
        return drained
    return asyncio.run(run())


def test_durable_strict_turn_dispatches_through_queue_with_tools_off(durable, rig):
    from sqlalchemy import event

    rt = durable.rt
    params: list[str] = []

    def bind(_conn, _cursor, _statement, parameters, *_rest):
        params.append(repr(parameters))

    event.listen(rig.engine, "before_cursor_execute", bind)
    try:
        with capture_sql(rig.engine) as sql:
            _run_durable(rt, durable, rig)
    finally:
        event.remove(rig.engine, "before_cursor_execute", bind)
    _assert_durable(durable, rig, sql, params)


def _run_durable(rt, durable, rig):
    durable.receipt = _stream(rt.service, _request(rig))
    receipt = durable.receipt
    assert '"delivery_id"' in receipt and "strict answer" not in receipt, receipt
    assert durable.published == []  # HTTP admission never publishes a turn
    assert _drain(durable.dispatcher, rig.bot) is True


def _assert_durable(durable, rig, sql, params):
    receipt = durable.receipt
    # The claimed row ran as one ordinary start through the strict instance.
    assert [type(i) for i in durable.built] == [strict.StrictReaderLLMBawt] * len(durable.built)
    dispatched = durable.built[-1]
    assert dispatched.bot.slug == rig.bot and dispatched.client.tools_disabled is True
    delivery_id = re.search(r'"delivery_id": "([^"]+)"', receipt).group(1)
    record = durable.deliveries.get(delivery_id)
    assert record.status == "DELIVERED", (record.status, record.last_error)
    turn = durable.turns.get_turn(record.turn_id)
    assert turn.status == "ok" and turn.bot_id == rig.bot
    assert len(durable.published) == 1
    from agent_bridge.houses import commands_stream
    from tests.test_strict_reader_stream_pg import HOUSE
    assert _RecordingRedis.streams == [commands_stream(HOUSE)]  # its house, never the mansion
    fields = _command_fields(durable.published[0])
    assert fields["action"] == "chat.send" and fields["bot_id"] == rig.bot
    assert fields["tools_disabled"] == "1" and fields["backend"] == "claude-code"
    assert "inbox_next" not in fields
    system = fields.get("system_prompt", "")
    for ref in APPROVED:
        assert _body(ref) in system, ref
    for ref in (*FOREIGN, "fx-hidden-skill"):
        assert _body(ref) not in system, ref

    profile_reads = [s for s in sql.all_statements if "FROM bot_profiles" in s]
    assert all(re.search(r"slug\s*=", s) for s in profile_reads), profile_reads
    assert not any(durable.foreign in s for s in sql.all_statements)
    # Bind parameters too: the foreign slug is never a query argument.
    assert not any(durable.foreign in p for p in params)
    # No fleet-wide tag snapshot (working memory / assigned cards are sealed).
    assert not any("GROUP BY p.slug" in s for s in sql.all_statements)


def test_forged_strict_delivery_without_validated_claim_refuses(strict_runtime, rig):  # noqa: F811
    request = _request(rig, inter_bot_delivery_id="forged")
    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(strict_runtime.rt.service, request)
    assert strict_runtime.built == []
