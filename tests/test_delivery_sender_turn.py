"""TASK-1249: deliveries record the sender's signed turn, never a caller field."""

import asyncio
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from llm_bawt import task_turn_context as codec
from llm_bawt.mcp_server import task_association
from llm_bawt.task_turn_context import (
    SENDER_TURN_CONTEXT_HEADER, TASK_TURN_CONTEXT_HEADER, mint_task_turn_context,
)
from tests.test_delivery_service_fields import (  # noqa: F401 - delivery_route is a fixture
    _body, _enqueued, _request, _send, delivery_route,
)

TURN = "turn-" + "c" * 32


@pytest.fixture
def fernet(monkeypatch):
    instance = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: instance)
    return instance


def _capability(bot_id="fixture-sender", turn_id=TURN):
    return mint_task_turn_context(
        session_id="12490000-0000-4000-8000-000000000001", turn_id=turn_id,
        trigger_message_id="12490000-0000-4000-8000-000000000002",
        bot_id=bot_id, user_id="fixture-user",
    )


def _bot_body(**kwargs):
    fields = {"session_policy": "continue", **kwargs}
    return _body(author_entity_type=None, author_entity_id=None, **fields)


@pytest.mark.parametrize("header", [SENDER_TURN_CONTEXT_HEADER, TASK_TURN_CONTEXT_HEADER])
def test_signed_sender_turn_is_stamped(delivery_route, fernet, header):
    _send(_bot_body(), _request({header: _capability()}))
    assert _enqueued(delivery_route)["sender_turn_id"] == TURN


@pytest.mark.parametrize("headers", [
    {},
    {SENDER_TURN_CONTEXT_HEADER: "not-a-token"},
    {SENDER_TURN_CONTEXT_HEADER: "__other_bot__"},
])
def test_unsigned_forged_or_foreign_turn_records_nothing_and_still_sends(
    delivery_route, fernet, headers,
):
    if headers.get(SENDER_TURN_CONTEXT_HEADER) == "__other_bot__":
        headers = {SENDER_TURN_CONTEXT_HEADER: _capability(bot_id="fixture-target")}
    _send(_bot_body(), _request(headers))
    assert _enqueued(delivery_route)["sender_turn_id"] is None


def test_unreadable_signing_key_records_nothing_and_still_sends(delivery_route, monkeypatch):
    def broken():
        raise PermissionError("secret.key")

    monkeypatch.setattr(codec, "_get_fernet", broken)
    _send(_bot_body(), _request({SENDER_TURN_CONTEXT_HEADER: "anything"}))
    assert _enqueued(delivery_route)["sender_turn_id"] is None


def test_user_authored_delivery_never_takes_a_turn(delivery_route, fernet):
    _send(_body(session_policy="continue"), _request({SENDER_TURN_CONTEXT_HEADER: _capability(bot_id="fixture-user")}))
    assert _enqueued(delivery_route)["sender_turn_id"] is None


def test_sender_header_alone_never_grants_the_gate_hold(fernet):
    """The separate header carries provenance only (self-model hold stays on the turn header)."""
    from llm_bawt import inter_bot_send as send
    from llm_bawt.service.system_gate_admission import ON_GATE_DENY

    body = _bot_body(message_kind="MODEL_SWITCH_CONTINUE", sender_bot_id="fixture-sender",
                     target_bot_id="fixture-sender")
    request = _request({SENDER_TURN_CONTEXT_HEADER: _capability()})
    assert send._gate_disposition(request, body, "fixture-sender", "fixture-sender") == ON_GATE_DENY


def test_reset_admission_carries_the_sender_turn(delivery_route, fernet, monkeypatch):
    from llm_bawt.service.session_lifecycle import lifecycle

    seen = {}

    def admit(self, intent, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(id="reset-delivery"), False

    monkeypatch.setattr(lifecycle.NewSessionLifecycle, "__init__", lambda self, service: None)
    monkeypatch.setattr(lifecycle.NewSessionLifecycle, "admit_bot_delivery", admit)
    _send(_bot_body(session_policy="reset_retain_history"),
          _request({SENDER_TURN_CONTEXT_HEADER: _capability()}))
    assert seen["sender_turn_id"] == TURN


class _Client:
    def __init__(self, sink):
        self.sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers=None, timeout=None):
        self.sink.append(headers)
        return SimpleNamespace(raise_for_status=lambda: None, status_code=202,
                               json=lambda: {"delivery_id": "d", "batch_id": "b"})


def _single(tools):
    return tools._enqueue_durable(
        prefer_steer=False, target_bot_id="t", message="m", sender_bot_id="s",
        max_tokens=None, temperature=0.7, timeout_seconds=10, idempotency_key=None,
        project_id=None, task_id=None, message_kind=None, metadata=None,
        session_policy=None, reset_session_before_delivery=None, retain_history=None,
        reset_reason=None,
    )


def _batch(fanout):
    from llm_bawt.mcp_server.inter_bot_send_mode import WHEN_IDLE, SendMode

    return fanout.send_batch("http://app", targets=["tag:x"], mode=SendMode(WHEN_IDLE, False),
                             exclude=[], fields={"sender_bot_id": "s", "message": "m"})


@pytest.mark.parametrize("path", ["single", "batch"])
def test_mcp_send_forwards_the_capability_only_under_the_sender_header(monkeypatch, path):
    from llm_bawt.mcp_server import server  # noqa: F401 - registers tools first
    from llm_bawt.mcp_server import inter_bot_fanout, inter_bot_tools

    sent: list = []
    module = inter_bot_tools if path == "single" else inter_bot_fanout
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda: _Client(sent))
    call = (lambda: _single(inter_bot_tools)) if path == "single" else (lambda: _batch(inter_bot_fanout))

    asyncio.run(call())
    token = task_association.set_current_task_turn_capability("signed-turn")
    try:
        asyncio.run(call())
    finally:
        task_association._current_capability.reset(token)

    assert not (sent[0] or {}).get(SENDER_TURN_CONTEXT_HEADER)
    assert sent[1] == {SENDER_TURN_CONTEXT_HEADER: "signed-turn"}
