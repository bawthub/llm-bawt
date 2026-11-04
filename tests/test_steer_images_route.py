"""TASK-774: the steer route carries uploaded images to a live run."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from contextlib import ExitStack

from agent_bridge.steer_contract import STEER_IMAGES_UNSUPPORTED
from tests.steer_session_fixture import owned_steer_store
from tests.test_steer_capabilities import _harness, _turn

IMAGE = {"mimeType": "image/png", "content": "UE5H"}
REF = {"asset_id": "ma_1", "kind": "image"}


def _steer(chat, service, *, ids=("ma_1",), priority="next", **kw):
    return asyncio.run(chat.steer_active_turn(service, chat.ChatSteerRequest(
        turn_id="turn-live", bot_id="target", user_id="nick", message_id="message-steer",
        message="what is this?", priority=priority, attachment_ids=list(ids),
    ), **kw))


@pytest.fixture
def store_factory():
    with ExitStack() as stack:
        yield lambda turn: stack.enter_context(owned_steer_store(turn))


@pytest.fixture
def media(monkeypatch):
    from llm_bawt.service import stream_request_attachments as resolver
    import llm_bawt.media as media_pkg
    import llm_bawt.media.serializers as serializers

    resolved = {"value": ([IMAGE], [REF])}

    async def resolve(ids, store, *, log):
        return resolved["value"]

    monkeypatch.setattr(resolver, "resolve_attachment_ids", resolve)
    monkeypatch.setattr(media_pkg, "get_media_store", lambda: SimpleNamespace(db=None))
    monkeypatch.setattr(serializers, "build_agent_image_manifest", lambda *a, **k: "MANIFEST")
    return resolved


def test_images_ride_the_steer_with_the_manifest_outbound_only(monkeypatch, store_factory, media):
    chat, service, subscriber = _harness(
        monkeypatch, store_factory, _turn(agent_backend="claude-code", agent_steer_caps="images,next,now"))
    memory = MagicMock()
    service.get_memory_client = lambda *args: memory

    class _Timeline:
        def __init__(self, *a, **k): pass
        async def content_offset(self, *a): return None
        async def publish(self, *a): return None

    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", _Timeline)
    _steer(chat, service)
    kwargs = subscriber.send_steer.await_args.kwargs
    assert kwargs["attachments"] == [IMAGE]
    assert kwargs["message"] == "what is this?\n\nMANIFEST"
    # History keeps the clean text plus the durable refs on the steer message.
    args = memory.add_message.call_args.args
    assert args[1] == "what is this?" and args[5] == [REF]


def test_text_only_steer_sends_no_attachments_field(monkeypatch, store_factory, media):
    chat, service, subscriber = _harness(
        monkeypatch, store_factory, _turn(agent_backend="claude-code", agent_steer_caps="images,next"))
    _steer(chat, service, ids=())
    assert "attachments" not in subscriber.send_steer.await_args.kwargs


@pytest.mark.parametrize("caps", ["cancel_queued,escalate,next,now", None])
def test_a_run_without_the_images_capability_refuses_before_any_rpc(
    monkeypatch, store_factory, media, caps,
):
    # A run pinned before TASK-774 (or a legacy NULL row) steers text only.
    chat, service, subscriber = _harness(
        monkeypatch, store_factory, _turn(agent_backend="claude-code", agent_steer_caps=caps))
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service)
    assert (raised.value.status_code, raised.value.detail) == (409, STEER_IMAGES_UNSUPPORTED)
    subscriber.send_steer.assert_not_awaited()


def test_a_partially_resolved_set_is_refused_not_sent_short(monkeypatch, store_factory, media):
    media["value"] = ([IMAGE], [REF])
    chat, service, subscriber = _harness(
        monkeypatch, store_factory, _turn(agent_backend="claude-code", agent_steer_caps="images,next"))
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service, ids=("ma_1", "ma_missing"))
    assert (raised.value.status_code, raised.value.detail) == (422, "steer_attachment_unresolved")
    subscriber.send_steer.assert_not_awaited()


def test_a_delivery_steer_cannot_carry_images(monkeypatch, store_factory, media):
    chat, service, subscriber = _harness(
        monkeypatch, store_factory, _turn(agent_backend="claude-code", agent_steer_caps="images,next"))
    with pytest.raises(HTTPException) as raised:
        _steer(chat, service, steer_request_id="steer_delivery_x")
    assert raised.value.detail == STEER_IMAGES_UNSUPPORTED
    subscriber.send_steer.assert_not_awaited()
