"""TASK-1252: repairing accepted-input history must never send another input."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.steer_persistence import SteerDestination, persist_accepted_input
from tests import test_steer_session

rig = test_steer_session.rig


def persist(rig, **changes):
    options = dict(
        destination=SteerDestination("target", "operator", "turn", "trigger", "durable-A"),
        message_id="message", message="saved bot input",
        author=AuthorReference("bot", "source"), subscriber=rig.sub,
        persist_anchor=Mock(return_value=True),
    )
    return asyncio.run(persist_accepted_input(rig.service, **(options | changes)))


def test_writer_uses_pinned_destination_and_authorship_without_sdk_or_relookup(rig):
    rig.service._turn_log_store.get_turn = Mock(side_effect=AssertionError("must not retarget"))
    assert persist(rig, attachment_refs=[{"asset_id": "image"}])
    assert rig.memory.add_message.call_args.args == (
        "user", "saved bot input", "durable-A", None, "message",
        [{"asset_id": "image"}], None, "bot", "source",
    )
    rig.sub.send_steer.assert_not_awaited()
    rig.service._turn_log_store.get_turn.assert_not_called()


def test_history_failure_can_be_repaired_using_same_message_without_new_input(rig):
    rig.memory.add_message.side_effect = [RuntimeError("history unavailable"), None]
    assert not persist(rig)
    assert persist(rig)
    first, second = rig.memory.add_message.call_args_list
    assert first == second
    rig.sub.send_steer.assert_not_awaited()


def test_missing_memory_client_is_not_successful_persistence(rig):
    rig.service.get_memory_client = lambda *_: None
    assert not persist(rig)
    rig.memory.add_message.assert_not_called()
    rig.sub.send_steer.assert_not_awaited()


def test_required_anchor_failure_stays_unpersisted_until_repair(rig, monkeypatch):
    timeline = SimpleNamespace(content_offset=AsyncMock(return_value=12), publish=AsyncMock())
    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", lambda *_a, **_kw: timeline)
    anchor = Mock(side_effect=[False, True])
    assert not persist(rig, persist_anchor=anchor)
    timeline.publish.assert_not_awaited()
    assert persist(rig, persist_anchor=anchor)
    first, second = rig.memory.add_message.call_args_list
    assert first == second
    assert anchor.call_args.kwargs == {
        "bot_id": "target", "message_id": "message", "source_message_id": "trigger", "content_offset": 12,
    }
    timeline.publish.assert_awaited_once()
    rig.sub.send_steer.assert_not_awaited()


def test_timeline_publication_failure_does_not_undo_successful_history(rig, monkeypatch):
    timeline = SimpleNamespace(
        content_offset=AsyncMock(return_value=None),
        publish=AsyncMock(side_effect=RuntimeError("event stream unavailable")),
    )
    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", lambda *_a, **_kw: timeline)
    assert persist(rig)
    rig.memory.add_message.assert_called_once()
    rig.sub.send_steer.assert_not_awaited()


def test_cancellation_cannot_be_returned_as_a_persistence_success(rig, monkeypatch):
    timeline = SimpleNamespace(
        content_offset=AsyncMock(side_effect=asyncio.CancelledError), publish=AsyncMock(),
    )
    monkeypatch.setattr("llm_bawt.service.steer_timeline.SteerTimeline", lambda *_a, **_kw: timeline)
    with pytest.raises(asyncio.CancelledError):
        persist(rig)
    rig.memory.add_message.assert_not_called()
    rig.sub.send_steer.assert_not_awaited()
