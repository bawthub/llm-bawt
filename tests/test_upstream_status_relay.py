"""TASK-1074: upstream_status relay contract + in-memory live phase."""

from __future__ import annotations

from types import SimpleNamespace

from llm_bawt.service.upstream_status import (
    LiveTurnPhases,
    upstream_status_queue_item,
    upstream_status_sse_payload,
)


def _event(status: dict, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        raw={"upstream_status": status},
        text=kw.get("text", "Thinking"),
        provider=kw.get("provider", "zai"),
        trigger_message_id=kw.get("trigger_message_id", "msg_1"),
    )


def test_phase_fields_survive_both_relay_hops():
    status = {
        "phase": "reasoning", "state": "reasoning", "since": 1000.0,
        "model": "glm-5", "context_tokens": 4200,
    }
    item = upstream_status_queue_item(_event(status))
    payload = upstream_status_sse_payload(
        item, turn_id="turn_1", trigger_message_id=None, bot_id="caid", user_id="nick",
    )

    assert payload["_type"] == "upstream_status"
    assert payload["turn_id"] == "turn_1"
    assert payload["trigger_message_id"] == "msg_1"
    assert payload["text"] == "Thinking"
    for field, value in status.items():
        assert payload[field] == value
    assert payload["provider"] == "zai"  # falls back to the event's provider


def test_legacy_proxy_fields_still_relay():
    item = upstream_status_queue_item(_event({
        "state": "reconnecting", "attempt": 1, "fallback_transport": "sse",
        "provider": "openai_chatgpt",
    }))
    assert item["state"] == "reconnecting"
    assert item["fallback_transport"] == "sse"
    assert item["provider"] == "openai_chatgpt"
    assert item["phase"] is None


def _status(turn_id: str, phase: str, **kw) -> dict:
    return {
        "_type": "upstream_status", "turn_id": turn_id, "phase": phase,
        "since": kw.get("since", 1.0), "text": kw.get("text", phase),
        "model": "glm-5", "context_tokens": 10, "attempt": 3,
    }


def test_live_phase_keeps_latest_and_clears_on_turn_complete():
    phases = LiveTurnPhases()
    phases.observe(_status("t1", "waiting_model"))
    phases.observe(_status("t1", "reasoning", since=5.0, text="Thinking"))

    assert phases.get("t1") == {
        "phase": "reasoning", "since": 5.0, "text": "Thinking",
        "model": "glm-5", "context_tokens": 10,
    }
    phases.observe({"_type": "turn_complete", "turn_id": "t1"})
    assert phases.get("t1") is None


def test_live_phase_ignores_phaseless_and_unrelated_events():
    phases = LiveTurnPhases()
    phases.observe({"_type": "upstream_status", "turn_id": "t1", "state": "reconnecting"})
    phases.observe({"_type": "text_delta", "turn_id": "t1", "delta": "x"})
    phases.observe(_status("", "reasoning"))
    assert phases.get("t1") is None


def test_live_phase_expires_after_ttl():
    now = [100.0]
    phases = LiveTurnPhases(ttl=10, clock=lambda: now[0])
    phases.observe(_status("t1", "stalled"))
    now[0] = 109.0
    assert phases.get("t1")["phase"] == "stalled"
    now[0] = 111.0
    assert phases.get("t1") is None
