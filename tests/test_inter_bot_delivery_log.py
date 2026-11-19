"""TASK-1308: one payload-free, single-line delivery_event per lifecycle emit."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

from llm_bawt.service.inter_bot_delivery_log import delivery_event_line, error_kind
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

SENTINEL = "SENTINEL-message-body-7f3a"


def _record(**overrides):
    base = dict(
        id="dlv-1", target_bot_id="target-bot", sender_bot_id="sender-bot",
        author_entity_id="owner-user", author_entity_type="user", status="QUEUED", delivery_mode="steer",
        attempt_count=2, revision=5, next_retry_at=datetime(2026, 1, 1, tzinfo=UTC),
        last_error=None, message=SENTINEL, metadata={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_line_has_the_fields_and_no_message_or_raw_error():
    line = delivery_event_line(
        _record(last_error=f"boom: {SENTINEL}\ninjected=\"x\""), steer_uncertain=False,
    )

    assert "\n" not in line
    assert SENTINEL not in line
    assert "injected" not in line
    for field in ("delivery_id=\"dlv-1\"", "target=\"target-bot\"", "sender=\"sender-bot\"",
                  "author_type=\"user\"", "author=\"owner-user\"",
                  "status=\"QUEUED\"", "mode=\"steer\"", "attempt=\"2\"", "revision=\"5\"",
                  "steer_uncertain=false", "has_error=true", "error_kind=\"other\""):
        assert field in line


def test_known_errors_get_stable_kinds():
    assert error_kind(None) is None
    assert error_kind("delivery lease expired; retrying same logical message") == "lease_expired"
    assert error_kind("Stopped by user") == "user_stop"
    assert error_kind(f"Traceback {SENTINEL}") == "other"


def test_values_are_escaped_onto_one_line():
    line = delivery_event_line(_record(target_bot_id="a\nb\" x=1"), steer_uncertain=True)
    assert "\n" not in line
    assert 'target="a\\nb\\" x=1"' in line


def test_emit_logs_even_without_a_redis_subscriber(caplog):
    d = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    d.service = SimpleNamespace(_redis_subscriber=None)
    record = _record(status="DELIVERED")

    with caplog.at_level(logging.INFO, logger="llm_bawt.delivery_event"):
        asyncio.run(d._emit(record))

    lines = [r.getMessage() for r in caplog.records if r.name == "llm_bawt.delivery_event"]
    assert len(lines) == 1 and lines[0].startswith("delivery_event ")
    assert SENTINEL not in lines[0]
