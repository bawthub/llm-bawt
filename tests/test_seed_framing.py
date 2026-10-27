"""Seed framing is registry-owned and rides the bridge command (TASK-1579)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_bridge.command_publisher import CommandPublisherMixin
from agent_bridge.continuity import SEED_FRAMING_FIELD, SeedFraming, SeedFramingMissing
from llm_bawt import seed_framing
from llm_bawt.chat_prompt_defaults import SESSION_SEED_ACK, SESSION_SEED_BRIEFING
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

SEED = [{"role": "user", "content": "hi"}]


def _resolver(monkeypatch, bodies=None, error=None):
    calls = []

    def resolve(key, scope_type="global", scope_id=None):
        calls.append((key, scope_type, scope_id))
        if error is not None:
            raise error
        body = (bodies or {}).get(key)
        return SimpleNamespace(body=body) if body is not None else None

    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver",
                        lambda: SimpleNamespace(resolve=resolve))
    return calls


def test_both_framing_prompts_are_registered_with_their_defaults():
    by_key = DEFAULT_PROMPT_DEFINITIONS
    assert by_key[seed_framing.SEED_BRIEFING_KEY].default_body() == SESSION_SEED_BRIEFING
    assert by_key[seed_framing.SEED_ACK_KEY].default_body() == SESSION_SEED_ACK


def test_empty_seed_needs_no_framing(monkeypatch):
    _resolver(monkeypatch, error=AssertionError("not resolved"))
    assert seed_framing.resolve_seed_framing([], "bot") is None
    assert seed_framing.resolve_seed_framing(None, "bot") is None


def test_bot_scope_override_wins(monkeypatch):
    calls = _resolver(monkeypatch, {seed_framing.SEED_BRIEFING_KEY: "BOT BRIEFING"})
    framing = seed_framing.resolve_seed_framing(SEED, "bot")
    assert framing == SeedFraming("BOT BRIEFING", SESSION_SEED_ACK)
    assert calls[0] == (seed_framing.SEED_BRIEFING_KEY, "bot", "bot")


@pytest.mark.parametrize("error", [None, RuntimeError("db down")])
def test_missing_or_failed_resolution_uses_the_registered_default(monkeypatch, error):
    _resolver(monkeypatch, {seed_framing.SEED_BRIEFING_KEY: "  "}, error=error)
    assert seed_framing.resolve_seed_framing(SEED, None) == SeedFraming(
        SESSION_SEED_BRIEFING, SESSION_SEED_ACK)


@pytest.mark.parametrize("raw", [
    None, "", "not-json", "[]", json.dumps({"briefing": "b"}),
    json.dumps({"briefing": " ", "ack": "a"}), json.dumps({"briefing": 1, "ack": "a"}),
])
def test_malformed_field_is_absent_and_a_seed_then_refuses(raw):
    assert SeedFraming.from_field(raw) is None
    with pytest.raises(SeedFramingMissing):
        SeedFraming.require(SeedFraming.from_field(raw), SEED)
    assert SeedFraming.require(None, []) is None


def test_field_round_trips():
    framing = SeedFraming("briefing — with unicode", "ack")
    assert SeedFraming.from_field(framing.to_field()) == framing


@pytest.mark.asyncio
@pytest.mark.parametrize("messages,framing,expected", [
    (SEED, SeedFraming("b", "a"), True),
    ([], SeedFraming("b", "a"), False),
    (SEED, None, False),
])
async def test_publisher_sends_framing_only_with_a_non_empty_seed(messages, framing, expected):
    sent = []

    async def xadd(_stream, fields, **_kwargs):
        sent.append(fields)
        return "1-0"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = SimpleNamespace(xadd=xadd)
    await publisher.send_command(
        "bot:user", "hello", "req", inject_messages=messages, inject_seed_framing=framing,
    )
    assert (SEED_FRAMING_FIELD in sent[0]) is expected
