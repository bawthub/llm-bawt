"""Hermetic tests for the session-pinned agent system prompt (TASK-1318)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service import system_prompt_pin as pin_mod
from llm_bawt.service.system_prompt_pin import (
    COLD_START,
    COMPACTION,
    LEGACY_ADOPT,
    MISSING_ROW,
    MISSING_TEXT,
    SHA_MISMATCH,
    PinDecision,
    apply_session_pin,
    is_cold_start,
    pin_move_reason,
    prompt_sha256,
)

PINNED = "pinned prompt"
PIN = {"turn_id": "turn-pin", "sha256": prompt_sha256(PINNED)}


# ── decision table ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"cold_start": True, "pin": PIN, "pin_row_found": True, "pinned_text": PINNED, "compacted_since_pin": False}, COLD_START),
        ({"cold_start": True, "pin": None, "pin_row_found": False, "pinned_text": None, "compacted_since_pin": False}, COLD_START),
        ({"cold_start": False, "pin": None, "pin_row_found": False, "pinned_text": None, "compacted_since_pin": False}, LEGACY_ADOPT),
        ({"cold_start": False, "pin": PIN, "pin_row_found": False, "pinned_text": None, "compacted_since_pin": False}, MISSING_ROW),
        ({"cold_start": False, "pin": PIN, "pin_row_found": True, "pinned_text": None, "compacted_since_pin": False}, MISSING_TEXT),
        ({"cold_start": False, "pin": PIN, "pin_row_found": True, "pinned_text": "tampered", "compacted_since_pin": False}, SHA_MISMATCH),
        ({"cold_start": False, "pin": PIN, "pin_row_found": True, "pinned_text": PINNED, "compacted_since_pin": True}, COMPACTION),
        ({"cold_start": False, "pin": PIN, "pin_row_found": True, "pinned_text": PINNED, "compacted_since_pin": False}, None),
    ],
)
def test_pin_move_reason(kwargs, expected):
    assert pin_move_reason(**kwargs) == expected


@pytest.mark.parametrize(
    ("binding", "reset", "prompt", "expected"),
    [
        (None, None, "hi", True),
        ({"thread_session_id": "s"}, None, "hi", True),
        ({"thread_session_id": "s", "thread_resume_id": "r"}, None, "hi", False),
        ({"thread_session_id": "s", "thread_resume_id": "r"}, "reset_follow_up", "hi", True),
        ({"thread_session_id": "s", "thread_resume_id": "r"}, None, "/new hello", True),
        # Explicit threads never process /new; the bridge resumes them.
        ({"thread_session_id": "s", "thread_resume_id": "r", "explicit_thread": True}, None, "/new hello", False),
    ],
)
def test_is_cold_start_mirrors_bridge_preflight(binding, reset, prompt, expected):
    assert is_cold_start(binding, bridge_reset_action=reset, user_prompt=prompt) is expected


def test_manual_compaction_gate():
    ok = '{"context_action": "compact", "context_action_outcome": "success"}'
    assert pin_mod._manual_compaction_succeeded("/compact", ok)
    assert pin_mod._manual_compaction_succeeded("  /compact now", ok)
    # CLI auto-compaction inside an ordinary turn is not a pin boundary.
    assert not pin_mod._manual_compaction_succeeded("Message from bot 'x': hi", ok)
    assert not pin_mod._manual_compaction_succeeded("/compactness review", ok)
    assert not pin_mod._manual_compaction_succeeded(
        "/compact", '{"context_action": "compact", "context_action_outcome": "failed"}')
    assert not pin_mod._manual_compaction_succeeded(
        "/compact", '{"context_action": "compact", "context_action_outcome": "error"}')
    assert not pin_mod._manual_compaction_succeeded("/compact", None)
    assert not pin_mod._manual_compaction_succeeded("/compact", "not json")


WM = {"working_memory": {"v": 1, "version": 7, "state": "ok"}}
OLD_WM = {"working_memory": {"v": 1, "version": 3, "state": "ok"}}


@pytest.mark.parametrize(
    ("reason", "pin", "expected"),
    [
        (COLD_START, {**PIN, "sections": OLD_WM}, WM),
        (COMPACTION, {**PIN, "sections": OLD_WM}, WM),
        # The harness still holds the old snapshot after a fallback re-pin.
        (MISSING_ROW, {**PIN, "sections": OLD_WM}, OLD_WM),
        (MISSING_TEXT, {**PIN, "sections": OLD_WM}, OLD_WM),
        (SHA_MISMATCH, {**PIN, "sections": OLD_WM}, OLD_WM),
        (SHA_MISMATCH, PIN, None),
        (LEGACY_ADOPT, None, None),
    ],
)
def test_pointer_sections_follow_what_the_harness_holds(reason, pin, expected):
    assert pin_mod._held_sections(reason, WM, pin) == expected


@pytest.mark.parametrize(
    ("reason", "expected"),
    [(COLD_START, None), (COMPACTION, None), (MISSING_ROW, WM), (MISSING_TEXT, WM),
     (SHA_MISMATCH, WM), (LEGACY_ADOPT, WM)],
)
def test_candidate_sections_only_on_unverified_repins(reason, expected):
    assert pin_mod._candidate_sections(reason, WM) == expected


@pytest.mark.parametrize(
    ("row", "text"),
    [(False, None), (True, None), (True, "tampered")],
    ids=["missing_row", "missing_text", "sha_mismatch"],
)
def test_proven_compaction_beats_fallbacks(row, text):
    assert pin_move_reason(
        cold_start=False, pin=PIN, pin_row_found=row, pinned_text=text, compacted_since_pin=True,
    ) == COMPACTION


def test_sections_are_copied_and_malformed_entries_dropped():
    source = {"working_memory": {"v": 1}, "junk": "text"}
    copied = pin_mod._sections_value(source)
    assert copied == {"working_memory": {"v": 1}}
    copied["working_memory"]["v"] = 2
    assert source["working_memory"]["v"] == 1
    assert pin_mod._sections_value("nope") is None
    assert pin_mod._held_sections(COLD_START, None, PIN) is None


def test_decision_audit_records_drift_without_text():
    kept = PinDecision(
        text=PINNED, pin_turn_id="turn-pin", sha256=prompt_sha256(PINNED), action="kept",
        reason=None, is_boundary=False, rendered_sha256=prompt_sha256("fresh render"),
    )
    audit = kept.audit()
    assert kept.drift
    assert audit["rendered_sha256"] == prompt_sha256("fresh render")
    assert audit["pin_turn_id"] == "turn-pin" and audit["action"] == "kept"
    assert "text" not in audit and "reason" not in audit

    same = PinDecision(
        text=PINNED, pin_turn_id="t", sha256=prompt_sha256(PINNED), action="pinned",
        reason=MISSING_ROW, is_boundary=True, rendered_sha256=prompt_sha256(PINNED),
        snapshot_unverified=True,
    )
    assert not same.drift
    assert "rendered_sha256" not in same.audit()
    assert same.audit()["snapshot_unverified"] is True
    assert same.audit()["reason"] == MISSING_ROW


# ── worker hook ─────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, pins=True):
        self._backend = SimpleNamespace(pins_system_prompt=pins)
        self.composed: list[str] = []

    def compose_system_prompt(self, base):
        self.composed.append(base)
        return f"<rc>\n\n{base}"


def _ctx(binding=None, reset=None, prompt="hi"):
    return SimpleNamespace(
        svc=SimpleNamespace(_turn_log_store=SimpleNamespace(engine=object())),
        bot_id="byte", user_prompt=prompt, bridge_reset_action=reset,
        thread_binding=binding if binding is not None else {
            "thread_session_id": "sess", "thread_resume_id": "resume",
        },
    )


def _messages():
    return [Message("system", "base"), Message("user", "hello")]


def test_hook_skips_backends_that_do_not_pin():
    messages = _messages()
    out, decision = apply_session_pin(_ctx(), _Client(pins=False), messages, turn_id="t")
    assert out is messages and decision is None


def test_hook_skips_bare_reset_and_unbound_turns(monkeypatch):
    marks = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary",
                        lambda self, **kw: marks.append(kw))
    messages = _messages()
    assert apply_session_pin(_ctx(binding={}), _Client(), messages, turn_id="t") == (messages, None)
    assert apply_session_pin(_ctx(binding={}, reset="reset_only"), _Client(), messages, turn_id="t") == (messages, None)
    assert marks == []  # no session, nothing to mark


def test_bare_reset_marks_the_bound_session_and_pins_nothing(monkeypatch):
    marks = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary",
                        lambda self, **kw: marks.append(kw))
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve",
                        lambda self, **kw: pytest.fail("a bare reset never reaches a model"))
    messages = _messages()
    successor = {"thread_session_id": "successor"}  # /new binds the rotated successor
    out = apply_session_pin(_ctx(binding=successor, reset="reset_only", prompt="/new"),
                            _Client(), messages, turn_id="turn-new")
    assert out == (messages, None)
    assert marks == [{"session_id": "successor", "turn_id": "turn-new", "resume_id": None}]
    # Non-pinning backends do nothing at all.
    apply_session_pin(_ctx(reset="reset_only"), _Client(pins=False), messages, turn_id="t")
    assert len(marks) == 1


def test_bare_reset_marker_failure_fails_open(monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary", boom)
    messages = _messages()
    assert apply_session_pin(_ctx(reset="reset_only"), _Client(), messages, turn_id="t") == (messages, None)


def test_hook_passes_the_resume_id_to_resolve(monkeypatch):
    captured = {}
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve",
                        lambda self, **kw: captured.update(kw))
    apply_session_pin(_ctx(), _Client(), _messages(), turn_id="t")
    assert captured["resume_id"] == "resume"


@pytest.mark.parametrize("marker, resume, expected", [
    ({"turn_id": "r", "resume_id": None}, "seeded", True),   # bare /new seeded a new session
    ({"turn_id": "r", "resume_id": "old"}, "seeded", True),  # same-session reset re-keyed it
    ({"turn_id": "r", "resume_id": "old"}, "old", False),    # the reset never ran
    ({"turn_id": "r", "resume_id": None}, None, False),       # no resume id: cold anyway
    ({"resume_id": None}, "seeded", False),                   # malformed marker
    ("junk", "seeded", False),
])
def test_reset_boundary_needs_a_new_resume_id(marker, resume, expected):
    assert pin_mod._reset_boundary_applies(marker, resume) is expected


def test_hook_replaces_system_messages_with_pinned_bytes(monkeypatch):
    captured = {}
    decision = PinDecision(
        text="PINNED BYTES", pin_turn_id="turn-pin", sha256=prompt_sha256("PINNED BYTES"),
        action="kept", reason=None, is_boundary=False, rendered_sha256="x",
    )

    def fake_resolve(self, **kwargs):
        captured.update(kwargs)
        return decision

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve", fake_resolve)
    client = _Client()
    messages = [Message("system", "a"), Message("system", "b"), Message("user", "hello")]
    out, got = apply_session_pin(_ctx(), client, messages, turn_id="turn-now")

    assert got is decision
    assert [(m.role, m.content) for m in out] == [("system", "PINNED BYTES"), ("user", "hello")]
    assert messages[0].content == "a"  # the prepared list is not mutated
    assert client.composed == ["a\n\nb"]  # the client joins system parts the same way
    assert captured["rendered"] == "<rc>\n\na\n\nb"
    assert captured["session_id"] == "sess" and captured["turn_id"] == "turn-now"
    assert captured["cold_start"] is False and captured["storage_bot"] == "byte"


def test_hook_fails_open_to_the_fresh_render(monkeypatch):
    def boom(self, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve", boom)
    messages = _messages()
    assert apply_session_pin(_ctx(), _Client(), messages, turn_id="t") == (messages, None)


_BOUNDARY = PinDecision(
    text="P", pin_turn_id="t", sha256="s", action="pinned", reason=COLD_START,
    is_boundary=True, rendered_sha256="s",
)

# Codex r2: what a cancelled or pre-run turn can put on the queue before any model call.
BRIDGE_BOOKKEEPING = [
    {"event": "turn_cancelled", "abort_cause": None},
    {"event": "upstream_status", "text": "retrying"},
    {"event": "metadata", "upstream_model": "m"},
    {"event": "token_usage", "token_usage": {}},
    {"event": "attachments", "attachments": []},
    "",
]


def test_confirmer_waits_for_model_output_and_confirms_once(monkeypatch):
    calls = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "confirm", lambda self, **kw: calls.append(kw))
    confirmer = pin_mod.SessionPinConfirmer(_ctx(), _BOUNDARY, turn_id="t")
    for item in BRIDGE_BOOKKEEPING:
        confirmer.observe(item)
    assert calls == []
    confirmer.observe("Hello")
    confirmer.observe({"event": "tool_call"})
    assert calls == [{"session_id": "sess", "turn_id": "t"}]


@pytest.mark.parametrize("item", ["text", {"event": "reasoning"}, {"event": "tool_call"},
                                  {"event": "live_usage"}, {"event": "approval_required"}])
def test_model_output_proves_prompt_consumed(item):
    assert pin_mod.proves_prompt_consumed(item)


def test_confirmer_ignores_kept_turns_and_fails_open(monkeypatch):
    calls = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "confirm", lambda self, **kw: calls.append(kw))
    kept = PinDecision(
        text="P", pin_turn_id="old", sha256="s", action="kept", reason=None,
        is_boundary=False, rendered_sha256="s",
    )
    pin_mod.SessionPinConfirmer(_ctx(), None, turn_id="t").observe("Hello")
    pin_mod.SessionPinConfirmer(_ctx(), kept, turn_id="t").observe("Hello")
    assert calls == []

    def boom(self, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "confirm", boom)
    pin_mod.SessionPinConfirmer(_ctx(), _BOUNDARY, turn_id="t").observe("Hello")  # logged, not raised


# ── client / backend plumbing ───────────────────────────────────────────────

def _decision(boundary):
    return PinDecision(
        text="P", pin_turn_id="t", sha256=prompt_sha256("P"), action="pinned" if boundary else "kept",
        reason=COLD_START if boundary else None, is_boundary=boundary, rendered_sha256=prompt_sha256("P"),
    )


def _client_config(backend, pin):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    client = object.__new__(AgentBackendClient)
    client._backend_name = backend.name
    client._bot_config = {"bot_id": "b", "model": "m"}
    client._backend = backend
    client.config = None
    seen = {}

    def stream_raw(prompt, config, **kwargs):
        seen.update(config)
        return iter(())

    backend.stream_raw = stream_raw
    with patch(
        "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", return_value=0,
    ):
        list(client.stream_raw(
            [Message("system", "P"), Message("user", "hi")], system_prompt_pin=pin,
        ))
    return seen


def test_client_marks_pinned_prompt_and_codex_delivery():
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend
    from llm_bawt.agent_backends.codex import CodexBackend

    claude = _client_config(ClaudeCodeBackend(), _decision(False))
    assert claude["system_prompt"] == "P" and claude["system_prompt_pinned"] is True
    assert "system_prompt_delivery" not in claude

    assert _client_config(CodexBackend(), _decision(True))["system_prompt_delivery"] == "now"
    assert _client_config(CodexBackend(), _decision(False))["system_prompt_delivery"] == "cold_start"

    unpinned = _client_config(CodexBackend(), None)
    assert "system_prompt_pinned" not in unpinned and "system_prompt_delivery" not in unpinned


def test_backend_flags():
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend
    from llm_bawt.agent_backends.codex import CodexBackend
    from llm_bawt.agent_backends.base import AgentBackend

    assert ClaudeCodeBackend.pins_system_prompt and CodexBackend.pins_system_prompt
    assert not AgentBackend.pins_system_prompt
    assert ClaudeCodeBackend().system_prompt_delivery(True) is None


def test_claude_backend_sends_pinned_bytes_verbatim():
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    base = {"model": "claude-opus-4-7", "bot_id": "loopy", "provider_system_prompt": "Provider."}
    with (
        patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", return_value=[]),
        patch(
            "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
            return_value=iter(()),
        ) as parent_stream,
    ):
        composed = backend.compose_system_prompt("Bot.", base)
        list(backend.stream_raw("hi", {**base, "system_prompt": "Bot."}))
        assert parent_stream.call_args.args[1]["system_prompt"] == composed
        list(backend.stream_raw("hi", {**base, "system_prompt": composed, "system_prompt_pinned": True}))
        assert parent_stream.call_args.args[1]["system_prompt"] == composed
    assert composed.index("<runtime-context>") < composed.index("Bot.") < composed.index("Provider.")


def test_claude_backend_still_requires_model_when_pinned():
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

    with pytest.raises(ValueError, match="no 'model'"):
        list(ClaudeCodeBackend().stream_raw("hi", {"bot_id": "x", "system_prompt_pinned": True}))


# ── transport + codex bridge ────────────────────────────────────────────────

def _publish(**kwargs):
    from agent_bridge.subscriber import RedisSubscriber

    subscriber = RedisSubscriber("redis://localhost:6379/0")
    captured: dict = {}

    async def fake_xadd(_stream, fields, **_kwargs):
        captured.update(fields)
        return b"1"

    subscriber._pub_redis = SimpleNamespace(xadd=fake_xadd)
    asyncio.run(subscriber.send_command(
        session_key="s", message="hello", request_id="req-test", **kwargs,
    ))
    return captured


def test_publisher_carries_delivery_mode():
    assert _publish(system_prompt="P", system_prompt_delivery="cold_start")["system_prompt_delivery"] == "cold_start"
    assert "system_prompt_delivery" not in _publish(system_prompt="P")
    with pytest.raises(ValueError, match="delivery"):
        _publish(system_prompt="P", system_prompt_delivery="sometimes")


@pytest.mark.parametrize(
    ("delivery", "resuming", "expected"),
    [
        (None, True, "P"),  # app did not pin: every turn (TASK-288)
        (None, False, "P"),
        ("now", True, "P"),  # pin moved this turn
        ("now", False, "P"),
        ("cold_start", False, "P"),  # fresh thread (incl. bridge reset / retry)
        ("cold_start", True, None),  # resumed thread already holds it
        ("bogus", True, "P"),  # unknown mode fails safe to pasting
    ],
)
def test_codex_bridge_paste_decision(delivery, resuming, expected):
    from codex_bridge._bridge_helpers import system_prompt_for_attempt

    assert system_prompt_for_attempt("P", delivery, resuming=resuming) == expected
    assert system_prompt_for_attempt(None, delivery, resuming=resuming) is None


# ── TASK-1750: the reset turn's injected skills feed the /new receipt ──────

def _skill_ctx(refs, injected, reset="reset_only", resolved=True):
    ctx = _ctx(binding={"thread_session_id": "successor"}, reset=reset, prompt="/new")
    resolution = SimpleNamespace(injected=tuple(
        SimpleNamespace(ref=ref, tokens=tokens) for ref, tokens in injected))
    ctx.llm_bawt = SimpleNamespace(_injected_skill_refs=tuple(refs),
                                   _skill_resolution=resolution if resolved else None)
    return ctx


@pytest.mark.parametrize("reset", ["reset_only", "reset_follow_up"])
def test_reset_records_the_renders_injected_skills(monkeypatch, reset):
    recorded = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "record_reset_skills",
                        lambda self, **kw: recorded.append(kw))
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary", lambda self, **kw: True)
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "resolve", lambda self, **kw: None)
    ctx = _skill_ctx(["agent-system", "llm-bawt"], [("agent-system", 4000), ("llm-bawt", 3500)], reset=reset)
    apply_session_pin(ctx, _Client(), _messages(), turn_id="turn-new")
    assert recorded == [{"session_id": "successor", "facts": {
        "refs": ["agent-system", "llm-bawt"], "tokens": 7500, "turn_id": "turn-new"}}]


def test_reset_records_an_empty_render_but_never_an_unknown_one(monkeypatch):
    recorded = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "record_reset_skills",
                        lambda self, **kw: recorded.append(kw["facts"]))
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary", lambda self, **kw: True)
    apply_session_pin(_skill_ctx([], []), _Client(), _messages(), turn_id="t")
    apply_session_pin(_skill_ctx(["x"], [("x", 1)], resolved=False), _Client(), _messages(), turn_id="t")
    apply_session_pin(_skill_ctx(["x"], [("y", 1)]), _Client(), _messages(), turn_id="t")
    apply_session_pin(_skill_ctx(["x"], [("x", 1)], reset=None), _Client(), _messages(), turn_id="t")
    assert recorded == [{"refs": [], "tokens": 0, "turn_id": "t"}]


def test_reset_skill_record_failure_fails_open(monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "record_reset_skills", boom)
    marks = []
    monkeypatch.setattr(pin_mod.SystemPromptPinStore, "mark_reset_boundary",
                        lambda self, **kw: marks.append(kw))
    messages = _messages()
    assert apply_session_pin(_skill_ctx(["x"], [("x", 1)]), _Client(), messages, turn_id="t") == (messages, None)
    assert len(marks) == 1
