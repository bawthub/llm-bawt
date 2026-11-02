"""TASK-1956: notice a kept session prompt that names a switched-away model."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from llm_bawt import model_switch_notice as notice
from llm_bawt.chat_prompt_defaults import MODEL_SWITCH_NOTICE, RUNTIME_CONTEXT_TEMPLATE
from llm_bawt.models.message import Message
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS


@dataclass(frozen=True)
class Decision:
    text: str
    pin_turn_id: str = "turn-pin"
    is_boundary: bool = False


def pinned_for(model: str) -> str:
    return RUNTIME_CONTEXT_TEMPLATE.format(model=model) + "\n\nYou are a bot."


@pytest.fixture
def claims(monkeypatch):
    seen: dict[str, dict] = {}

    def fake_claim(engine, session_id, decision, current_model):
        model = notice.notice_model(seen.get(session_id), decision, current_model)
        if model is not None:
            seen[session_id] = {"pin_turn_id": decision.pin_turn_id, "model": model}
        return model

    monkeypatch.setattr(notice, "_claim", fake_claim)
    monkeypatch.setattr(notice, "render_notice", lambda model, bot_id: f"NOTICE {model}")
    return seen


def run(decision, current, *, eligible=True, session="s1"):
    return notice.model_switch_notice(
        object(), bot_id="loopy", session_id=session, decision=decision,
        current_model=current, eligible=eligible,
    )


def test_reads_model_from_the_registered_runtime_block():
    assert notice.pinned_model(pinned_for("deepseek-v4.1-flash")) == "deepseek-v4.1-flash"
    assert notice.pinned_model("no runtime block here") is None


def test_kept_pin_naming_old_model_notices_once(claims):
    decision = Decision(pinned_for("deepseek-v4.1-flash"))
    assert run(decision, "gpt-6.1-sol") == "NOTICE gpt-6.1-sol"
    assert run(decision, "gpt-6.1-sol") is None
    # Switching again within the same pin episode notices the new model.
    assert run(decision, "claude-opus-5-5") == "NOTICE claude-opus-5-5"


@pytest.mark.parametrize("decision,current,eligible", [
    (Decision(pinned_for("gpt-6.1-sol")), "gpt-6.1-sol", True),            # unchanged
    (Decision(pinned_for("old"), is_boundary=True), "new", True),          # fresh render
    (Decision("custom prompt without the block"), "new", True),            # unparseable
    (Decision(pinned_for("old")), "", True),                               # no current model
    (Decision(pinned_for("old")), "new", False),                           # /new, /compact
    (None, "new", True),                                                   # backend does not pin
])
def test_no_notice(claims, decision, current, eligible):
    assert run(decision, current, eligible=eligible) is None
    assert claims == {}


def test_return_to_pinned_model_and_repeat_switches_notice(claims):
    decision = Decision(pinned_for("A"))
    assert run(decision, "B") == "NOTICE B"
    assert run(decision, "B") is None
    assert run(decision, "A") == "NOTICE A"   # corrects the earlier "report B"
    assert run(decision, "A") is None
    assert run(decision, "B") == "NOTICE B"   # not suppressed by a stale mark
    assert claims["s1"] == {"pin_turn_id": "turn-pin", "model": "B"}


def test_marker_from_older_pin_is_ignored():
    old = {"pin_turn_id": "older", "model": "B"}
    assert notice.notice_model(old, Decision(pinned_for("A")), "A") is None
    assert notice.notice_model(old, Decision(pinned_for("A")), "B") == "B"


def test_new_pin_episode_can_notice_again(claims):
    assert run(Decision(pinned_for("old"), pin_turn_id="a"), "new")
    assert run(Decision(pinned_for("old"), pin_turn_id="b"), "new")


def test_failure_never_blocks_the_turn(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(notice, "_claim", boom)
    assert run(Decision(pinned_for("old")), "new") is None


def test_default_prompt_is_registered_and_renders():
    definition = DEFAULT_PROMPT_DEFINITIONS[notice.MODEL_SWITCH_NOTICE_KEY]
    assert definition.default_body() == MODEL_SWITCH_NOTICE
    assert "`gpt-6.1-sol`" in definition.default_body().format(model="gpt-6.1-sol")


def test_apply_appends_to_last_user_message(monkeypatch, claims):
    ctx = SimpleNamespace(
        svc=SimpleNamespace(_turn_log_store=SimpleNamespace(engine=object())),
        bot_id="loopy", thread_binding={"thread_session_id": "s1"}, user_prompt="hi",
    )
    client = SimpleNamespace(_bot_config={"model": "gpt-6.1-sol"})
    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]
    out = notice.apply_model_switch_notice(ctx, client, messages, Decision(pinned_for("old")))
    assert out[-1].content == "hi\n\nNOTICE gpt-6.1-sol"
    assert messages[-1].content == "hi"  # outbound-only copy
    ctx.user_prompt = "/compact"
    ctx.thread_binding = {"thread_session_id": "s2"}
    assert notice.apply_model_switch_notice(ctx, client, messages, Decision(pinned_for("old"))) is messages
