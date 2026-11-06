"""TASK-2064: title cleaning and the non-blocking schedule guard (no DB)."""
import pytest

from llm_bawt.prompt_registry import get_prompt_resolver
from llm_bawt.sessions import generated_title as gt


@pytest.mark.parametrize("raw,expected", [
    ('"Git rebase question"', "Git rebase question"),
    ("Title: Fix flaky login test.", "Fix flaky login test"),
    ("**Weekend packing list**\n\nExtra explanation", "Weekend packing list"),
    ("“Curly quoted”", "Curly quoted"),
    ("one two three four five six seven eight nine ten eleven twelve", "one two three four five six seven eight nine ten"),
    ("", None), ("   ", None), (None, None),
    ("I'm sorry, I can't help with that", None),
    ("I cannot do that", None),
])
def test_clean_title(raw, expected):
    assert gt.clean_title(raw) == expected


def test_prompt_is_in_registry_and_renders_message():
    resolver = get_prompt_resolver()
    assert resolver.definition_for(gt.TITLE_PROMPT_KEY) is not None
    assert "hello world" in gt.TITLE_PROMPT_TEMPLATE.format(message="hello world")


def test_schedule_skips_commands_and_missing_service(monkeypatch):
    started = []
    monkeypatch.setattr(gt.threading, "Thread", lambda **kw: started.append(kw) or type("T", (), {"start": lambda self: None})())
    gt.schedule_session_title(None, session_id="s", bot_id="b", user_id="u", message_id="m", content="/new")
    gt.schedule_session_title(None, session_id=None, bot_id="b", user_id="u", message_id="m", content="hi")
    import llm_bawt.service.dependencies as deps
    monkeypatch.setattr(deps, "_service", None)
    gt.schedule_session_title(None, session_id="s", bot_id="b", user_id="u", message_id="m", content="hi")
    assert started == []


def test_schedule_runs_once_per_session(monkeypatch):
    started = []
    monkeypatch.setattr(gt.threading, "Thread", lambda **kw: started.append(kw) or type("T", (), {"start": lambda self: None})())
    import llm_bawt.service.dependencies as deps
    monkeypatch.setattr(deps, "_service", object())
    monkeypatch.setattr(gt, "_attempted", set()); monkeypatch.setattr(gt, "_pending", set()); monkeypatch.setattr(gt, "_triggers", {}); monkeypatch.setattr(gt, "_running", set())
    for _ in range(3):
        gt.schedule_session_title(None, session_id="once", bot_id="b", user_id="u", message_id="m", content="hi")
    assert len(started) == 1
