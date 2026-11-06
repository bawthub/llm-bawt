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
        gt.schedule_session_title("E", session_id="once", bot_id="b", user_id="u", message_id="m", content="hi")
    assert len(started) == 1


def test_chat_write_path_schedules_title_for_user_messages(monkeypatch):
    """Chat turns write through the short-term manager, not MCP storage."""
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    calls, rows = [], []
    monkeypatch.setattr(gt, "schedule_session_title", lambda engine, **kw: calls.append((engine, kw)))
    manager = object.__new__(PostgreSQLShortTermManager)
    manager.bot_id, manager.user_id = "bot-a", "nick"
    manager._backend = type("B", (), {"engine": "ENG", "add_message": lambda self, **kw: rows.append(kw)})()

    mid = manager.add_message("user", "How do I rebase?", session_id="s1")
    manager.add_message("assistant", "Like this", session_id="s1")

    assert len(rows) == 2
    assert calls == [("ENG", {"session_id": "s1", "bot_id": "bot-a", "user_id": "nick",
                              "message_id": mid, "content": "How do I rebase?"})]


def test_chat_write_survives_title_hook_failure(monkeypatch):
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    def boom(engine, **kw):
        raise RuntimeError("x")
    monkeypatch.setattr(gt, "schedule_session_title", boom)
    manager = object.__new__(PostgreSQLShortTermManager)
    manager.bot_id, manager.user_id = "bot-a", "nick"
    manager._backend = type("B", (), {"engine": None, "add_message": lambda self, **kw: None})()
    assert manager.add_message("user", "hi", session_id="s1")


def test_schedule_resolves_session_from_message_when_unknown(monkeypatch):
    """App chat writes go via MCP and don't know the resolved thread."""
    import llm_bawt.service.dependencies as deps
    monkeypatch.setattr(deps, "_service", object())
    monkeypatch.setattr(gt, "_attempted", set()); monkeypatch.setattr(gt, "_pending", set()); monkeypatch.setattr(gt, "_triggers", {}); monkeypatch.setattr(gt, "_running", set())
    workers = []

    class T:
        def __init__(self, target, args, **kw):
            self.target, self.args = target, args
        def start(self):
            if self.target is gt._worker:
                workers.append(self.args)
            else:
                self.target(*self.args)
    monkeypatch.setattr(gt.threading, "Thread", T)

    class Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params):
            assert params == {"id": "m1"}
            return type("R", (), {"scalar": lambda self: "sess-9"})()
    engine = type("E", (), {"connect": lambda self: Conn()})()

    gt.schedule_session_title(engine, session_id=None, bot_id="b", user_id="u", message_id="m1", content="hi")
    assert workers == [(engine, "sess-9", "b", "u")]
    assert gt._triggers == {"sess-9": {"m1"}}


def test_mcp_client_write_schedules_title_for_user_messages(monkeypatch):
    from llm_bawt.mcp_server.client_short_term import _MCPShortTermManager

    calls = []
    monkeypatch.setattr(gt, "schedule_session_title", lambda engine, **kw: calls.append((engine, kw)))
    principal = type("P", (), {"bot_id": "codex", "user_id": "nick"})()
    msgs = iter(["m-user", "m-bot"])
    client = type("C", (), {"_service_principal": principal,
                            "add_message": lambda self, **kw: type("M", (), {"id": next(msgs), "session_id": None})()})()
    manager = _MCPShortTermManager(client)

    assert manager.add_message("user", "How do I rebase?") == "m-user"
    manager.add_message("assistant", "Like this")
    assert calls == [(None, {"session_id": None, "bot_id": "codex", "user_id": "nick",
                             "message_id": "m-user", "content": "How do I rebase?"})]
