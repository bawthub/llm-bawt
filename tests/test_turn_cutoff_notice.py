"""A system cut-off reaches the next turn as a notice, not a user rejection (TASK-1738)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_bawt import turn_cutoff_notice as notice
from llm_bawt.chat_prompt_defaults import TURN_CUTOFF_NOTICE
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

ENDED = datetime(2026, 10, 29, 19, 54, 57, tzinfo=timezone.utc)
SHUTDOWN = json.dumps({"abort_cause": {"cause": "host_shutdown", "at": "x"}})


def _row(**fields):
    return {"status": "ok", "ended_at": ENDED, "request_json": None, "error_text": None, **fields}


class _Engine:
    """Answers the previous-turn query with ``row`` and records its parameters."""

    def __init__(self, row=None, error=None):
        self.row, self.error, self.params = row, error, None

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, _sql, params):
                if engine.error is not None:
                    raise engine.error
                engine.params = params
                return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: engine.row))
        return _Conn()


@pytest.fixture
def registry(monkeypatch):
    bodies = {}

    def resolve(key, scope_type="global", scope_id=None):
        body = bodies.get(key, DEFAULT_PROMPT_DEFINITIONS[key].default_body())
        return SimpleNamespace(body=body, source="db")

    from llm_bawt.prompt_registry import PromptResolver
    resolver = PromptResolver.__new__(PromptResolver)
    resolver.resolve = resolve
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: resolver)
    return bodies


def test_notice_is_registered_with_its_default():
    definition = DEFAULT_PROMPT_DEFINITIONS[notice.TURN_CUTOFF_NOTICE_KEY]
    assert definition.default_body() == TURN_CUTOFF_NOTICE
    assert set(definition.required_vars) == {"cause", "ended_at"}


@pytest.mark.parametrize("row, cause", [
    (_row(status="aborted", request_json=SHUTDOWN), notice.RESTART_CAUSE),
    (_row(status="error", error_text="bridge run req_x produced no event for 600s"),
     notice.WATCHDOG_CAUSE),
])
def test_system_cutoffs_are_recognized(row, cause):
    assert notice.cutoff_of(row) == notice.TurnCutoff(cause, ENDED)


@pytest.mark.parametrize("row", [
    None,
    _row(),
    _row(status="aborted"),  # no machine cause: never inferred
    _row(status="aborted", request_json=json.dumps(  # the user's own Stop
        {"abort_request": {"by": "user"}, "abort_cause": {"cause": "host_shutdown"}})),
    _row(status="error", error_text="codex error: model at capacity"),
    _row(status="aborted", request_json=SHUTDOWN, ended_at=None),
])
def test_other_ends_are_not_cutoffs(row):
    assert notice.cutoff_of(row) is None


def test_the_real_watchdog_error_is_recognized():
    import asyncio

    from agent_bridge.subscriber import RedisSubscriber

    async def run():
        sub = RedisSubscriber.__new__(RedisSubscriber)
        sub._redis = None  # an expired deadline raises before any read
        async for _ in sub.subscribe_run("req_x", timeout_s=0):
            pass

    with pytest.raises(TimeoutError) as error:
        asyncio.run(run())
    row = _row(status="error", error_text=str(error.value))
    assert notice.cutoff_of(row) == notice.TurnCutoff(notice.WATCHDOG_CAUSE, ENDED)


def test_next_turn_after_restart_gets_the_notice(registry):
    engine = _Engine(_row(status="aborted", request_json=SHUTDOWN))
    out = notice.with_cutoff_notice("do the thing", engine=engine, bot_id="al",
                                    session_key="al:nick", request_id="req_new")
    assert out == TURN_CUTOFF_NOTICE.format(
        cause=notice.RESTART_CAUSE, ended_at="2026-10-29 19:54:57Z") + "\n\ndo the thing"
    assert "Nobody rejected" in out
    assert engine.params == {"bot_id": "al", "session_key": "al:nick", "request_id": "req_new"}


def test_bot_override_is_used(registry):
    registry[notice.TURN_CUTOFF_NOTICE_KEY] = "CUT by {cause} at {ended_at}"
    engine = _Engine(_row(status="error", error_text="x produced no event for 600s"))
    out = notice.with_cutoff_notice("m", engine=engine, bot_id="al",
                                    session_key="al:nick", request_id="r")
    assert out == f"CUT by {notice.WATCHDOG_CAUSE} at 2026-10-29 19:54:57Z\n\nm"


def test_broken_override_falls_back_to_the_default(registry):
    registry[notice.TURN_CUTOFF_NOTICE_KEY] = "missing {nope}"
    engine = _Engine(_row(status="aborted", request_json=SHUTDOWN))
    out = notice.with_cutoff_notice("m", engine=engine, bot_id="al",
                                    session_key="al:nick", request_id="r")
    assert out.startswith("[System notice:")


@pytest.mark.parametrize("engine", [
    None, _Engine(_row()), _Engine(None), _Engine(error=RuntimeError("db down")),
])
def test_prompt_is_unchanged_without_a_cutoff(registry, engine):
    assert notice.with_cutoff_notice("m", engine=engine, bot_id="al",
                                     session_key="al:nick", request_id="r") == "m"


@pytest.mark.parametrize("prompt, reset_action", [
    ("/new start fresh", None),       # typed /new: the bridge must still see the command
    ("  /new", None),
    ("/clear", None),
    ("/compact keep the plan", None),
    ("hello", "reset_only"),          # structured reset: fresh session, nothing to explain
    ("hello", "reset_retain_history"),
])
def test_cutoff_then_session_command_is_left_untouched(registry, prompt, reset_action):
    from agent_bridge.maintenance import is_new_command

    engine = _Engine(_row(status="aborted", request_json=SHUTDOWN))
    out = notice.with_cutoff_notice(prompt, engine=engine, bot_id="al", session_key="al:nick",
                                    request_id="r", reset_action=reset_action)
    assert out == prompt
    assert engine.params is None  # no lookup at all
    if prompt.strip().startswith("/new"):
        assert is_new_command(out)


def test_newspaper_is_not_a_command(registry):
    engine = _Engine(_row(status="aborted", request_json=SHUTDOWN))
    out = notice.with_cutoff_notice("/newspaper idea", engine=engine, bot_id="al",
                                    session_key="al:nick", request_id="r")
    assert out.startswith("[System notice:")
