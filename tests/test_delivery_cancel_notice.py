"""Unit tests for TASK-1327 D2's cancelled-delivery notice: text, mode and registration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bawt import runtime_setting_resolution
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
from llm_bawt.service import delivery_cancel_notice as cn
from llm_bawt.service.delivery_cancel_notice_prompt import DEFAULT_CANCEL_NOTICE, PROMPT_KEY
from llm_bawt.setting_definitions import SETTING_DEFINITIONS

VARS = {"delivery_id": "d-1", "target_bot_id": "loopy", "task_id": "TASK-9",
        "reason": "Stopped by user", "excerpt": "run the migration"}


def _pump(engine=None):
    return cn.CancelNoticePump(SimpleNamespace(engine=engine, config=None))


def _setting(monkeypatch, value):
    def resolve(config, key, fallback=None):
        assert key == cn.MODE_SETTING
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runtime_setting_resolution, "resolve_global_runtime_setting", resolve)


def test_excerpt_flattens_whitespace_and_caps_the_length():
    assert cn.excerpt("  run\n the\tmigration  ") == "run the migration"
    assert cn.excerpt(None) == ""
    long = cn.excerpt("x" * (cn.EXCERPT_CHARS * 2))
    assert len(long) == cn.EXCERPT_CHARS and long.endswith("…")
    assert cn.excerpt("y" * cn.EXCERPT_CHARS) == "y" * cn.EXCERPT_CHARS


def test_the_prompt_is_registered_with_every_variable_the_notice_renders():
    definition = DEFAULT_PROMPT_DEFINITIONS[PROMPT_KEY]
    assert set(definition.required_vars) == set(VARS)
    assert definition.default_body() == DEFAULT_CANCEL_NOTICE
    rendered = DEFAULT_CANCEL_NOTICE.format(**VARS)
    assert all(value in rendered for value in VARS.values())


@pytest.mark.parametrize("failure", ["blank", "none", "error"])
def test_notice_text_never_sends_the_seed_in_place_of_the_registry(monkeypatch, failure):
    import llm_bawt.prompt_registry as registry

    def render(**kwargs):
        if failure == "error":
            raise RuntimeError("registry down")
        return "   " if failure == "blank" else None

    monkeypatch.setattr(registry, "get_prompt_resolver", lambda: SimpleNamespace(render=render))
    with pytest.raises((RuntimeError, ValueError)):
        cn.notice_text("caid", VARS)


def test_notice_text_uses_the_senders_rendered_body(monkeypatch):
    import llm_bawt.prompt_registry as registry

    seen = {}

    def render(**kwargs):
        seen.update(kwargs)
        return "custom {delivery_id}".format(**kwargs["variables"])

    monkeypatch.setattr(registry, "get_prompt_resolver", lambda: SimpleNamespace(render=render))
    assert cn.notice_text("caid", VARS) == "custom d-1"
    assert (seen["key"], seen["scope_type"], seen["scope_id"]) == (PROMPT_KEY, "bot", "caid")


def test_the_mode_setting_defaults_to_notify():
    definition = SETTING_DEFINITIONS[cn.MODE_SETTING]
    assert (definition.type, definition.default, definition.applies_to) == ("str", cn.NOTIFY, ())


@pytest.mark.parametrize("value,mode", [
    ("notify", cn.NOTIFY), (" Report ", cn.REPORT), ("OFF", cn.OFF), (None, cn.NOTIFY),
    ("bogus", cn.REPORT),                       # unknown: report only, never notify
    (RuntimeError("settings store down"), cn.REPORT),
])
def test_mode_parsing(monkeypatch, value, mode):
    _setting(monkeypatch, value)
    assert _pump().mode() == mode


def test_off_postgresql_the_pump_and_schema_do_nothing():
    sqlite = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    assert _pump(sqlite).run_pass() == cn.PassResult()
    assert _pump(None).run_pass() == cn.PassResult()
    cn.ensure_cancel_notice_schema(sqlite)  # no connection is ever opened
    cn.ensure_cancel_notice_schema(None)


@pytest.mark.parametrize("notified", [[], ["n1", "n2"]])
def test_the_leader_pass_announces_each_notice_and_wakes_only_when_there_are_some(notified):
    import asyncio

    from llm_bawt.service.leader_side_pumps import LeaderSidePumps

    woken, emitted = [], []
    dispatcher = SimpleNamespace(wake=lambda *args: woken.append(args))
    pump = SimpleNamespace(run_pass=lambda: cn.PassResult(notified=list(notified)))
    side = LeaderSidePumps(SimpleNamespace(_inter_bot_dispatcher=dispatcher),
                           SimpleNamespace(engine=None), cancel_notices=pump)

    async def emit(record):
        emitted.append(record)

    asyncio.run(side.run_pass(emit))
    assert emitted == notified
    assert len(woken) == (1 if notified else 0)


@pytest.mark.parametrize("failures,delay", [
    (0, cn.RETRY_BASE_S), (1, 2 * cn.RETRY_BASE_S), (4, 16 * cn.RETRY_BASE_S),
    (5, cn.RETRY_MAX_S), (1024, cn.RETRY_MAX_S), (10**6, cn.RETRY_MAX_S),  # 2**1024 overflowed a float
])
def test_the_backoff_doubles_to_its_cap_and_never_overflows(failures, delay):
    pump = cn.CancelNoticePump(SimpleNamespace(engine=None, config=None), clock=lambda: 100.0)
    if failures:
        pump._backoff["d-1"] = (failures, 0.0)
    assert pump._defer("d-1") == delay
    assert pump._backoff["d-1"] == (failures + 1, 100.0 + delay)
