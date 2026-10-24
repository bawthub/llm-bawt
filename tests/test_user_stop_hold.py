"""Unit tests for TASK-1327's user-Stop queue hold: who gets one, and for how long."""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

import pytest

from llm_bawt import runtime_setting_resolution
from llm_bawt.service.routes.chat import ChatAbortRequest
from llm_bawt.service.user_stop_hold import HOLD_SETTING, USER_STOP_SOURCES, snapshot, stop_hold_ttl
from llm_bawt.setting_definitions import SETTING_DEFINITIONS

PG = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
SQLITE = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))


def _setting(monkeypatch, value):
    def resolve(config, key, fallback=None):
        assert key == HOLD_SETTING
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runtime_setting_resolution, "resolve_global_runtime_setting", resolve)


def test_user_stop_sources_are_every_human_stop_source():
    sources = set(get_args(ChatAbortRequest.model_fields["source"].annotation))
    assert USER_STOP_SOURCES == sources - {"unknown"}


def test_the_setting_defaults_to_ten_minutes():
    assert SETTING_DEFINITIONS[HOLD_SETTING].default == 600


@pytest.mark.parametrize("source", sorted(USER_STOP_SOURCES))
def test_a_user_stop_on_postgresql_holds_for_the_setting(monkeypatch, source):
    _setting(monkeypatch, 600)
    assert stop_hold_ttl(PG, None, source) == 600


@pytest.mark.parametrize("engine,source,value", [
    (PG, "unknown", 600),           # not a human Stop
    (SQLITE, "chat_stop", 600),     # gates are PostgreSQL-only
    (None, "chat_stop", 600),
    (PG, "chat_stop", 0),           # 0 disables the hold
    (PG, "chat_stop", None),
    (PG, "chat_stop", RuntimeError("settings store down")),  # Stop never fails on the hold
])
def test_no_hold(monkeypatch, engine, source, value):
    _setting(monkeypatch, value)
    assert stop_hold_ttl(engine, None, source) is None


def test_snapshot_off_postgresql_is_none():
    assert snapshot(SQLITE, "bot") is None
