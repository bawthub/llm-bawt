"""/v1/bots alias field fails closed when aliases cannot be read (TASK-1900)."""
from __future__ import annotations

from types import SimpleNamespace

from llm_bawt.service import dependencies
from llm_bawt.service.routes import models


def test_unreadable_aliases_are_null_not_empty(monkeypatch):
    def broken(_config):
        raise RuntimeError("fixture: profile DB down")

    monkeypatch.setattr(dependencies, "get_bot_profile_store", broken)
    assert models._read_slug_aliases(None) is None


def test_missing_engine_is_null(monkeypatch):
    monkeypatch.setattr(dependencies, "get_bot_profile_store", lambda _c: SimpleNamespace(engine=None))
    assert models._read_slug_aliases(None) is None
