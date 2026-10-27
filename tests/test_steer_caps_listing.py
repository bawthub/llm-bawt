"""Steer capabilities on the bot and turn-log listings (TASK-1243 P4).

BawtHub reads these to choose between steering a running turn and queueing a
message until it ends. ``/v1/bots`` carries what a run started now would bind;
``/v1/turn-logs`` carries each in-flight run's own pinned value.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from llm_bawt.service.routes import models as models_route
from llm_bawt.service.routes import turn_logs as turn_logs_route

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
ALL = ["cancel_queued", "escalate", "next", "now"]


def _row(turn_id, *, bot_id="worker", backend=None, caps=None, ended=False):
    return SimpleNamespace(
        id=turn_id, created_at=NOW, request_id=None, path="/v1/chat/completions",
        stream=True, model=None, bot_id=bot_id, user_id="user", status="streaming",
        latency_ms=None, user_prompt="hi", response_text="", error_text=None,
        ended_at=NOW if ended else None, agent_backend=backend, agent_steer_caps=caps,
    )


class _Store:
    engine = object()

    def __init__(self, rows):
        self.rows = rows

    def list_turns(self, **_kwargs):
        return self.rows, len(self.rows)

    def changed_files_summaries_for_turns(self, _ids):
        return {}


def _list(monkeypatch, rows, bots=None, *, fail=False):
    lookups: list[str] = []

    class _Bots:
        def __init__(self, _config):
            if fail:
                raise RuntimeError("bot registry down")

        def get_bot(self, slug):
            lookups.append(slug)
            return (bots or {}).get(slug)

    monkeypatch.setattr(turn_logs_route, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(turn_logs_route, "get_turn_log_store", lambda: _Store(rows))
    monkeypatch.setattr(turn_logs_route, "_live_tool_call_counts", lambda _store, _ids: {})
    monkeypatch.setattr("llm_bawt.bots.BotManager", _Bots)
    response = turn_logs_route.list_turn_logs(
        bot_id=None, user_id=None, model=None, request_id=None, status=None,
        active_only=True, stream=None, has_tools=None, since_hours=168, limit=100, offset=0,
    )
    return {item.id: item.steer_caps for item in response.turns}, lookups


def test_in_flight_rows_carry_their_runs_pinned_caps(monkeypatch):
    caps, lookups = _list(monkeypatch, [
        _row("codex-app", backend="codex", caps="next"),
        _row("codex-exec", backend="codex", caps=""),
        _row("legacy-claude", backend="claude-code"),
        _row("legacy-codex", backend="codex"),
        _row("ended", backend="claude-code", caps="next,now", ended=True),
    ])
    assert caps == {
        "codex-app": ["next"],
        "codex-exec": [],
        "legacy-claude": ALL,
        "legacy-codex": [],
        "ended": None,
    }
    # A bound row never consults the bot's current config.
    assert lookups == []


def test_an_unbound_row_stands_in_with_its_bots_caps_read_once(monkeypatch):
    bots = {"worker": SimpleNamespace(
        agent_backend="codex", agent_backend_config={"codex_transport": "app_server"},
    )}
    caps, lookups = _list(monkeypatch, [_row("a"), _row("b"), _row("gone", bot_id="missing")], bots)
    assert caps == {"a": ["next"], "b": ["next"], "gone": []}
    assert lookups == ["worker", "missing"]


def test_a_bot_lookup_failure_lists_the_row_with_no_caps(monkeypatch):
    caps, _ = _list(monkeypatch, [_row("a"), _row("bound", backend="codex", caps="next")], fail=True)
    assert caps == {"a": [], "bound": ["next"]}


def _bot(slug, backend, config):
    return SimpleNamespace(
        slug=slug, name=slug.title(), description=None, system_prompt="", prompt_override_id=None,
        requires_memory=True, voice_optimized=False, tts_mode=False, include_summaries=True,
        include_in_global_search=True, tags=[], default_voice=None, uses_tools=False,
        uses_search=False, uses_home_assistant=False, default_model=None, color=None,
        avatar=None, avatar_render=None, bot_type="agent", harness=None, endpoint_id=None,
        agent_backend=backend, agent_backend_config=config, settings={},
    )


def test_bot_listing_carries_what_a_new_run_would_bind(monkeypatch):
    bots = [
        _bot("one", "claude-code", {"model": "m"}),
        _bot("two", "codex", {"codex_transport": "app_server"}),
        _bot("three", "codex", {"codex_transport": "exec"}),
        _bot("four", None, {}),
    ]
    monkeypatch.setattr(models_route, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(models_route, "BotManager", lambda _config: SimpleNamespace(list_bots=lambda: bots))
    listed = {bot.slug: bot.steer_capabilities for bot in models_route.list_bots().data}
    assert listed == {"one": ALL, "two": ["next"], "three": [], "four": []}
