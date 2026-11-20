"""TASK-2395: agent history reads are bounded and /new reads history once."""
from types import SimpleNamespace

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service import core as service_core
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.routes.history_seed import build_context_seed
from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
from llm_bawt.utils.history import HistoryManager

BUDGET = 9000
SETTINGS = {"summary_count": 3, "history_reload_ttl_seconds": 2.0}


class Backend:
    def __init__(self):
        self.calls = []
        self.rows = [
            Message(role="summary", content="earlier", timestamp=1.0, db_id="s1"),
            Message(role="user", content="hi", timestamp=2.0, db_id="u1"),
            Message(role="assistant", content="hello", timestamp=3.0, db_id="a1"),
            Message(role="user", content="next", timestamp=4.0, db_id="u2"),
        ]

    def get_messages(self, since_minutes=None, *, user_id=None):
        self.calls.append(("full", since_minutes, user_id))
        return list(self.rows)

    def get_context_window(self, *, token_budget, summary_limit, user_id=None):
        self.calls.append(("window", token_budget, summary_limit, user_id))
        return list(self.rows)

    def rotate_session(self, **_kw):
        return "new-session"


def _bawt(kind="claude-code"):
    backend = Backend()
    bawt = object.__new__(ServiceLLMBawt)
    bawt.model_definition = {"type": kind}
    bawt.client = SimpleNamespace(model_alias="m")
    bawt.config = SimpleNamespace(
        resolve_context_budget=lambda _alias: (100000, 1000, BUDGET),
        HISTORY_RELOAD_TTL_SECONDS=2.0,
    )
    bawt._resolve_setting = lambda key, fallback=None: SETTINGS.get(key, fallback)
    bawt.resolved_model_alias = "m"
    bawt.config_resolver = SimpleNamespace(
        resolve_config_setting=lambda key: (_ for _ in ()).throw(KeyError(key)),
    )
    bawt.history_manager = HistoryManager(
        client=None, config=SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="sys"),
        db_backend=backend, bot_id="bot", user_id="user_a",
        settings_getter=lambda key, fallback: SETTINGS.get(key, fallback),
    )
    bawt._last_history_load_at = 0.0
    return bawt, backend


def _service():
    service = SimpleNamespace(
        config=SimpleNamespace(resolve_context_budget=lambda _alias: (100000, 1000, BUDGET)),
        _default_bot="bot",
        _publish_thread_switched=lambda *a, **k: None,
    )
    service._rotate_chat_session = (
        lambda llm, bot, **kw: BridgeSessionResetMixin._rotate_chat_session(service, llm, bot, **kw)
    )
    return service


def _new_command(bawt):
    """The /new order: seed read, rotation, then the turn's history refresh."""
    service = _service()
    seed = build_context_seed("bot", None, service, llm_bawt=bawt)
    assert service._rotate_chat_session(bawt, "bot", working_memory_mode="pinned") == "new-session"
    bawt._refresh_history_if_stale()
    return seed


def test_agent_turn_reads_a_bounded_window():
    bawt, backend = _bawt()
    bawt._refresh_history_if_stale()
    assert backend.calls == [("window", BUDGET, 3, "user_a")]


def test_agent_window_uses_history_tokens_when_below_budget(monkeypatch):
    monkeypatch.setitem(SETTINGS, "history_tokens", 1000)
    bawt, backend = _bawt()
    assert bawt.history_window() == (1000, 3)
    _new_command(bawt)
    assert backend.calls == [("window", 1000, 3, "user_a")]


def test_non_agent_turn_keeps_the_full_read():
    bawt, backend = _bawt(kind="openai")
    bawt._refresh_history_if_stale()
    assert backend.calls == [("full", None, "user_a")]
    assert bawt.history_window() is None


def test_unbudgeted_agent_model_keeps_the_full_read():
    bawt, backend = _bawt()
    bawt.config.resolve_context_budget = lambda _alias: (0, 0, 0)
    bawt._refresh_history_if_stale()
    assert backend.calls == [("full", None, "user_a")]


def test_new_reads_history_once_and_the_turn_sees_the_same_rows():
    bawt, backend = _bawt()
    seed = _new_command(bawt)
    assert backend.calls == [("window", BUDGET, 3, "user_a")]
    assert [m.db_id for m in bawt.history_manager.messages] == ["s1", "u1", "a1", "u2"]
    assert [m["source_id"] for m in seed["messages"]] == ["s1", "u1", "a1", "u2"]


def test_seed_read_is_reused_only_once():
    bawt, backend = _bawt()
    _new_command(bawt)
    bawt.invalidate_history_cache(keep_seed_history=True)
    bawt._refresh_history_if_stale()
    assert [c[0] for c in backend.calls] == ["window", "window"]


@pytest.mark.parametrize("invalidate", [
    lambda bawt: bawt.invalidate_history_cache(),  # e.g. a summary write
    lambda bawt: setattr(service_core, "SEED_HISTORY_REUSE_SECONDS", -1.0),
    lambda bawt: setattr(bawt.config, "resolve_context_budget", lambda _a: (1, 1, 123)),
])
def test_invalidation_expiry_or_window_change_reads_again(invalidate, monkeypatch):
    monkeypatch.setattr(service_core, "SEED_HISTORY_REUSE_SECONDS", 30.0)
    bawt, backend = _bawt()
    service = _service()
    build_context_seed("bot", None, service, llm_bawt=bawt)
    service._rotate_chat_session(bawt, "bot", working_memory_mode="pinned")
    invalidate(bawt)
    bawt._refresh_history_if_stale()
    assert len(backend.calls) == 2


def test_thread_scoped_seed_is_not_reused():
    bawt, backend = _bawt()
    backend.load_session_scoped = lambda **kw: (
        backend.calls.append(("scoped",)) or list(backend.rows[1:2])
    )
    service = _service()
    build_context_seed("bot", None, service, llm_bawt=bawt, session_id="old-thread")
    service._rotate_chat_session(bawt, "bot", working_memory_mode="pinned")
    bawt._refresh_history_if_stale()
    assert [c[0] for c in backend.calls] == ["scoped", "window"]
    assert len(bawt.history_manager.messages) == 4


def test_fresh_cache_within_ttl_does_not_read():
    bawt, backend = _bawt()
    bawt._refresh_history_if_stale()
    bawt._refresh_history_if_stale()
    assert len(backend.calls) == 1


@pytest.mark.parametrize("failure", ["raise", "empty"])
def test_failed_bounded_read_falls_back_to_the_full_read(failure):
    # Codex finding 1: a raised error, or the MCP path's swallowed-error [],
    # must not leave the turn with no history.
    bawt, backend = _bawt()

    def broken(**kw):
        backend.calls.append(("window", kw["token_budget"], kw["summary_limit"], kw["user_id"]))
        if failure == "raise":
            raise RuntimeError("bounded query failed")
        return []

    backend.get_context_window = broken
    bawt._refresh_history_if_stale()
    assert backend.calls == [
        ("window", BUDGET, 3, "user_a"), ("full", None, "user_a"),
    ]
    assert [m.db_id for m in bawt.history_manager.messages] == ["s1", "u1", "a1", "u2"]


def test_chat_bot_seed_preview_keeps_the_full_pool():
    # Codex finding 3: previewing a chat bot's seed must not replace its
    # shared cache with a seed-only slice.
    bawt, backend = _bawt(kind="openai")
    build_context_seed("bot", None, _service(), llm_bawt=bawt)
    assert backend.calls == [("full", None, "user_a")]
    assert getattr(bawt, "_seed_history", None) is None
