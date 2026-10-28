"""A profile save that crosses runtime families starts a fresh thread.

Regression: Nova went agent -> chat -> agent while one thread stayed active,
so the agent turn resumed a days-old SDK transcript instead of starting over.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.service.routes import settings
from llm_bawt.service.routes.sessions import resolve_agent_session_key
from llm_bawt.service.runtime_switch_rotation import (
    RUNTIME_SWITCH_SOURCE,
    rotate_active_threads,
    runtime_family,
    runtime_family_changed,
)


def bot(agent_backend=None, harness="chat", slug="nova"):
    return SimpleNamespace(slug=slug, agent_backend=agent_backend, harness=harness)


CHAT = bot()
CLAUDE = bot("claude-code", "claude-code")
CLAUDE_PROXY = bot("claude-code", "claude-proxy")
CODEX = bot("codex", "codex")


@pytest.mark.parametrize(("before", "after", "expected"), [
    (CHAT, CLAUDE_PROXY, True),
    (CLAUDE_PROXY, CHAT, True),
    (CODEX, CLAUDE, True),
    (CLAUDE, CLAUDE_PROXY, False),  # same SDK family: transcripts resume across models
    (CHAT, CHAT, False),
    (None, CLAUDE, False),  # a create has no prior thread state to invalidate
])
def test_runtime_family_change_detection(before, after, expected):
    assert runtime_family_changed(before, after) is expected


def test_claude_pair_is_one_family():
    assert runtime_family(CLAUDE) == runtime_family(CLAUDE_PROXY) == "claude-code"
    assert runtime_family(CHAT) == "chat"


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP, session_metadata TEXT
        )"""))
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT)"))
        conn.execute(text("INSERT INTO bot_profiles (slug) VALUES ('nova'), ('caid')"))
    yield eng
    eng.dispose()


def insert(engine, id, *, bot_id="nova", user="nick", status="active", metadata=None):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sessions (id,bot_id,user_id,status,session_metadata) "
            "VALUES (:id,:bot,:user,:status,:meta)"
        ), dict(id=id, bot=bot_id, user=user, status=status, meta=json.dumps(metadata or {})))


def rows(engine, **where):
    clause = " AND ".join(f"{k}=:{k}" for k in where)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            text(f"SELECT * FROM sessions WHERE {clause}"), where,
        ).mappings()]


def test_rotation_drops_stale_sdk_key_for_every_user_of_that_bot_only(engine):
    stale = {"agent_session_keys": {"claude_code": "sdk-from-last-week"}}
    insert(engine, "nick-old", metadata=stale)
    insert(engine, "smoke-old", user="smoke")
    insert(engine, "archived", status="archived")
    insert(engine, "other-bot", bot_id="caid", metadata=stale)

    rotated = dict(rotate_active_threads(engine, "nova"))

    assert set(rotated) == {"nick", "smoke"}
    active = rows(engine, bot_id="nova", status="active")
    assert {r["id"] for r in active} == set(rotated.values())
    fresh = next(r for r in active if r["user_id"] == "nick")
    # The resolver the bridge binding uses finds nothing to resume -> cold start + seed.
    assert resolve_agent_session_key(json.loads(fresh["session_metadata"]), "claude-code") is None
    old = rows(engine, id="nick-old")[0]
    assert old["status"] == "archived"
    assert json.loads(old["session_metadata"])["reset_successor_id"] == rotated["nick"]
    assert rows(engine, id="other-bot")[0]["status"] == "active"


def point_to(engine, session_id):
    with engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET last_session_id=:id WHERE slug='nova'"),
                     {"id": session_id})


def pointer(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug='nova'")).scalar_one()


def test_runtime_switch_moves_only_pointed_owner_not_last_user(engine):
    insert(engine, "selected", user="a-owner")
    insert(engine, "other", user="z-owner")
    point_to(engine, "selected")
    successors = dict(rotate_active_threads(engine, "nova"))
    assert pointer(engine) == successors["a-owner"]
    assert pointer(engine) != successors["z-owner"]
    fresh = rows(engine, id=pointer(engine))[0]
    metadata = json.loads(fresh["session_metadata"])
    assert metadata["context_lifecycle"]["action"] == RUNTIME_SWITCH_SOURCE
    assert "agent_session_keys" not in metadata
    # A runtime switch is not a suppression request (WORKING_MEMORY.md 10.1).
    for successor in successors.values():
        assert json.loads(rows(engine, id=successor)[0]["session_metadata"])["working_memory"] == {"mode": "pinned"}


@pytest.mark.parametrize("selected", [None, "archived"])
def test_runtime_switch_never_guesses_null_or_archived_pointer(engine, selected):
    insert(engine, "active")
    insert(engine, "archived", status="archived")
    point_to(engine, selected)
    assert rotate_active_threads(engine, "nova")
    assert pointer(engine) == selected


def test_runtime_switch_pointer_and_all_rotations_rollback_together(engine, monkeypatch):
    from llm_bawt.memory.session_reset_lineage import SessionResetLineage
    insert(engine, "selected", user="a-owner")
    insert(engine, "other", user="z-owner")
    point_to(engine, "selected")
    rotate = SessionResetLineage.rotate
    def fail_second(engine, **kwargs):
        if kwargs["user_id"] == "z-owner":
            raise RuntimeError("second scope failed")
        return rotate(engine, **kwargs)
    monkeypatch.setattr(SessionResetLineage, "rotate", fail_second)
    with pytest.raises(RuntimeError, match="second scope failed"):
        rotate_active_threads(engine, "nova")
    assert pointer(engine) == "selected"
    assert {row["id"] for row in rows(engine, bot_id="nova", status="active")} == {"selected", "other"}
    assert all("reset_successor_id" not in json.loads(row["session_metadata"])
               for row in rows(engine, bot_id="nova"))


def _persist(monkeypatch, previous, profile, *, rotate=None):
    store = SimpleNamespace(engine=object(), get=lambda _slug: previous, upsert=lambda _p, *, guard: profile)
    subscriber = SimpleNamespace(
        publish_global_config_event=AsyncMock(return_value="1-0"),
        publish_tool_event=AsyncMock(return_value="2-0"),
    )
    service = SimpleNamespace(config=object(), _redis_subscriber=subscriber)
    calls = []
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: None)
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_a, **_k: None)
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response", lambda profile, settings, tags=None: profile)
    monkeypatch.setattr(settings, "rotate_active_threads", rotate or (
        lambda engine, slug: (calls.append(slug), [("nick", "fresh-thread")])[1]
    ))
    asyncio.run(settings._persist_bot_profile({"slug": "nova"}, create_only=False))
    return calls, subscriber


def test_chat_to_agent_save_rotates_and_tells_the_ui(monkeypatch):
    calls, subscriber = _persist(monkeypatch, CHAT, CLAUDE_PROXY)
    assert calls == ["nova"]
    args = subscriber.publish_tool_event.await_args.args
    assert args[:2] == ("nova", "nick")
    assert args[2]["_type"] == "thread_switched"
    assert args[2]["new_session_id"] == "fresh-thread"
    assert args[2]["source"] == RUNTIME_SWITCH_SOURCE


def test_same_family_save_keeps_the_thread(monkeypatch):
    calls, subscriber = _persist(monkeypatch, CLAUDE, CLAUDE_PROXY)
    assert calls == []
    subscriber.publish_tool_event.assert_not_awaited()


def test_rotation_failure_does_not_fail_the_committed_save(monkeypatch):
    def boom(_engine, _slug):
        raise RuntimeError("db hiccup")

    _calls, subscriber = _persist(monkeypatch, CLAUDE, CHAT, rotate=boom)
    subscriber.publish_global_config_event.assert_awaited_once()
    subscriber.publish_tool_event.assert_not_awaited()
