"""TASK-638 — per-thread SDK session keys are canonical everywhere.

Each durable thread records backend-specific SDK identity and model metadata in
``agent_session_keys`` + ``agent_session_key_models``. Every Claude/Codex turn
binds either the user-selected thread or the active thread; the retired bot
scalar is never read or written.

Coverage includes model-agnostic resolution, Redis field parsing, explicit +
active binding, first-turn thread creation, `/new` rotation/seed behavior, request-local
wire plumbing, MCP adapter parity, and live-DB merge semantics.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.service.routes.sessions import (
    agent_key_name,
    resolve_agent_session_key,
)


# ──────────────────────────────────────────────────────────────────────────
# Resolver (pure)
# ──────────────────────────────────────────────────────────────────────────
class TestAgentKeyName:
    def test_dash_to_underscore(self):
        assert agent_key_name("claude-code") == "claude_code"

    def test_plain_passthrough(self):
        assert agent_key_name("codex") == "codex"

    def test_empty(self):
        assert agent_key_name("") == ""


class TestResolveAgentSessionKey:
    def test_canonical_hit(self):
        meta = {"agent_session_keys": {"claude_code": "sid-123"}}
        assert resolve_agent_session_key(meta, "claude-code") == "sid-123"

    def test_legacy_mirror_keys_no_longer_fall_back(self):
        # TASK-638: legacy provider/provider_session_id fallback removed.
        meta = {"provider": "claude-code", "provider_session_id": "sid-legacy"}
        assert resolve_agent_session_key(meta, "claude-code") is None

    def test_routing_key_guard(self):
        meta = {"agent_session_keys": {"claude_code": "byte:nick"}}
        assert resolve_agent_session_key(meta, "claude-code") is None

    def test_claude_model_metadata_does_not_block_resume(self):
        meta = {
            "agent_session_keys": {"claude_code": "sid-1"},
            "agent_session_key_models": {"claude_code": "model-a"},
        }
        assert (
            resolve_agent_session_key(meta, "claude-code", "model-b") == "sid-1"
        )

    def test_other_backend_model_mismatch_still_blocks_resume(self):
        meta = {
            "agent_session_keys": {"codex": "sid-cx"},
            "agent_session_key_models": {"codex": "codex-model"},
        }
        assert resolve_agent_session_key(meta, "codex", "other-model") is None

    def test_legacy_model_scalar_only_applies_to_matching_provider(self):
        meta = {
            "agent_session_keys": {"claude_code": "sid-cc", "codex": "sid-cx"},
            "provider": "codex",
            "provider_session_model": "codex-model",
        }
        assert (
            resolve_agent_session_key(meta, "claude-code", "claude-model")
            == "sid-cc"
        )
        assert resolve_agent_session_key(meta, "codex", "other-model") is None

    def test_no_stored_model_passes(self):
        meta = {"agent_session_keys": {"claude_code": "sid-1"}}
        assert resolve_agent_session_key(meta, "claude-code", "model-b") == "sid-1"

    def test_empty_meta(self):
        assert resolve_agent_session_key({}, "claude-code") is None
        assert resolve_agent_session_key(None, "claude-code") is None


# ──────────────────────────────────────────────────────────────────────────
# Bridge SendRequest parsing
# ──────────────────────────────────────────────────────────────────────────
class TestSendRequestThreadFields:
    def _base_fields(self, **extra):
        return {
            "request_id": "req_x",
            "session_key": "byte:nick",
            "message": "hi",
            "model": "m1",
            **extra,
        }

    def test_absent_fields_default_none(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(self._base_fields())
        assert req.thread_session_id is None
        assert req.thread_resume_id is None
        assert req.task_turn_capability is None

    def test_thread_fields_parsed(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(
            self._base_fields(
                thread_session_id="thread-1", thread_resume_id="sid-9",
            )
        )
        assert req.thread_session_id == "thread-1"
        assert req.thread_resume_id == "sid-9"

    def test_task_turn_capability_parsed_as_opaque_value(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(
            self._base_fields(task_turn_capability="opaque-capability")
        )
        assert req.task_turn_capability == "opaque-capability"

    def test_routing_key_resume_dropped(self):
        from claude_code_bridge.send_request import SendRequest

        req = SendRequest.from_fields(
            self._base_fields(
                thread_session_id="thread-1", thread_resume_id="byte:nick",
            )
        )
        assert req.thread_session_id == "thread-1"
        assert req.thread_resume_id is None


# ──────────────────────────────────────────────────────────────────────────
# Claude /new preprocessing — one fresh SDK session, never two
# ──────────────────────────────────────────────────────────────────────────
class TestClaudeNewPreprocessing:
    @staticmethod
    def _harness():
        from claude_code_bridge.session_ops import ClaudeSessionMixin

        class _Publisher:
            def __init__(self):
                self.done = []

            def publish_run_done(self, request_id):
                self.done.append(request_id)

        class _Harness(ClaudeSessionMixin):
            def __init__(self):
                self.resets = []
                self.seeds = []
                self.events = []
                self._publisher = _Publisher()

            def _publish_session_reset_unified(self, *args, **kwargs):
                self.resets.append((args, kwargs))

            async def _seed_new_session(self, *args, **kwargs):
                self.seeds.append((args, kwargs))
                return {
                    "seeded": True,
                    "session_id": "sdk-new",
                    "approx_tokens": 1_234,
                }

            @staticmethod
            def _format_seed_ack(seed_stats):
                return f"seeded:{seed_stats['session_id']}"

            def _publish_event(self, *args, **kwargs):
                self.events.append((args, kwargs))

        class _Redis:
            def __init__(self):
                self.acks = []

            async def xack(self, *args):
                self.acks.append(args)

        return _Harness(), _Redis()

    @staticmethod
    def _run(harness, redis, message, *, explicit_thread=False):
        return asyncio.run(harness._preprocess_new_command(
            message,
            explicit_thread=explicit_thread,
            bot_slug="snark",
            session_key="snark:nick",
            request_id="req-1",
            model="model-1",
            context_window=372_000,
            inject_messages=[{"role": "summary", "content": "prior"}],
            thread_session_id="thread-new",
            msg_id="redis-1",
            async_redis=redis,
        ))

    def test_clean_reset_reports_zero_context_instead_of_stale_usage(self):
        from claude_code_bridge.session_ops import ClaudeSessionMixin

        assert ClaudeSessionMixin._seed_usage_estimate(
            {"seeded": False, "clean_start": True, "approx_tokens": 0},
            context_window=200_000,
        ) == {
            "input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 0,
            "resident_tokens": 0,
            "resident_source": "clean_session_reset",
            "context_window": 200_000,
            "total_cost_usd": 0.0,
            "usage_status": "reset",
        }

    def test_new_with_message_defers_the_only_seed_to_cold_start(self):
        harness, redis = self._harness()

        remaining = self._run(harness, redis, "/new continue here")

        assert remaining == "continue here"
        assert len(harness.resets) == 1
        assert harness.seeds == []
        assert harness.events == []
        assert harness._publisher.done == []
        assert redis.acks == []

    def test_bare_new_seeds_once_and_finishes(self):
        harness, redis = self._harness()

        remaining = self._run(harness, redis, "  /new")

        assert remaining is None
        assert len(harness.resets) == 1
        assert len(harness.seeds) == 1
        assert harness.seeds[0][1]["thread_session_id"] == "thread-new"
        assert len(harness.events) == 1
        assert harness.events[0][1]["token_usage"] == {
            "input_tokens": 1_234,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 0,
            "resident_tokens": 1_234,
            "resident_source": "session_seed_char_estimate",
            "context_window": 372_000,
            "total_cost_usd": 0.0,
            "usage_status": "estimated",
        }
        # TASK-1233: the send's admission publishes the run's done, after
        # preprocessing returns; preprocessing never does.
        assert harness._publisher.done == []
        assert redis.acks == [("agent:commands", "claude-code-bridge", "redis-1")]

    def test_explicit_thread_treats_new_as_literal_text(self):
        harness, redis = self._harness()

        remaining = self._run(
            harness, redis, "/new do not rotate this old thread",
            explicit_thread=True,
        )

        assert remaining == "/new do not rotate this old thread"
        assert harness.resets == []
        assert harness.seeds == []
        assert redis.acks == []


# ──────────────────────────────────────────────────────────────────────────
# Dispatch binding (_bind_agent_thread) — hermetic fakes
# ──────────────────────────────────────────────────────────────────────────
from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin


class _Bridge(ChatStreamingBridgeMixin):
    pass


def _fake_llm_bawt(
    backend: str = "claude-code",
    bot_config: dict | None = None,
    session_row: dict | None = None,
    *,
    active_row: dict | None = None,
    created_session_id: str = "fresh-thread",
):
    bc = bot_config if bot_config is not None else {}
    db_backend = SimpleNamespace(
        get_session=lambda sid: session_row,
        get_active_session=lambda **kwargs: active_row,
        get_or_create_active_session=lambda **kwargs: created_session_id,
    )
    return SimpleNamespace(
        bot=SimpleNamespace(agent_backend=backend, slug="byte"),
        user_id="nick",
        client=SimpleNamespace(_bot_config=bc),
        history_manager=SimpleNamespace(_db_backend=db_backend),
    )


class TestBindAgentThread:
    def test_unscoped_turn_returns_none(self):
        lb = _fake_llm_bawt()
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id=None))
        assert out is None

    def test_non_claude_code_never_binds(self):
        lb = _fake_llm_bawt(backend="openclaw")
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out is None

    def test_scoped_turn_binds_thread_and_resume(self):
        row = {
            "session_metadata": {"agent_session_keys": {"claude_code": "sid-42"}}
        }
        lb = _fake_llm_bawt(session_row=row)
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1", "thread_resume_id": "sid-42", "explicit_thread": True}

    def test_scoped_turn_without_stored_key_binds_thread_only(self):
        lb = _fake_llm_bawt(session_row={"session_metadata": {}})
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1", "explicit_thread": True}

    def test_model_mismatch_resumes_same_transcript(self):
        row = {
            "session_metadata": {
                "agent_session_keys": {"claude_code": "sid-42"},
                "agent_session_key_models": {"claude_code": "old-model"},
            }
        }
        lb = _fake_llm_bawt(bot_config={"model": "new-model"}, session_row=row)
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {
            "thread_session_id": "t-1",
            "thread_resume_id": "sid-42",
            "explicit_thread": True,
        }

    def test_binding_is_request_local_not_instance_state(self):
        # Gavel review finding 1: the binding must never be written to the
        # shared cached client config — concurrent turns would cross-bind.
        row = {
            "session_metadata": {"agent_session_keys": {"claude_code": "sid-42"}}
        }
        bc = {}
        lb = _fake_llm_bawt(bot_config=bc, session_row=row)
        _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert bc == {}  # untouched — binding travels by value

    def test_never_raises_on_broken_backend(self):
        lb = SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code"),
            client=SimpleNamespace(_bot_config={}),
            history_manager=SimpleNamespace(_db_backend=None),
        )
        out = _Bridge()._bind_agent_thread(lb, SimpleNamespace(session_id="t-1"))
        assert out == {"thread_session_id": "t-1", "explicit_thread": True}


class TestMCPShortTermSessionContract:
    def test_active_session_methods_delegate_to_bound_memory_client(self):
        from llm_bawt.mcp_server.client import _MCPShortTermManager

        calls = []
        memory = SimpleNamespace(
            get_active_session=lambda: calls.append("get") or {"id": "active-1"},
            get_or_create_active_session=lambda: calls.append("create") or "active-1",
        )
        manager = _MCPShortTermManager(memory)

        assert manager.get_active_session(bot_id="byte", user_id="nick") == {
            "id": "active-1",
        }
        assert manager.get_or_create_active_session(
            bot_id="byte", user_id="nick",
        ) == "active-1"
        assert calls == ["get", "create"]


class TestResolveActiveThreadBinding:
    def test_existing_active_thread_resumes_its_backend_key(self):
        row = {
            "id": "active-thread",
            "session_metadata": {
                "agent_session_keys": {"claude_code": "sdk-active"},
            },
        }
        lb = _fake_llm_bawt(session_row=row, active_row=row)

        assert _Bridge()._resolve_active_thread_binding(lb) == {
            "thread_session_id": "active-thread",
            "thread_resume_id": "sdk-active",
        }

    def test_first_turn_creates_thread_for_sdk_writeback(self):
        lb = _fake_llm_bawt(active_row=None, created_session_id="thread-first")

        assert _Bridge()._resolve_active_thread_binding(lb) == {
            "thread_session_id": "thread-first",
        }


class TestRotationScopedGuard:
    def test_explicit_thread_never_rotates(self):
        lb = _fake_llm_bawt()
        # _rotate_chat_session would need a real backend; the guard must
        # return False BEFORE reaching it.
        assert (
            _Bridge()._maybe_rotate_agent_session(
                lb, "byte", "/new",
                thread_binding={"thread_session_id": "t-1", "explicit_thread": True},
            )
            is False
        )

    def test_active_thread_new_still_reaches_rotation(self):
        lb = _fake_llm_bawt()
        called = []

        b = _Bridge()
        b._rotate_chat_session = lambda *a, **k: called.append(1) or True
        assert b._maybe_rotate_agent_session(
            lb, "byte", "/new", thread_binding={"thread_session_id": "active"},
        ) is True
        assert called


# ──────────────────────────────────────────────────────────────────────────
# Seed decision (maybe_build_session_seed scoped branch) — hermetic
# ──────────────────────────────────────────────────────────────────────────
class TestScopedSeedDecision:
    def _llm_bawt(self, scope="inline+summaries"):
        return SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code", agent_backend_config={}),
            client=SimpleNamespace(_bot_config={}),
            config_resolver=SimpleNamespace(
                resolve_config_setting=lambda key: SimpleNamespace(value=scope),
            ),
        )

    def test_thread_with_stored_key_gets_no_seed(self):
        from llm_bawt.service.routes.history import maybe_build_session_seed

        assert (
            maybe_build_session_seed(
                self._llm_bawt(), "byte", "m", "hello", None,
                thread_binding={
                    "thread_session_id": "t-1",
                    "thread_resume_id": "sid-1",
                    "explicit_thread": True,
                },
            )
            is None
        )

    def test_thread_without_key_builds_scoped_seed(self, monkeypatch):
        from llm_bawt.service.routes import history_seed

        captured = {}

        def _fake_seed(bot_id, model, service, session_id=None, *, llm_bawt=None):
            captured["session_id"] = session_id
            captured["owner"] = llm_bawt
            return {"messages": [{"role": "user", "content": "x"}]}

        monkeypatch.setattr(history_seed, "build_context_seed", _fake_seed)
        owner = self._llm_bawt()
        out = history_seed.maybe_build_session_seed(
            owner, "byte", "m", "hello", None,
            thread_binding={"thread_session_id": "t-1", "explicit_thread": True},
        )
        assert out == [{"role": "user", "content": "x"}]
        assert captured["session_id"] == "t-1"
        assert captured["owner"] is owner

    @pytest.mark.parametrize("prompt", ["hello", "/new"])
    def test_cold_active_thread_uses_preview_history_pool(self, monkeypatch, prompt):
        """Normal cold starts and /new must match the unscoped seed preview.

        The active thread id is still the destination for SDK-key writeback, but
        it must not narrow context assembly to that one durable thread. Doing so
        makes back-to-back /new carry only the previous reset exchange.
        """
        from llm_bawt.service.routes import history_seed

        captured = {}

        def _fake_seed(bot_id, model, service, session_id=None, *, llm_bawt=None, history_scope=None):
            captured["session_id"] = session_id
            captured["owner"] = llm_bawt
            captured["history_scope"] = history_scope
            return {"messages": [{"role": "summary", "content": "prior"}]}

        monkeypatch.setattr(history_seed, "build_context_seed", _fake_seed)
        owner = self._llm_bawt()
        out = history_seed.maybe_build_session_seed(
            owner, "byte", "m", prompt, None,
            thread_binding={"thread_session_id": "active-1"},
            history_scope="summaries",
        )
        assert out == [{"role": "summary", "content": "prior"}]
        assert captured["session_id"] is None
        assert captured["owner"] is owner
        assert captured["history_scope"] == "summaries"

    def test_warm_active_thread_does_not_double_seed(self):
        from llm_bawt.service.routes.history import maybe_build_session_seed

        assert maybe_build_session_seed(
            self._llm_bawt(), "byte", "m", "hello", None,
            thread_binding={
                "thread_session_id": "active-1",
                "thread_resume_id": "sdk-active",
            },
        ) is None

    def test_unbound_turn_ignores_scoped_branch(self):
        from llm_bawt.service.routes.history import maybe_build_session_seed

        # No binding, continuity resolution blows up on the None resolver →
        # helper swallows and returns None (never raises).
        lb = SimpleNamespace(
            bot=SimpleNamespace(agent_backend="claude-code", agent_backend_config={}),
            client=SimpleNamespace(_bot_config={}),
            config_resolver=None,
        )
        assert maybe_build_session_seed(lb, "byte", "m", "hello", None) is None


# ──────────────────────────────────────────────────────────────────────────
# Kwarg channel: binding reaches the backend config per-call, never shared
# ──────────────────────────────────────────────────────────────────────────
class TestKwargChannelRequestLocal:
    def test_stream_raw_merges_binding_into_call_config_only(self):
        from llm_bawt.clients.agent_backend_client import AgentBackendClient
        from llm_bawt.models.message import Message

        captured: dict = {}

        class _Backend:
            def stream_raw(self, prompt, config, **kw):
                captured.update(config)
                yield "ok"

        client = AgentBackendClient.__new__(AgentBackendClient)
        client._bot_config = {"bot_id": "byte"}
        client._backend = _Backend()

        out = list(
            client.stream_raw(
                [Message(role="user", content="hi")],
                thread_binding={
                    "thread_session_id": "t-1",
                    "thread_resume_id": "s-1",
                },
            )
        )
        assert out == ["ok"]
        assert captured["thread_session_id"] == "t-1"
        assert captured["thread_resume_id"] == "s-1"
        # Shared instance config untouched — the binding was per-call.
        assert client._bot_config == {"bot_id": "byte"}


# ──────────────────────────────────────────────────────────────────────────
# Live-DB: endpoint merge semantics + resolver round-trip
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="class")
def live_thread(private_app_pg):
    """Throwaway bot + one real thread row for endpoint tests."""
    from sqlalchemy import text
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

    bot = f"_t252_pytest_{uuid.uuid4().hex[:8]}"
    user = "nick"
    from tests.pg_isolation import require_test_db_config
    config = require_test_db_config()
    manager = PostgreSQLShortTermManager(config, bot_id=bot, user_id=user)
    thread_id = manager.get_or_create_active_session(bot_id=bot, user_id=user)

    yield SimpleNamespace(bot=bot, user=user, manager=manager, thread_id=thread_id)

    from llm_bawt.memory.postgresql import PARENT_TABLES, partition_name
    with manager._backend.engine.begin() as conn:
        for parent in PARENT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
        conn.execute(text("DELETE FROM sessions WHERE bot_id = :b"), {"b": bot})


@pytest.mark.integration
class TestAgentKeyEndpointLiveDB:
    def _put(self, env, backend="claude-code", key="sid-live-1", model="m1"):
        from llm_bawt.service.routes.sessions import (
            AgentSessionKeyRequest,
            put_agent_session_key,
        )

        return asyncio.run(
            put_agent_session_key(
                env.thread_id,
                AgentSessionKeyRequest(
                    backend=backend, session_key=key, model=model
                ),
                bot_id=env.bot,
            )
        )

    def test_put_stores_canonical_key_and_model(self, live_thread):
        out = self._put(live_thread)
        assert out["stored"] is True
        row = live_thread.manager.get_session(live_thread.thread_id)
        meta = row["session_metadata"]
        assert meta["agent_session_keys"]["claude_code"] == "sid-live-1"
        assert meta["agent_session_key_models"]["claude_code"] == "m1"
        assert "provider_session_id" not in meta
        assert "provider_session_model" not in meta

    def test_resolver_round_trip(self, live_thread):
        self._put(live_thread, key="sid-live-2")
        row = live_thread.manager.get_session(live_thread.thread_id)
        assert (
            resolve_agent_session_key(row["session_metadata"], "claude-code")
            == "sid-live-2"
        )

    def test_second_backend_merges_keys_and_models(self, live_thread):
        self._put(live_thread, key="sid-cc", model="claude-model")
        self._put(
            live_thread, backend="codex", key="sid-cx", model="codex-model"
        )
        row = live_thread.manager.get_session(live_thread.thread_id)
        meta = row["session_metadata"]
        assert meta["agent_session_keys"] == {
            "claude_code": "sid-cc",
            "codex": "sid-cx",
        }
        assert meta["agent_session_key_models"] == {
            "claude_code": "claude-model",
            "codex": "codex-model",
        }
        assert (
            resolve_agent_session_key(meta, "claude-code", "claude-model")
            == "sid-cc"
        )
        assert resolve_agent_session_key(meta, "codex", "codex-model") == "sid-cx"

    def test_replacing_key_without_model_clears_old_model_gate(self, live_thread):
        self._put(live_thread, key="sid-modeled", model="model-a")
        self._put(live_thread, key="sid-unmodeled", model="")
        row = live_thread.manager.get_session(live_thread.thread_id)
        meta = row["session_metadata"]
        assert meta["agent_session_keys"]["claude_code"] == "sid-unmodeled"
        assert "claude_code" not in meta["agent_session_key_models"]
        assert (
            resolve_agent_session_key(meta, "claude-code", "other-model")
            == "sid-unmodeled"
        )

    def test_routing_key_rejected(self, live_thread):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            self._put(live_thread, key="byte:nick")
        assert exc.value.status_code == 422

    def test_legacy_completed_rows_migrate_idempotently(self, live_thread):
        """TASK-250 review finding: bootstrap must retire 'completed' and
        backfill archived_at from ended_at, idempotently."""
        from sqlalchemy import text

        legacy_id = str(uuid.uuid4())
        engine = live_thread.manager._backend.engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sessions (id, bot_id, user_id, started_at, "
                    "ended_at, status) VALUES (:id, :b, 'nick', "
                    "NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', "
                    "'completed')"
                ),
                {"id": legacy_id, "b": live_thread.bot},
            )
            # The exact statements the bootstrap runs (postgresql.py /
            # migrations_memory.py — kept in sync by this test).
            for stmt in (
                "UPDATE sessions SET status='archived', "
                "archived_at=COALESCE(archived_at, ended_at) "
                "WHERE status='completed'",
                "UPDATE sessions SET archived_at=ended_at "
                "WHERE status='archived' AND archived_at IS NULL "
                "AND ended_at IS NOT NULL",
            ):
                conn.execute(text(stmt))
        row = live_thread.manager.get_session(legacy_id)
        assert row["status"] == "archived"
        assert row["archived_at"] is not None
        first_archived = row["archived_at"]
        # Idempotency: re-running touches nothing.
        with engine.begin() as conn:
            r1 = conn.execute(
                text("UPDATE sessions SET status='archived', "
                     "archived_at=COALESCE(archived_at, ended_at) "
                     "WHERE status='completed'")
            )
            r2 = conn.execute(
                text("UPDATE sessions SET archived_at=ended_at "
                     "WHERE status='archived' AND archived_at IS NULL "
                     "AND ended_at IS NOT NULL")
            )
            assert (r1.rowcount or 0) == 0
        assert live_thread.manager.get_session(legacy_id)["archived_at"] == first_archived

    def test_backfill_insert_stamps_archived_at(self, live_thread):
        """TASK-250 review finding: backfill_sessions' INSERT must stamp
        archived_at = inferred ended_at (shape-level check of the SQL)."""
        import inspect
        from llm_bawt.memory import migrations_memory

        src = inspect.getsource(migrations_memory)
        # The historical-session INSERT carries archived_at, valued from the
        # same inferred ended_at.
        assert "ended_at, archived_at, status" in src
        assert ":ended_at, :ended_at,\n                         'archived'" in src

    def test_unknown_thread_404(self, live_thread):
        from fastapi import HTTPException
        from llm_bawt.service.routes.sessions import (
            AgentSessionKeyRequest,
            put_agent_session_key,
        )

        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                put_agent_session_key(
                    str(uuid.uuid4()),
                    AgentSessionKeyRequest(
                        backend="claude-code", session_key="sid-x", model=None
                    ),
                    bot_id=live_thread.bot,
                )
            )
        assert exc.value.status_code == 404
