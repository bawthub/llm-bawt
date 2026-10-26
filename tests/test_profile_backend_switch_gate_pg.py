"""PostgreSQL tests for the TASK-1233 S3 profile backend-switch guard.

The profile route's real ``_persist_bot_profile`` runs against the real
``BotProfileStore`` and real system gates; only payload validation (model
endpoint lookups, covered elsewhere) and the post-commit side effects
(registry reload, cache drops, events, thread rotation) are stubbed. Each test
holds a backend name no other bot uses, so parallel PG tests are never held.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service.routes import settings
from llm_bawt.service.turn_admission import activate_hold, lock_target
from llm_bawt.service.turn_gates import GateActor, GateScope, GateSpec, ensure_gate_schema, gate_blocks
from tests.pg_lock_wait import wait_blocked_by

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(private_app_pg, monkeypatch):
    store = BotProfileStore(private_app_pg)
    assert store.engine is not None
    ensure_gate_schema(store.engine)
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(store=store, engine=store.engine, held=f"held-{tag}", free=f"free-{tag}",
                          owner=f"job-switch-{tag}", bots=[])

    service = SimpleNamespace(config=private_app_pg, _redis_subscriber=SimpleNamespace(
        publish_global_config_event=AsyncMock(return_value="1-0"),
        publish_tool_event=AsyncMock(return_value="2-0")))
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: None)
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_a, **_k: None)
    monkeypatch.setattr(settings, "_resolve_effective_bot_prompt", lambda *_args: "")
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response", lambda profile, settings, tags=None: profile)
    monkeypatch.setattr(settings, "rotate_active_threads", lambda engine, slug: [])

    def bot(backend=None):
        slug = f"test-switch-{tag}-{len(ctx.bots)}"
        ctx.bots.append(slug)
        store.upsert({"slug": slug, "agent_backend": backend})
        return slug

    def save(payload, *, create_only=False):
        return asyncio.run(settings._persist_bot_profile(dict(payload), create_only=create_only))

    def hold():
        spec = GateSpec(owner_kind="job", owner_id=ctx.owner, mode="hold",
                        scope=GateScope.on_backends(ctx.held), ttl_seconds=600,
                        reason="test switch gate", allow_actors=frozenset())
        with ctx.engine.begin() as conn:
            return activate_hold(conn, spec).gate.id

    def runtime(slug):
        row = store.get(slug)
        return None if row is None else (row.bot_type, row.agent_backend)

    def blocked(slug):
        with ctx.engine.begin() as conn:
            lock_target(conn, slug)
            return gate_blocks(conn, slug, GateActor("reserve")) is not None

    def members(gate_id):
        with ctx.engine.connect() as conn:
            return conn.execute(text(
                "SELECT turn_id, bot_id, disposition FROM gate_members WHERE gate_id=:g ORDER BY id"
            ), {"g": gate_id}).all()

    ctx.bot, ctx.save, ctx.hold, ctx.runtime, ctx.blocked, ctx.members = (
        bot, save, hold, runtime, blocked, members)
    yield ctx
    with ctx.engine.begin() as conn:
        conn.execute(text("DELETE FROM gate_members WHERE gate_id IN"
                          " (SELECT id FROM system_gates WHERE owner_id=:o)"), {"o": ctx.owner})
        conn.execute(text("DELETE FROM system_gates WHERE owner_id=:o"), {"o": ctx.owner})
        for slug in ctx.bots:
            conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:b"), {"b": slug})
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})


def _refused(rig, payload, **kwargs) -> HTTPException:
    with pytest.raises(HTTPException) as caught:
        rig.save(payload, **kwargs)
    return caught.value


def test_switch_into_a_held_backend_is_refused_and_writes_nothing(rig):
    agent = rig.bot(rig.free)
    chat = rig.bot()
    rig.hold()

    refused = _refused(rig, {"slug": agent, "agent_backend": rig.held, "description": "moved"})
    assert refused.status_code == 409 and refused.detail.startswith("system_held")
    assert rig.runtime(agent) == ("agent", rig.free)
    assert rig.store.get(agent).description != "moved"  # the whole save rolled back

    assert _refused(rig, {"slug": chat, "agent_backend": rig.held}).status_code == 409
    assert rig.runtime(chat) == ("chat", None)

    created = f"{rig.bots[0]}-new"
    rig.bots.append(created)
    assert _refused(rig, {"slug": created, "agent_backend": rig.held}, create_only=True).status_code == 409
    assert rig.runtime(created) is None


def test_an_edit_of_a_bot_already_held_is_admitted_and_stays_held(rig):
    agent = rig.bot(rig.held)
    rig.hold()
    assert rig.blocked(agent)

    profile = rig.save({"slug": agent, "description": "edited while held"})
    assert profile.description == "edited while held"
    assert rig.runtime(agent) == ("agent", rig.held)
    assert rig.blocked(agent)


def test_switch_out_mid_hold_is_admitted_and_leaves_the_roster_alone(rig):
    agent = rig.bot(rig.held)
    turn_id = f"turn-{uuid.uuid4().hex}"
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, ended_at, agent_request_id)
            VALUES (:id, now(), '/test', true, :bot, 'nick', 'streaming', '', '', NULL, :req)
        """), {"id": turn_id, "bot": agent, "req": f"req-{turn_id}"})
    gate = rig.hold()
    roster = rig.members(gate)
    assert roster == [(turn_id, agent, "running")]

    rig.save({"slug": agent, "agent_backend": rig.free})
    assert rig.runtime(agent) == ("agent", rig.free)
    assert not rig.blocked(agent)
    assert rig.members(gate) == roster


def test_the_save_takes_the_admission_lock_before_its_row_lock(rig):
    """Rotation paths lock the target, then UPDATE bot_profiles; the save must not invert that."""
    agent = rig.bot(rig.free)
    done = threading.Event()
    errors: list[BaseException] = []

    def save():
        try:
            rig.save({"slug": agent, "agent_backend": rig.held})
        except BaseException as exc:  # surfaced below
            errors.append(exc)
        finally:
            done.set()

    with rig.engine.connect() as conn:
        txn = conn.begin()
        lock_target(conn, agent)
        worker = threading.Thread(target=save)
        worker.start()
        holder_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
        wait_blocked_by(rig.engine, holder_pid, worker)  # parked on the admission lock
        # ...without holding the profile row: a row lock now is still free.
        conn.execute(text("SELECT 1 FROM bot_profiles WHERE slug=:b FOR UPDATE NOWAIT"), {"b": agent})
        txn.commit()
    worker.join(10)
    assert not worker.is_alive()
    assert done.is_set() and errors == []
    assert rig.runtime(agent) == ("agent", rig.held)
