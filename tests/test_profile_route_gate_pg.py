"""PostgreSQL tests for the TASK-1302 A1′ profile route gate.

The profile route's real ``_persist_bot_profile`` runs against the real
``BotProfileStore``, the real normalized catalog (``migrate_model_catalog`` on
this module's private DB, so the endpoint trigger is live) and real system
gates. Stubbed: payload validation (covered in ``test_route_eligibility.py`` and
``test_model_consolidation.py``), the post-commit side effects, and provider
health, which is injected per test through ``access_path_connections`` so no
test depends on a real credential.

Every route here is synthetic: a model key unique to the run on the seeded
access paths. No live bot row is read or written.
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

from llm_bawt.memory.model_catalog_migration import migrate_model_catalog
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service import profile_route_gate
from llm_bawt.service.routes import settings
from llm_bawt.service.turn_admission import activate_hold, lock_target
from llm_bawt.service.turn_gates import GateScope, GateSpec, ensure_gate_schema
from tests.pg_lock_wait import wait_blocked_by

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

#: (route key, harness, agent_backend) for every route the tests use.
_ROUTES = {
    "dead": ("openai-api", "chat", None),          # api-key, injected unconfigured
    "live": ("xai-chat", "chat", None),            # api-key, injected ok
    "local": ("local-vllm", "chat", None),         # auth none: no credential needed
    "untracked": ("openai-compatible", "chat", None),  # no provider owns it
    "kimi": ("kimi-coding-chat", "chat", None),    # state varies per test
    "oauth": ("openai-oauth", "codex", "codex"),
    "oauth2": ("openai-oauth", "codex", "codex"),  # same path, second model
}


def _connection(state: str, detail: str | None = None) -> dict:
    return {"provider_id": "test", "label": "Test Provider", "state": state,
            "detail": detail, "connected": state in {"ok", "warning"}}


@pytest.fixture
def rig(private_app_pg, monkeypatch):
    store = BotProfileStore(private_app_pg)
    assert store.engine is not None
    ensure_gate_schema(store.engine)
    migrate_model_catalog(store.engine)
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(store=store, engine=store.engine, owner=f"job-route-{tag}",
                          bots=[], endpoints={}, models=[])

    with ctx.engine.begin() as conn:
        for name, (path_key, _harness, _backend) in _ROUTES.items():
            model_key = f"route-gate-{tag}-{name}"
            ctx.models.append(model_key)
            model_id = conn.execute(text(
                "INSERT INTO models (key, vendor, display_name, default_context_window)"
                " VALUES (:k, 'test', :k, 1000) RETURNING id"), {"k": model_key}).scalar_one()
            ctx.endpoints[name] = conn.execute(text(
                "INSERT INTO model_endpoints (model_id, access_path_id)"
                " SELECT :m, id FROM access_paths WHERE key = :p RETURNING id"),
                {"m": model_id, "p": path_key}).scalar_one()

    # Provider health per access path; tests override entries. Unowned paths
    # are None, exactly as access_path_connections reports them.
    ctx.connections = {
        "openai-api": _connection("unconfigured"),
        "xai-chat": _connection("ok"),
        "local-vllm": None,
        "openai-compatible": None,
        "kimi-coding-chat": _connection("ok"),
        "openai-oauth": _connection("ok"),
    }
    ctx.probes = []

    def connections(_config, keys):
        keys = list(keys)
        ctx.probes.append(keys)
        return {key: ctx.connections.get(key) for key in keys}

    service = SimpleNamespace(config=private_app_pg, _redis_subscriber=SimpleNamespace(
        publish_global_config_event=AsyncMock(return_value="1-0"),
        publish_tool_event=AsyncMock(return_value="2-0")))
    monkeypatch.setattr(profile_route_gate, "access_path_connections", connections)
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: None)
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_a, **_k: None)
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response", lambda profile, settings, tags=None: profile)
    monkeypatch.setattr(settings, "rotate_active_threads", lambda engine, slug: [])

    def route(name):
        """The payload fields validation would produce for route ``name``."""
        _path, harness, backend = _ROUTES[name]
        return {"endpoint_id": ctx.endpoints[name], "harness": harness, "agent_backend": backend}

    def new_slug():
        slug = f"test-route-{tag}-{len(ctx.bots)}"
        ctx.bots.append(slug)
        return slug

    def bot(name):
        """A bot already on route ``name``, written directly (no gate)."""
        slug = new_slug()
        store.upsert({"slug": slug, **route(name)})
        return slug

    def save(payload, *, create_only=False):
        return asyncio.run(settings._persist_bot_profile(dict(payload), create_only=create_only))

    def endpoint(slug):
        row = store.get(slug)
        return None if row is None else row.endpoint_id

    def hold(backend):
        spec = GateSpec(owner_kind="job", owner_id=ctx.owner, mode="hold",
                        scope=GateScope.on_backends(backend), ttl_seconds=600,
                        reason="test route gate", allow_actors=frozenset())
        with ctx.engine.begin() as conn:
            return activate_hold(conn, spec).gate.id

    ctx.route, ctx.new_slug, ctx.bot, ctx.save, ctx.endpoint, ctx.hold = (
        route, new_slug, bot, save, endpoint, hold)
    yield ctx
    with ctx.engine.begin() as conn:
        conn.execute(text("DELETE FROM gate_members WHERE gate_id IN"
                          " (SELECT id FROM system_gates WHERE owner_id=:o)"), {"o": ctx.owner})
        conn.execute(text("DELETE FROM system_gates WHERE owner_id=:o"), {"o": ctx.owner})
        for slug in ctx.bots:
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})
        for model_key in ctx.models:
            conn.execute(text("DELETE FROM models WHERE key=:k"), {"k": model_key})


def _refused(rig, payload, **kwargs) -> HTTPException:
    with pytest.raises(HTTPException) as caught:
        rig.save(payload, **kwargs)
    return caught.value


def test_a_stale_save_cannot_restore_a_dead_route_over_a_newer_one(rig):
    """The gate judges the row read under the admission lock, not a pre-lock read.

    The bot sits on the dead route; a name edit carrying that endpoint parks
    on the lock while another writer moves the bot to a live route and
    commits. Landing after it, the stale save would put the bot back on the
    dead route, so it is refused whole.
    """
    agent = rig.bot("dead")
    done = threading.Event()
    outcome: list[object] = []

    def stale_save():
        try:
            outcome.append(rig.save({"slug": agent, "name": "Stale", **rig.route("dead")}))
        except BaseException as exc:  # asserted below
            outcome.append(exc)
        finally:
            done.set()

    with rig.engine.connect() as conn:
        txn = conn.begin()
        lock_target(conn, agent)
        worker = threading.Thread(target=stale_save)
        worker.start()
        holder_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
        wait_blocked_by(rig.engine, holder_pid, worker)  # parked on the admission lock
        conn.execute(text("UPDATE bot_profiles SET endpoint_id=:e WHERE slug=:b"),
                     {"e": rig.endpoints["live"], "b": agent})
        txn.commit()
    worker.join(10)
    assert not worker.is_alive()

    assert done.is_set() and len(outcome) == 1
    refused = outcome[0]
    assert isinstance(refused, HTTPException) and refused.status_code == 409
    assert "not connected" in refused.detail
    assert rig.endpoint(agent) == rig.endpoints["live"]
    assert rig.store.get(agent).name != "Stale"  # the whole save rolled back


def test_an_edit_that_keeps_a_dead_route_is_admitted(rig):
    agent = rig.bot("dead")

    assert rig.save({"slug": agent, "description": "persona edit"}).description == "persona edit"
    kept = rig.save({"slug": agent, "description": "again", **rig.route("dead")})
    assert kept.description == "again"
    assert rig.endpoint(agent) == rig.endpoints["dead"]


@pytest.mark.parametrize(("name", "state", "message"), [
    ("dead", None, "Test Provider is not connected. Connect it in Integrations"),
    ("kimi", "broken", "Test Provider needs reconnecting (token revoked)."),
    ("kimi", "unknown", "Test Provider status couldn't be checked (token revoked)."),
    ("untracked", None, "BawtHub doesn't manage the credential for openai-compatible"),
])
def test_moving_onto_an_ineligible_route_is_refused_and_writes_nothing(rig, name, state, message):
    if state is not None:
        rig.connections[_ROUTES[name][0]] = _connection(state, "token revoked")
    agent = rig.bot("live")

    refused = _refused(rig, {"slug": agent, "description": "moved", **rig.route(name)})
    assert refused.status_code == 409
    assert refused.detail.startswith(message)
    assert rig.endpoint(agent) == rig.endpoints["live"]
    assert rig.store.get(agent).description != "moved"


@pytest.mark.parametrize(("name", "state"), [
    ("live", "ok"),
    ("kimi", "warning"),
    ("local", None),
])
def test_moving_onto_a_connected_or_local_route_is_admitted(rig, name, state):
    if state is not None:
        rig.connections[_ROUTES[name][0]] = _connection(state)
    agent = rig.bot("dead")

    rig.save({"slug": agent, **rig.route(name)})
    assert rig.endpoint(agent) == rig.endpoints[name]


def test_a_synthetic_xai_chat_route_is_eligible_when_xai_is_connected(rig):
    """xai-chat is owned by the xAI adapter, so it is assignable once connected (Snark).

    The bot is synthetic; no live bot row (Mira's included) is touched.
    """
    agent = rig.bot("dead")
    rig.save({"slug": agent, **rig.route("live")})
    assert rig.endpoint(agent) == rig.endpoints["live"]

    rig.connections["xai-chat"] = _connection("unconfigured")
    other = rig.bot("dead")
    assert _refused(rig, {"slug": other, **rig.route("live")}).status_code == 409


def test_a_held_bot_may_change_its_endpoint_on_the_same_runtime(rig):
    """The switch gate compares the runtime (bot_type, agent_backend), not the route."""
    agent = rig.bot("oauth")
    rig.hold("codex")

    rig.save({"slug": agent, **rig.route("oauth2")})
    assert rig.endpoint(agent) == rig.endpoints["oauth2"]


def test_creating_a_bot_on_an_ineligible_route_writes_nothing(rig):
    slug = rig.new_slug()
    refused = _refused(rig, {"slug": slug, **rig.route("dead")}, create_only=True)
    assert refused.status_code == 409
    assert rig.store.get(slug) is None

    created = rig.new_slug()
    rig.save({"slug": created, **rig.route("live")}, create_only=True)
    assert rig.endpoint(created) == rig.endpoints["live"]


def test_a_held_runtime_is_reported_before_the_route(rig):
    rig.connections["openai-oauth"] = _connection("unconfigured")
    agent = rig.bot("live")
    rig.hold("codex")

    refused = _refused(rig, {"slug": agent, **rig.route("oauth")})
    assert refused.status_code == 409 and refused.detail.startswith("system_held")
    assert rig.endpoint(agent) == rig.endpoints["live"]


def test_health_is_probed_before_the_lock_and_only_for_the_payload_route(rig):
    agent = rig.bot("dead")
    rig.probes.clear()

    rig.save({"slug": agent, "description": "no route in payload"})
    assert rig.probes == []
    rig.save({"slug": agent, **rig.route("live")})
    assert rig.probes == [["xai-chat"]]


def test_a_route_missing_from_the_snapshot_is_refused_not_probed_under_the_lock(rig):
    """Fail closed: the gate never probes inside the transaction."""
    agent = rig.bot("live")
    gate = profile_route_gate.ProfileRouteGate({})  # nothing snapshotted
    row = rig.store.get(agent)
    row.endpoint_id = rig.endpoints["kimi"]
    with rig.engine.connect() as conn:
        with pytest.raises(profile_route_gate.RouteIneligible, match="changed during the save"):
            gate.check(conn, None, row)
        row.endpoint_id = rig.endpoints["local"]
        gate.check(conn, None, row)  # a credential-free route needs no snapshot
