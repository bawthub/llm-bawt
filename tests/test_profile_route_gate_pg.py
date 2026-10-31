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
from sqlalchemy import event, text

from llm_bawt.memory.model_catalog_migration import migrate_model_catalog
from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore
from llm_bawt.prompt_registry import PromptTemplateStore
from llm_bawt.service.schemas_bots import BotCreateRequest, BotProfileUpsertRequest
from llm_bawt.service import profile_route_gate
from llm_bawt.service.routes import bot_profile_validation, settings
from llm_bawt.service.turn_admission import activate_hold, lock_target
from llm_bawt.utils.bot_partition import lock_bot_storage, sanitize_table_name
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
    assert RuntimeSettingsStore(private_app_pg).engine is not None
    assert store.engine is not None
    ensure_gate_schema(store.engine)
    migrate_model_catalog(store.engine)
    tag = uuid.uuid4().hex[:12]
    ctx = SimpleNamespace(store=store, engine=store.engine, config=private_app_pg,
                          owner=f"job-route-{tag}", bots=[], endpoints={}, models=[])

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
            conn.execute(text("DELETE FROM runtime_settings WHERE scope_type='bot' AND scope_id=:b"), {"b": slug})
            conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug=:b"), {"b": slug})
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


def _put(rig, slug, *, name="Changed"):
    return asyncio.run(settings.upsert_bot_profile(slug, BotProfileUpsertRequest(
        name=name, system_prompt="Fixture persona", **rig.route("local"),
    )))


def test_legacy_put_refuses_missing_profile_without_creating_any_data(rig):
    slug = rig.new_slug()
    with pytest.raises(HTTPException) as refused:
        _put(rig, slug)
    assert refused.value.status_code == 404
    assert rig.store.get(slug) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM bot_profile_tags WHERE bot_slug=:b"),
                            {"b": slug}).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM runtime_settings "
                                 "WHERE scope_type='bot' AND scope_id=:b"),
                            {"b": slug}).scalar_one() == 0


def test_legacy_put_does_not_resurrect_profile_deleted_while_waiting_on_lock(rig):
    slug = rig.bot("local")
    outcome: list[object] = []
    with rig.engine.connect() as conn:
        txn = conn.begin()
        lock_target(conn, slug)

        def stale_update():
            try:
                outcome.append(_put(rig, slug))
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=stale_update)
        worker.start()
        holder_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
        wait_blocked_by(rig.engine, holder_pid, worker)
        conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})
        txn.commit()
    worker.join(10)
    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], HTTPException) and outcome[0].status_code == 404
    assert rig.store.get(slug) is None


def test_legacy_put_keeps_existing_profile_update(rig):
    slug = rig.bot("local")
    updated = _put(rig, slug, name="Still editable")
    assert updated.name == "Still editable"
    assert rig.store.get(slug).name == "Still editable"


def _create(rig, slug, *, tags=None, runtime_settings=None, **fields):
    return asyncio.run(settings.create_bot(BotCreateRequest(
        slug=slug, name="New", system_prompt="Fixture persona",
        tags=[] if tags is None else tags,
        runtime_settings={} if runtime_settings is None else runtime_settings,
        **rig.route("local"), **fields,
    )))


def test_complete_create_commits_profile_tags_and_only_explicit_settings(rig):
    slug = rig.new_slug()
    created = _create(rig, slug, tags=["private"],
                      runtime_settings={"history_scope": "inline", "working_memory_enabled": None},
                      prompt_override_id=None)
    with rig.engine.connect() as conn:
        assert created.slug == slug
        assert created.prompt_override_id is None
        assert conn.execute(text("SELECT tag_slug FROM bot_profile_tags WHERE bot_slug=:b"),
                            {"b": slug}).scalars().all() == ["private"]
        assert conn.execute(text("SELECT key, value_json FROM runtime_settings "
                                 "WHERE scope_type='bot' AND scope_id=:b ORDER BY key"),
                            {"b": slug}).all() == [
                                ("history_scope", '"inline"'),
                                ("session_memory_continuity", "true"),
                            ]
        assert conn.execute(text("SELECT include_in_global_search FROM bot_profiles WHERE slug=:b"),
                            {"b": slug}).scalar_one() is False


def test_create_rolls_back_profile_and_tags_if_late_setting_fails(rig):
    slug = rig.new_slug()
    with pytest.raises(HTTPException) as refused:
        _create(rig, slug, tags=["private"],
                runtime_settings={"history_scope": "inline", "compact_context": object()})
    assert refused.value.status_code == 422
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM bot_profiles WHERE slug=:b"),
                            {"b": slug}).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM bot_profile_tags WHERE bot_slug=:b"),
                            {"b": slug}).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM runtime_settings "
                                 "WHERE scope_type='bot' AND scope_id=:b"),
                            {"b": slug}).scalar_one() == 0


def test_create_db_setting_failure_rolls_back_flushed_profile_tags_and_audit(rig):
    slug = rig.new_slug()

    def fail_setting_write(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith("INSERT INTO runtime_settings"):
            raise RuntimeError("synthetic settings write failure")

    event.listen(rig.engine, "before_cursor_execute", fail_setting_write)
    try:
        with pytest.raises(RuntimeError, match="synthetic settings write failure"):
            _create(rig, slug, tags=["private"], runtime_settings={"compact_context": True})
    finally:
        event.remove(rig.engine, "before_cursor_execute", fail_setting_write)
    with rig.engine.connect() as conn:
        for sql in (
            "SELECT count(*) FROM bot_profiles WHERE slug=:b",
            "SELECT count(*) FROM bot_profile_tags WHERE bot_slug=:b",
            "SELECT count(*) FROM bot_profile_changes WHERE bot_slug=:b",
            "SELECT count(*) FROM runtime_settings WHERE scope_type='bot' AND scope_id=:b",
        ):
            assert conn.execute(text(sql), {"b": slug}).scalar_one() == 0


def test_create_only_checks_existing_slug_after_bot_lock(rig):
    slug = rig.new_slug()
    outcome: list[object] = []
    with rig.engine.connect() as conn:
        txn = conn.begin()
        lock_target(conn, slug)

        def competing_create():
            try:
                outcome.append(_create(rig, slug, runtime_settings={"history_scope": "none"}))
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=competing_create)
        worker.start()
        holder_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
        wait_blocked_by(rig.engine, holder_pid, worker)
        conn.execute(text(
            "INSERT INTO bot_profiles (slug,name,description,system_prompt,harness,endpoint_id,"
            "default_model,requires_memory,voice_optimized,uses_tools,uses_search,"
            "uses_home_assistant,created_at,updated_at) "
            "SELECT :b,'Existing','','Fixture persona','chat',e.id,m.key,true,false,false,"
            "false,false,now(),now() "
            "FROM model_endpoints e JOIN models m ON m.id=e.model_id WHERE e.id=:e"
        ), {"b": slug, "e": rig.endpoints["local"]})
        txn.commit()
    worker.join(10)
    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], HTTPException) and outcome[0].status_code == 409
    assert rig.store.get(slug).name == "Existing"
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM runtime_settings "
                                 "WHERE scope_type='bot' AND scope_id=:b"),
                            {"b": slug}).scalar_one() == 0


@pytest.mark.parametrize("overrides", [
    {"unknown_setting": True},
    {"maintenance_model": True},
    {"bot_send_wait_seconds": 30},  # global-only
    {"include_summaries": True},  # request flag, not a bot runtime row
    {"history_scope": "unsupported"},
    {"working_memory_enabled": True},  # protected/private bot
])
def test_create_rejects_invalid_overrides_without_profile_or_tags(rig, overrides):
    slug = rig.new_slug()
    with pytest.raises(HTTPException) as refused:
        _create(rig, slug, tags=["private"], runtime_settings=overrides)
    assert refused.value.status_code in {400, 422}
    assert rig.store.get(slug) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM bot_profile_tags WHERE bot_slug=:b"),
                            {"b": slug}).scalar_one() == 0


def test_create_rejects_storage_alias_of_registered_bot(rig):
    existing = rig.new_slug()
    rig.store.upsert({"slug": existing, **rig.route("local")})
    # Removing hyphens yields a *different* slug but the same partition ID.
    alias = existing.replace("-", "")
    with pytest.raises(HTTPException) as refused:
        _create(rig, alias, runtime_settings={"history_scope": "none"})
    assert refused.value.status_code == 409
    assert rig.store.get(alias) is None
    assert rig.store.get(existing) is not None


def test_create_rejects_retained_audit_identity_without_profile(rig):
    deleted = rig.new_slug()
    alias = deleted.replace("-", "")
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_profile_changes (bot_slug,changed_by,changes) "
                          "VALUES (:b,'fixture','{}'::jsonb)"), {"b": deleted})
    with pytest.raises(HTTPException) as refused:
        _create(rig, alias)
    assert refused.value.status_code == 409
    assert rig.store.get(alias) is None
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM bot_profile_changes WHERE bot_slug=:b"), {"b": deleted})


def test_create_normalizes_slug_before_writing_bot_settings(rig):
    slug = rig.new_slug()
    created = _create(rig, f"  {slug.upper()}  ", runtime_settings={"history_scope": "none"})
    assert created.slug == slug
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT scope_id FROM runtime_settings WHERE key='history_scope' "
                                 "AND scope_type='bot' AND scope_id=:b"),
                            {"b": slug}).scalar_one() == slug


def test_create_refuses_reusing_profileless_tag_or_settings_identity(rig):
    first = rig.new_slug()
    second = rig.new_slug()
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_profile_tags (bot_slug,tag_slug) VALUES (:b,'private')"),
                     {"b": first})
        conn.execute(text("INSERT INTO runtime_settings (scope_type,scope_id,key,value_json,updated_at) "
                          "VALUES ('bot',:b,'history_scope','\"none\"',now())"), {"b": second})
    for slug in (first, second):
        with pytest.raises(HTTPException) as refused:
            _create(rig, slug, runtime_settings={"history_scope": "inline"})
        assert refused.value.status_code == 409
        assert rig.store.get(slug) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT tag_slug FROM bot_profile_tags WHERE bot_slug=:b"),
                            {"b": first}).scalar_one() == "private"
        assert conn.execute(text("SELECT value_json FROM runtime_settings "
                                 "WHERE scope_type='bot' AND scope_id=:b"),
                            {"b": second}).scalar_one() == '"none"'


def test_create_persona_selection_is_validated_before_commit(rig):
    assert PromptTemplateStore(rig.config).engine is not None
    invalid = rig.new_slug()
    with pytest.raises(HTTPException) as refused:
        _create(rig, invalid, prompt_override_id=2_000_000_000)
    assert refused.value.status_code == 422
    assert rig.store.get(invalid) is None

    valid = rig.new_slug()
    with rig.engine.begin() as conn:
        persona = conn.execute(text(
            "INSERT INTO prompt_templates (key, scope_type, scope_id, category, format, body, "
            "created_at, updated_at) VALUES (:k,'global','*','persona','plain_text',"
            "'Fixture persona',now(),now()) RETURNING id"
        ), {"k": f"persona-test-{valid}"}).scalar_one()
    try:
        created = _create(rig, valid, prompt_override_id=persona)
        assert created.prompt_override_id == persona
    finally:
        with rig.engine.begin() as conn:
            conn.execute(text("DELETE FROM prompt_templates WHERE id=:id"), {"id": persona})


@pytest.mark.parametrize("route_fields", [
    {},
    {"endpoint_id": None},
    {"harness": "codex", "agent_backend": "codex"},
])
def test_create_requires_compatible_real_endpoint(rig, monkeypatch, route_fields):
    monkeypatch.setattr(settings, "_validate_profile_payload", bot_profile_validation._validate_profile_payload)
    monkeypatch.setattr(bot_profile_validation, "get_service", settings.get_service)
    slug = rig.new_slug()
    kwargs = {**rig.route("local"), **route_fields} if route_fields else {}
    with pytest.raises(HTTPException) as refused:
        asyncio.run(settings.create_bot(BotCreateRequest(
            slug=slug, name="New", system_prompt="Fixture persona", tags=[], **kwargs,
        )))
    assert refused.value.status_code == 422
    assert rig.store.get(slug) is None


def test_create_rejects_conflicting_requested_model_without_writing(rig, monkeypatch):
    slug = rig.new_slug()
    monkeypatch.setattr(settings, "_validate_profile_payload",
                        lambda payload: payload.update(default_model="selected-endpoint-model"))
    with pytest.raises(HTTPException) as refused:
        _create(rig, slug, default_model="different-model")
    assert refused.value.status_code == 422
    assert rig.store.get(slug) is None


def test_create_requires_explicit_tags_name_and_system_prompt():
    with pytest.raises(Exception):
        BotCreateRequest(slug="fixture", name="Example", system_prompt="Fixture persona")
    with pytest.raises(Exception):
        BotCreateRequest(slug="fixture", name="   ", system_prompt="Fixture persona", tags=[])
    with pytest.raises(Exception):
        BotCreateRequest(slug="  ", name="Example", system_prompt="Fixture persona", tags=[])
    with pytest.raises(Exception):
        BotCreateRequest(slug="fixture", name="Example", system_prompt="", tags=[])
    with pytest.raises(Exception):
        BotCreateRequest(slug="fixture", name="Example", system_prompt="   ", tags=[])
    with pytest.raises(Exception):
        BotCreateRequest(slug="fixture", name="Example", system_prompt="Fixture persona",
                         tags=[], unexpected_field="ignored-before")
