"""TASK-1233: the operator freeze on real PostgreSQL, through its API.

Every freeze here is scoped to this test's own bots, so parallel PG tests are
never frozen by it. The route runs with a test client and the real gate store.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text as sa_text

from llm_bawt.bawthub_auth import TOKEN_ENV, TOKEN_FILE_ENV
from llm_bawt.service import operator_freeze
from llm_bawt.service.routes import admin_gates as route
from llm_bawt.service.turn_admission import SystemGateRefused, activate_hold, lock_target, refuse_if_gated
from llm_bawt.service.turn_gates import ACTOR_KINDS, GateActor, GateScope, GateSpec, gate_is_live
from tests.pg_isolation import isolated_pg_store

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

TOKEN = "fixture-service-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URL = "/v1/admin/gates"


@pytest.fixture
def fz(monkeypatch):
    """A client on the real gate store, two agent bots, and cleanup of every gate here."""
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.service.turn_logs import TurnLogStore

    store = isolated_pg_store(TurnLogStore)
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    assert BotProfileStore(store.config).engine is not None  # creates bot_profiles
    engine = store.engine
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_gate_engine] = lambda: engine
    wakes: list[int] = []
    app.dependency_overrides[route.get_dispatch_wake] = lambda: (lambda: wakes.append(1))
    tag = uuid.uuid4().hex[:12]
    bots = [f"test-freeze-{tag}-{i}" for i in range(2)]
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        for slug in bots:
            conn.execute(sa_text("""
                INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                    voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                    agent_backend, created_at, updated_at)
                VALUES (:slug, :slug, '', '', false, false, false, false, false, 'agent',
                    'claude-code', :now, :now)
            """), {"slug": slug, "now": now})

    class Ctx:
        pass

    ctx = Ctx()
    ctx.engine, ctx.client, ctx.bots, ctx.tag, ctx.wakes = engine, TestClient(app), bots, tag, wakes
    ctx.actor = lambda name: f"{name}-{tag}"
    yield ctx
    with engine.begin() as conn:
        owned = "SELECT id FROM system_gates WHERE owner_id LIKE :owners"
        conn.execute(sa_text(f"DELETE FROM gate_members WHERE gate_id IN ({owned})"),
                     {"owners": f"%{tag}%"})
        conn.execute(sa_text("DELETE FROM system_gates WHERE owner_id LIKE :owners"),
                     {"owners": f"%{tag}%"})
        for slug in bots:
            conn.execute(sa_text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})


def _freeze(fz, actor, *, bots=None, ttl=3600, reason="deploy window"):
    return fz.client.post(URL, headers=AUTH, json={
        "actor": actor, "reason": reason, "ttl_seconds": ttl,
        "scope": {"bots": bots or fz.bots[:1]}})


def _thaw(fz, gate_id, actor, **extra):
    return fz.client.post(f"{URL}/{gate_id}/thaw", headers=AUTH, json={"actor": actor, **extra})


def _refusal(fz, bot, kind="reserve"):
    """The code a real admission edge would refuse with, or None when admitted."""
    with fz.engine.begin() as conn:
        lock_target(conn, bot)
        try:
            refuse_if_gated(conn, bot, GateActor(kind))
        except SystemGateRefused as refused:
            return refused.status_code, refused.code
    return None


def _ours(fz, status):
    return {f["gate_id"]: f for f in status.json()["freezes"] if f["actor"].endswith(fz.tag)}


def test_freeze_is_an_operator_deny_gate_with_a_db_clock_lease(fz):
    response = _freeze(fz, fz.actor("op"), ttl=900)
    assert response.status_code == 201
    frozen = response.json()["freeze"]
    assert frozen["actor"] == fz.actor("op") and frozen["reason"] == "deploy window"
    assert frozen["scope"] == {"bots": [fz.bots[0]]}
    with fz.engine.connect() as conn:
        row = conn.execute(sa_text(
            "SELECT owner_kind, owner_id, mode, allow_actors,"
            " expires_at > now() + interval '840 seconds' AS long_enough,"
            " expires_at <= clock_timestamp() + interval '900 seconds' AS capped"
            " FROM system_gates WHERE id=:id"), {"id": frozen["gate_id"]}).mappings().one()
    assert (row["owner_kind"], row["owner_id"], row["mode"]) == ("operator", f"freeze:{fz.actor('op')}", "deny")
    assert row["allow_actors"] == []
    assert row["long_enough"] and row["capped"]


def test_freeze_refuses_every_edge_with_system_frozen_only_in_scope(fz):
    gate_id = _freeze(fz, fz.actor("op")).json()["freeze"]["gate_id"]
    for kind in sorted(ACTOR_KINDS):
        assert _refusal(fz, fz.bots[0], kind) == (503, "system_frozen"), kind
        assert _refusal(fz, fz.bots[1], kind) is None, kind
    assert _thaw(fz, gate_id, fz.actor("op")).json()["released"] is True
    for kind in sorted(ACTOR_KINDS):
        assert _refusal(fz, fz.bots[0], kind) is None, kind


def test_a_frozen_system_can_still_read_its_status_and_thaw_itself(fz):
    """Status and thaw are not admission edges: the freeze never blocks them."""
    actor = fz.actor("op")
    gate_id = _freeze(fz, actor, bots=fz.bots).json()["freeze"]["gate_id"]
    assert all(_refusal(fz, bot) == (503, "system_frozen") for bot in fz.bots)

    status = fz.client.get(URL, headers=AUTH)
    assert status.status_code == 200 and status.json()["frozen"] is True
    assert _ours(fz, status)[gate_id]["reason"] == "deploy window"

    thawed = _thaw(fz, gate_id, actor)
    assert thawed.status_code == 200
    assert thawed.json()["released"] is True and thawed.json()["override"] is False
    assert all(_refusal(fz, bot) is None for bot in fz.bots)
    assert gate_id not in _ours(fz, fz.client.get(URL, headers=AUTH))
    assert fz.wakes == [1]  # held deliveries dispatch at once
    # Idempotent: a second thaw is no error and releases nothing.
    again = _thaw(fz, gate_id, actor)
    assert again.status_code == 200 and again.json()["released"] is False
    assert fz.wakes == [1]


def test_only_the_owner_thaws_unless_overridden(fz):
    gate_id = _freeze(fz, fz.actor("owner")).json()["freeze"]["gate_id"]
    refused = _thaw(fz, gate_id, fz.actor("other"))
    assert refused.status_code == 403 and refused.json() == {"detail": "not_owner"}
    with fz.engine.connect() as conn:
        assert gate_is_live(conn, gate_id)

    overridden = _thaw(fz, gate_id, fz.actor("other"), override=True)
    assert overridden.status_code == 200
    assert overridden.json()["released"] is True and overridden.json()["override"] is True
    assert _refusal(fz, fz.bots[0]) is None


def test_one_live_freeze_per_actor(fz):
    first = _freeze(fz, fz.actor("op")).json()["freeze"]
    second = _freeze(fz, fz.actor("op"))
    assert second.status_code == 409
    assert second.json()["detail"] == "already_frozen"
    assert second.json()["freeze"]["gate_id"] == first["gate_id"]
    # Another actor's freeze stacks; each thaws its own.
    other = _freeze(fz, fz.actor("other"))
    assert other.status_code == 201
    _thaw(fz, first["gate_id"], fz.actor("op"))
    assert _refusal(fz, fz.bots[0]) == (503, "system_frozen")
    _thaw(fz, other.json()["freeze"]["gate_id"], fz.actor("other"))
    assert _refusal(fz, fz.bots[0]) is None


def test_concurrent_freezes_by_one_actor_have_one_winner(fz):
    count = 6
    barrier = threading.Barrier(count)
    codes: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait()
            try:
                operator_freeze.freeze(fz.engine, actor=fz.actor("op"), reason="race",
                                       scope=GateScope.for_bots(fz.bots[0]), ttl_seconds=600)
                code = 201
            except operator_freeze.FreezeConflict:
                code = 409
        except BaseException as error:  # noqa: BLE001 - surfaced below
            with lock:
                errors.append(error)
            return
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    assert sorted(codes) == [201] + [409] * (count - 1)
    with fz.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT count(*) FROM system_gates WHERE owner_id=:o"
        ), {"o": f"freeze:{fz.actor('op')}"}).scalar() == 1


def test_an_expired_freeze_lifts_with_no_thaw(fz):
    """Expiry is the gate predicate on the DB clock: nothing has to run."""
    actor = fz.actor("op")
    gate_id = _freeze(fz, actor).json()["freeze"]["gate_id"]
    with fz.engine.begin() as conn:  # the lease runs out
        conn.execute(sa_text(
            "UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id"
        ), {"id": gate_id})
    assert _refusal(fz, fz.bots[0]) is None
    assert gate_id not in _ours(fz, fz.client.get(URL, headers=AUTH))
    # A lapsed freeze doesn't count against its actor, and thawing it is a no-op.
    assert _freeze(fz, actor).status_code == 201
    lapsed = _thaw(fz, gate_id, actor)
    assert lapsed.status_code == 200 and lapsed.json()["released"] is False


def test_override_never_thaws_a_gate_that_is_not_a_freeze(fz):
    specs = [
        GateSpec(owner_kind="job", owner_id=f"job-{fz.tag}", mode="hold",
                 scope=GateScope.for_bots(fz.bots[1]), ttl_seconds=600, reason="quiesce"),
        GateSpec(owner_kind="operator", owner_id=f"user_stop:{fz.tag}", mode="hold",
                 scope=GateScope.for_bots(fz.bots[1]), ttl_seconds=600, reason="user_stop"),
        GateSpec(owner_kind="job", owner_id=f"freeze:job-{fz.tag}", mode="deny",
                 scope=GateScope.for_bots(fz.bots[1]), ttl_seconds=600, reason="job"),
    ]
    with fz.engine.begin() as conn:
        ids = [activate_hold(conn, spec).gate.id for spec in specs]
    for gate_id in ids:
        response = _thaw(fz, gate_id, fz.actor("op"), override=True)
        assert response.status_code == 404, gate_id
    with fz.engine.connect() as conn:
        assert all(gate_is_live(conn, gate_id) for gate_id in ids)
    listed = {f["gate_id"] for f in fz.client.get(URL, headers=AUTH).json()["freezes"]}
    assert not set(ids) & listed  # nor are they reported as freezes


def test_thaw_of_an_unknown_gate_is_404(fz):
    with fz.engine.connect() as conn:
        missing = conn.execute(sa_text("SELECT COALESCE(max(id), 0) + 1000000 FROM system_gates")).scalar()
    assert _thaw(fz, missing, fz.actor("op")).status_code == 404
