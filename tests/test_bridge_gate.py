"""TASK-1477 phase 1: one verifier for the X-Bridge-Token gates."""

from __future__ import annotations

import ast
import json
import os
import re
import stat
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service import bridge_gate
from llm_bawt.service.bridge_gate import (
    BROKER,
    DEFAULT_ENFORCED_ENV,
    EXPORT_UID_ENV,
    GATES,
    GIT_CREDENTIAL,
    VAULT,
    BridgeGates,
    Outcome,
    State,
    seed_enforced_defaults,
)

DERIVED = {gate.label: f"derived-{gate.slug}-" + "d" * 40 for gate in GATES}
LEGACY = {BROKER: "legacy-broker-fixture", GIT_CREDENTIAL: "legacy-git-fixture"}
SETTABLE = tuple(g for g in GATES if not g.always_enforced)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Settings:
    def __init__(self, values=None, *, fail=False) -> None:
        self.values = dict(values or {})
        self.fail = fail
        self.reads = 0

    def __call__(self):
        self.reads += 1
        if self.fail:
            raise RuntimeError("db down")
        return dict(self.values)


def make(environ=None, settings=None, *, derive=None, clock=None):
    return BridgeGates(settings or Settings(), environ=environ or {},
                       derive=derive or DERIVED.__getitem__, clock=clock or Clock())


OPT_OUT = {DEFAULT_ENFORCED_ENV: "0"}  # a network-only tenant's explicit open gate


def legacy_env(*gates):
    return {gate.legacy_env: LEGACY[gate] for gate in gates if gate.legacy_env}


def refused(gates, gate, header) -> bool:
    try:
        gates.check(gate, header)
    except HTTPException as error:
        assert error.status_code == 401
        return True
    return False


# --- classification ------------------------------------------------------
@pytest.mark.parametrize("gate", SETTABLE, ids=lambda g: g.slug)
def test_classifies_all_four_outcomes_per_gate(gate):
    gates = make(legacy_env(*GATES))
    assert gates.classify(gate, DERIVED[gate.label]) is Outcome.DERIVED
    assert gates.classify(gate, LEGACY[gate]) is Outcome.LEGACY
    assert gates.classify(gate, None) is Outcome.MISSING
    assert gates.classify(gate, "") is Outcome.BAD
    assert gates.classify(gate, "not-the-token") is Outcome.BAD


def test_gates_do_not_accept_each_others_tokens():
    gates = make(legacy_env(*GATES))
    assert gates.classify(BROKER, DERIVED[GIT_CREDENTIAL.label]) is Outcome.BAD
    assert gates.classify(GIT_CREDENTIAL, LEGACY[BROKER]) is Outcome.BAD


def test_comparisons_use_compare_digest(monkeypatch):
    calls = []
    real = bridge_gate.hmac.compare_digest
    monkeypatch.setattr(bridge_gate.hmac, "compare_digest",
                        lambda a, b: calls.append(1) or real(a, b))
    make(legacy_env(BROKER)).classify(BROKER, "anything")
    assert len(calls) == 2  # derived and legacy, both checked


def test_underivable_root_means_no_derived_token_not_an_error():
    def broken(_label):
        raise RuntimeError("key not persisted")

    gates = make(legacy_env(BROKER), derive=broken)
    assert gates.classify(BROKER, LEGACY[BROKER]) is Outcome.LEGACY
    assert gates.classify(BROKER, "x") is Outcome.BAD
    assert gates.derived_token(BROKER) is None


# --- default rule ---------------------------------------------------------
def test_default_enforces_without_a_setting_row():
    gates = make()
    assert refused(gates, BROKER, None)
    assert refused(gates, GIT_CREDENTIAL, None)
    assert not refused(gates, BROKER, DERIVED[BROKER.label])


def test_network_only_opt_out_allows_missing_refuses_bad():
    gates = make(OPT_OUT)
    assert not refused(gates, BROKER, None)
    assert not refused(gates, GIT_CREDENTIAL, None)
    assert refused(gates, BROKER, "wrong")
    assert not refused(gates, BROKER, DERIVED[BROKER.label])


def test_legacy_secret_enforces_despite_opt_out():
    gates = make({**legacy_env(BROKER), DEFAULT_ENFORCED_ENV: "0"})
    assert refused(gates, BROKER, None)
    assert not refused(gates, BROKER, LEGACY[BROKER])
    assert not refused(gates, BROKER, DERIVED[BROKER.label])
    assert refused(gates, BROKER, "wrong")
    # The git gate has its own legacy var; it stays network-only here.
    assert not refused(gates, GIT_CREDENTIAL, None)


def test_fresh_tenant_marker_enforces_by_default():
    gates = make({DEFAULT_ENFORCED_ENV: "1"})
    assert refused(gates, BROKER, None)
    assert refused(gates, GIT_CREDENTIAL, None)
    assert not refused(gates, GIT_CREDENTIAL, DERIVED[GIT_CREDENTIAL.label])


def test_setting_overrides_default_and_flip_lands_within_ttl():
    clock, settings = Clock(), Settings({BROKER.setting_key: True})
    gates = make(settings=settings, clock=clock)
    assert refused(gates, BROKER, None)
    settings.values[BROKER.setting_key] = False
    clock.now += 1
    assert refused(gates, BROKER, None)  # still cached
    clock.now += bridge_gate.SETTINGS_TTL_SECONDS
    assert not refused(gates, BROKER, None)


def test_setting_false_overrides_legacy_default():
    gates = make(legacy_env(BROKER), Settings({BROKER.setting_key: False}))
    assert not refused(gates, BROKER, None)
    assert refused(gates, BROKER, "wrong")  # bad is refused in every phase


def test_non_bool_setting_falls_back_to_default():
    gates = make(legacy_env(BROKER), Settings({BROKER.setting_key: "false"}))
    assert refused(gates, BROKER, None)


def test_unreadable_settings_fail_closed():
    assert refused(make(settings=Settings(fail=True)), BROKER, None)
    assert not refused(make(OPT_OUT, Settings(fail=True)), BROKER, None)
    assert refused(make(legacy_env(BROKER), Settings(fail=True)), BROKER, None)


# --- counters and health ------------------------------------------------
def test_counts_each_outcome_and_health_never_shows_values():
    gates = make(legacy_env(*GATES))
    gates.check(BROKER, DERIVED[BROKER.label])
    gates.check(BROKER, LEGACY[BROKER])
    for header in (None, "wrong"):
        with pytest.raises(HTTPException):
            gates.check(BROKER, header)
    assert gates.counts(BROKER) == {"derived": 1, "legacy": 1, "missing": 1, "bad": 1}
    assert gates.counts(GIT_CREDENTIAL) == {"derived": 0, "legacy": 0, "missing": 0, "bad": 0}
    blob = json.dumps(gates.health_checks())
    for value in [*DERIVED.values(), *LEGACY.values(), "wrong"]:
        assert value not in blob


def _row(gates, gate):
    return next(r for r in gates.health_checks() if r["name"] == f"bridge_gate.{gate.slug}")


def test_health_states():
    clock = Clock()
    gates = make(OPT_OUT, clock=clock)
    assert gates.state(BROKER) is State.IDLE
    gates.check(BROKER, None)
    assert gates.state(BROKER) is State.OPEN_UNAUTHENTICATED
    row = _row(gates, BROKER)
    assert row["status"] == "ok" and row["reason"].startswith("warning:")
    clock.now += bridge_gate.RECENT_SECONDS + 1
    gates.check(BROKER, DERIVED[BROKER.label])
    assert gates.state(BROKER) is State.OPEN_DERIVED_SEEN
    assert _row(gates, BROKER)["status"] == "degraded"
    clock.now += bridge_gate.RECENT_SECONDS + 1
    gates.check(BROKER, None)  # derived has moved before: not "unauthenticated"
    assert gates.state(BROKER) is State.IDLE
    enforced = make(legacy_env(BROKER))
    row = _row(enforced, BROKER)
    assert row["state"] == "enforced" and row["enforced"] is True and row["status"] == "ok"


def test_mixed_traffic_does_not_recommend_enforcing():
    # TASK-1670: a few derived requests amid many token-less ones must not
    # recommend enforcement, which would 401 every token-less bridge.
    clock = Clock()
    gates = make(OPT_OUT, clock=clock)
    for _ in range(3):
        gates.check(BROKER, None)
    gates.check(BROKER, DERIVED[BROKER.label])
    assert gates.state(BROKER) is State.OPEN_MIXED
    row = _row(gates, BROKER)
    assert row["status"] == "ok" and row["reason"].startswith("warning:")
    assert "don't set" in row["reason"]
    # Legacy callers still pass under enforcement, so they don't block it.
    legacy = make(legacy_env(BROKER), Settings({BROKER.setting_key: False}), clock=clock)
    legacy.check(BROKER, LEGACY[BROKER])
    legacy.check(BROKER, DERIVED[BROKER.label])
    assert legacy.state(BROKER) is State.OPEN_DERIVED_SEEN
    # Once token-less traffic ages out, the recommendation returns.
    clock.now += bridge_gate.RECENT_SECONDS + 1
    gates.check(BROKER, DERIVED[BROKER.label])
    assert gates.state(BROKER) is State.OPEN_DERIVED_SEEN


def test_collect_includes_gate_rows(monkeypatch):
    from types import SimpleNamespace

    from llm_bawt.service import health_checks

    monkeypatch.setattr(health_checks, "check_database", lambda _: {"name": "database", "status": "ok"})
    monkeypatch.setattr(health_checks, "check_redis", lambda _: {"name": "redis", "status": "ok"})
    monkeypatch.setattr(bridge_gate, "_GATES", make())
    ops = SimpleNamespace(executors=lambda: {})
    store = SimpleNamespace(engine=None)
    monkeypatch.setattr(health_checks, "check_ops_queue", lambda _s: {"name": "ops.queue", "status": "ok"})
    result = health_checks.collect(SimpleNamespace(REDIS_URL=""), store, ops)
    names = {row["name"] for row in result["checks"]}
    assert {"bridge_gate.broker", "bridge_gate.git_credential"} <= names


# --- export ------------------------------------------------------------
def test_export_writes_each_gate_0600_atomically(tmp_path):
    paths = {gate: tmp_path / "bridge-tokens" / gate.slug for gate in GATES}
    environ = {gate.export_env: str(path) for gate, path in paths.items()}
    assert make(environ).export_tokens() == {g.slug: "written" for g in GATES}
    for gate, path in paths.items():
        assert path.read_text().strip() == DERIVED[gate.label]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in (tmp_path / "bridge-tokens").iterdir()) == sorted(g.slug for g in GATES)


def test_export_chowns_to_bridge_uid(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: seen.append(uid))
    path = tmp_path / "broker"
    make({BROKER.export_env: str(path), EXPORT_UID_ENV: "1000"}).export_tokens()
    assert seen == [1000]


def test_export_disabled_without_path_and_removes_stale_file_without_root(tmp_path):
    assert make().export_tokens() == {g.slug: "disabled" for g in GATES}
    stale = tmp_path / "broker"
    stale.write_text("old")

    def broken(_label):
        raise RuntimeError("no root")

    assert make({BROKER.export_env: str(stale)}, derive=broken).export_tokens()["broker"] == "removed"
    assert not stale.exists()


def test_vault_host_copy_is_owned_by_the_config_dir_owner(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: seen.append(uid))
    host = tmp_path / "config" / "vault-token"
    host.parent.mkdir()
    result = make({bridge_gate.VAULT_HOST_EXPORT_ENV: str(host)}).export_tokens()
    assert result["vault_host"] == "written"
    assert host.read_text().strip() == DERIVED[VAULT.label]
    assert stat.S_IMODE(host.stat().st_mode) == 0o600
    assert seen == [host.parent.stat().st_uid]

    def broken(_label):
        raise RuntimeError("no root")

    assert make({bridge_gate.VAULT_HOST_EXPORT_ENV: str(host)}, derive=broken).export_tokens()[
        "vault_host"] == "removed"
    assert not host.exists()


def test_export_failure_never_raises(tmp_path):
    result = make({BROKER.export_env: str(tmp_path / "x"), EXPORT_UID_ENV: "nope"}).export_tokens()
    assert result["broker"] == "failed"


def test_derived_token_follows_the_root(monkeypatch):
    from cryptography.fernet import Fernet

    from llm_bawt.service.providers import crypto

    keys = iter([Fernet.generate_key(), Fernet.generate_key()])
    monkeypatch.setattr(crypto, "_load_or_create_key", lambda **_: next(keys))
    first = BridgeGates(Settings(), environ={}).derived_token(BROKER)
    second = BridgeGates(Settings(), environ={}).derived_token(BROKER)
    assert re.fullmatch(r"[0-9a-f]{64}", first) and first != second


# --- seeding -----------------------------------------------------------
@pytest.fixture
def engine():
    made = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(made, tables=[RuntimeSetting.__table__])
    with made.begin() as conn:  # mirrors the production unique index
        conn.execute(text("CREATE UNIQUE INDEX idx_runtime_settings_scope_key "
                          "ON runtime_settings (scope_type, scope_id, key)"))
    return made


def _settings(engine) -> dict:
    with Session(engine) as session:
        rows = session.exec(select(RuntimeSetting)).all()
    return {(r.scope_type, r.scope_id, r.key): json.loads(r.value_json) for r in rows}


def test_fresh_marker_seeds_enforcement_then_missing_is_refused(engine):
    env = {DEFAULT_ENFORCED_ENV: "1"}
    assert sorted(seed_enforced_defaults(engine, env)) == sorted(g.setting_key for g in SETTABLE)
    stored = _settings(engine)
    assert all(stored[("global", "*", g.setting_key)] is True for g in SETTABLE)
    assert ("global", "*", VAULT.setting_key) not in stored  # no row to delete or flip
    # The row is now the truth even if the marker is lost.
    gates = make({}, Settings({k[2]: v for k, v in stored.items()}))
    assert refused(gates, BROKER, None)
    assert seed_enforced_defaults(engine, env) == []  # idempotent


def test_seeding_respects_an_operator_false_row(engine):
    with Session(engine) as session:
        session.add(RuntimeSetting(scope_type="global", scope_id="*",
                                   key=BROKER.setting_key, value_json="false"))
        session.commit()
    assert seed_enforced_defaults(engine, {DEFAULT_ENFORCED_ENV: "1"}) == [GIT_CREDENTIAL.setting_key]
    assert _settings(engine)[("global", "*", BROKER.setting_key)] is False


def test_no_marker_seeds_nothing(engine):
    assert seed_enforced_defaults(engine, {}) == []
    assert seed_enforced_defaults(engine, legacy_env(BROKER)) == []
    assert _settings(engine) == {}


# --- routes --------------------------------------------------------------
def test_routes_go_through_the_gate(monkeypatch):
    from llm_bawt.service.routes import providers

    monkeypatch.setattr(bridge_gate, "_GATES", make(legacy_env(GIT_CREDENTIAL)))
    monkeypatch.setattr(providers, "get_service", lambda: type("S", (), {"config": None})())
    monkeypatch.setattr(providers, "get_adapter", lambda _config, _id: None)
    app = FastAPI()
    app.include_router(providers.router)
    client = TestClient(app)
    url = "/v1/providers/github/git-credential"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"X-Bridge-Token": "wrong"}).status_code == 401
    for token in (LEGACY[GIT_CREDENTIAL], DERIVED[GIT_CREDENTIAL.label]):
        assert client.get(url, headers={"X-Bridge-Token": token}).status_code == 404
    assert client.get("/v1/providers/openrouter/token", headers={"X-Bridge-Token": "wrong"}).status_code == 401


def test_vault_password_route_opens_only_for_the_vault_token(monkeypatch):
    from llm_bawt.service.routes import providers

    password = "v" * 9  # synthetic
    adapter = type("A", (), {"api_key": lambda self: password})()
    monkeypatch.setattr(bridge_gate, "_GATES", make(legacy_env(*SETTABLE), Settings({VAULT.setting_key: False})))
    monkeypatch.setattr(providers, "get_service", lambda: type("S", (), {"config": None})())
    monkeypatch.setattr(providers, "get_adapter", lambda _config, _id: adapter if _id == "vaultwarden" else None)
    app = FastAPI()
    app.include_router(providers.router)
    client = TestClient(app)
    url = "/v1/providers/vaultwarden/password"
    for header in (None, "wrong", DERIVED[BROKER.label], DERIVED[GIT_CREDENTIAL.label], *LEGACY.values()):
        response = client.get(url, headers={"X-Bridge-Token": header} if header else {})
        assert response.status_code == 401 and password not in response.text
    ok = client.get(url, headers={"X-Bridge-Token": DERIVED[VAULT.label]})
    assert ok.status_code == 200 and ok.json() == {"password": password}
    assert ok.headers["cache-control"] == "no-store"
    adapter.api_key = lambda: None
    assert client.get(url, headers={"X-Bridge-Token": DERIVED[VAULT.label]}).status_code == 503


def test_generic_token_broker_never_serves_the_vault_password(monkeypatch):
    """Codex: /v1/providers/vaultwarden/token must not bypass the vault gate."""
    from llm_bawt.service.providers.vaultwarden import VaultwardenAdapter
    from llm_bawt.service.routes import providers

    password = "v" * 9  # synthetic
    adapter = object.__new__(VaultwardenAdapter)
    adapter.api_key = lambda: password
    monkeypatch.setattr(bridge_gate, "_GATES", make({}))  # BROKER left open
    monkeypatch.setattr(providers, "get_service", lambda: type("S", (), {"config": None})())
    monkeypatch.setattr(providers, "get_adapter", lambda _config, _id: adapter if _id == "vaultwarden" else None)
    app = FastAPI()
    app.include_router(providers.router)
    client = TestClient(app)
    for header in (None, DERIVED[BROKER.label], DERIVED[VAULT.label]):
        response = client.get("/v1/providers/vaultwarden/token", headers={"X-Bridge-Token": header} if header else {})
        assert response.status_code in (401, 404) and password not in response.text


def test_present_empty_header_is_refused_on_an_open_gate(monkeypatch):
    """Opted-out gate: an absent header passes, an empty one is BAD (401)."""
    from llm_bawt.service.routes import providers

    gates = make(OPT_OUT)
    monkeypatch.setattr(bridge_gate, "_GATES", gates)
    monkeypatch.setattr(providers, "get_service", lambda: type("S", (), {"config": None})())
    monkeypatch.setattr(providers, "get_adapter", lambda _config, _id: None)
    app = FastAPI()
    app.include_router(providers.router)
    client = TestClient(app)
    url = "/v1/providers/github/git-credential"
    assert client.get(url).status_code == 404  # missing, gate open: passes to the route
    assert client.get(url, headers={"X-Bridge-Token": ""}).status_code == 401
    assert gates.counts(GIT_CREDENTIAL)["bad"] == 1


_GATE_NAMES = {"x-bridge-token", "bridge_claude_token_secret", "bridge_git_credential_token"}


# --- vault: enforced in code (TASK-2386) -----------------------------------
@pytest.mark.parametrize("settings", [
    Settings(),  # setting absent
    Settings({VAULT.setting_key: False}),
    Settings({VAULT.setting_key: "false"}),
    Settings(fail=True),  # settings store unreadable
], ids=["absent", "false", "non-bool", "unreadable"])
def test_vault_refuses_missing_and_bad_whatever_the_settings(settings):
    gates = make({DEFAULT_ENFORCED_ENV: "0"}, settings)
    assert gates.enforced(VAULT)
    assert refused(gates, VAULT, None)
    assert refused(gates, VAULT, "wrong")
    assert not refused(gates, VAULT, DERIVED[VAULT.label])


def test_vault_refuses_after_its_setting_row_is_deleted():
    settings = Settings({VAULT.setting_key: True})
    gates = make(settings=settings, clock=Clock())
    assert refused(gates, VAULT, None)
    settings.values.clear()
    gates._cached_at = None  # force a re-read
    assert refused(gates, VAULT, None)


def test_vault_accepts_no_legacy_or_other_gate_token():
    gates = make(legacy_env(*SETTABLE))
    for header in [*LEGACY.values(), DERIVED[BROKER.label], DERIVED[GIT_CREDENTIAL.label], ""]:
        assert refused(gates, VAULT, header)


def test_vault_fails_closed_when_the_root_cannot_derive():
    def broken(_label):
        raise RuntimeError("no root")

    gates = make(derive=broken)
    assert refused(gates, VAULT, None)
    assert refused(gates, VAULT, DERIVED[VAULT.label])


def test_vault_health_row_is_enforced():
    row = _row(make(), VAULT)
    assert row["enforced"] is True and row["status"] == "ok"


def test_vault_route_401_over_http_with_setting_absent_false_deleted(monkeypatch):
    settings = Settings()
    monkeypatch.setattr(bridge_gate, "_GATES", BridgeGates(
        settings, environ={}, derive=DERIVED.__getitem__, clock=Clock(), settings_ttl=0))
    app = FastAPI()

    @app.get("/vault")
    def vault(request: bridge_gate.Request):
        bridge_gate.check(request, VAULT)
        return {"ok": True}

    client = TestClient(app)
    for values in ({}, {VAULT.setting_key: False}, {}):
        settings.values = dict(values)
        assert client.get("/vault").status_code == 401
    assert client.get("/vault", headers={bridge_gate.HEADER: DERIVED[VAULT.label]}).status_code == 200


def _gate_name_literals(tree: ast.AST) -> list[int]:
    """Line numbers of non-docstring string constants naming a gate header/env."""
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings and node.value.strip().lower() in _GATE_NAMES
    ]


def test_no_route_compares_the_bridge_header_outside_bridge_gate():
    """Guard: only bridge_gate names the gate header or its env vars.

    Matches string constants rather than lines, so an alias
    (``NAME = "X-Bridge-Token"; request.headers.get(NAME)``) is caught too.
    """
    root = Path(bridge_gate.__file__).resolve().parents[1]  # src/llm_bawt
    offenders = []
    for path in root.rglob("*.py"):
        if path == Path(bridge_gate.__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        # Parse only files that mention a gate name at all (keeps this fast).
        if not any(name in source.lower() for name in _GATE_NAMES):
            continue
        offenders += [f"{path.relative_to(root)}:{line}" for line in _gate_name_literals(ast.parse(source))]
    assert offenders == []


def test_guard_catches_an_aliased_header_read():
    src = 'NAME = "X-Bridge-Token"\ndef f(request):\n    """X-Bridge-Token"""\n    return request.headers.get(NAME)\n'
    assert _gate_name_literals(ast.parse(src)) == [1]
