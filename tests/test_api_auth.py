"""TASK-2380: per-caller API keys, fail-closed mode, middleware."""

import asyncio

import pytest

from llm_bawt.api_auth import keys
from llm_bawt.api_auth.middleware import ApiAuthMiddleware, Observations, Principal
from llm_bawt.api_auth.mode import ModeResolver

ROOT = lambda: b"r" * 32  # noqa: E731


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Store:
    def __init__(self, values=None):
        self.values, self.fail = values or {}, False

    def __call__(self):
        if self.fail:
            raise RuntimeError("db down")
        return self.values


# --- keys -----------------------------------------------------------------
def test_key_roundtrip_and_rejections():
    good = keys.header_value("bawthub-web", root=ROOT)
    assert keys.verify(good, root=ROOT).principal == "bawthub-web"
    assert keys.verify(good, denylist=["bawthub-web"], root=ROOT).reason == "revoked"
    assert keys.verify("bawthub-web." + "0" * 64, root=ROOT).reason == "invalid"
    assert keys.verify("nodot", root=ROOT).reason == "malformed"
    assert keys.verify("Bad_Name.x", root=ROOT).reason == "malformed"
    other = keys.header_value("house", root=ROOT).split(".", 1)[1]
    assert not keys.verify(f"bawthub-web.{other}", root=ROOT).ok


# --- mode -----------------------------------------------------------------
def test_cold_start_with_db_down_stays_closed_even_with_env_off():
    store = Store()
    store.fail = True
    r = ModeResolver(store, environ={"LLM_BAWT_API_AUTH_MODE": "off"}, clock=Clock())
    assert r.mode() == "unresolved"


def test_first_read_adopts_stored_mode_and_env_default_only_when_absent():
    clock = Clock()
    assert ModeResolver(Store({"api_auth_mode": "log"}), environ={}, clock=clock).mode() == "log"
    assert ModeResolver(Store({}), environ={"LLM_BAWT_API_AUTH_MODE": "log"}, clock=clock).mode() == "log"
    assert ModeResolver(Store({}), environ={}, clock=clock).mode() == "off"
    assert ModeResolver(Store({"api_auth_mode": "bogus"}), environ={}, clock=clock).mode() == "enforce"


def test_warm_failure_keeps_enforce_then_recovery_and_rollback():
    clock, store = Clock(), Store({"api_auth_mode": "enforce"})
    r = ModeResolver(store, environ={"LLM_BAWT_API_AUTH_MODE": "off"}, clock=clock)
    assert r.mode() == "enforce"
    store.fail = True
    clock.t = 60
    assert r.mode() == "enforce"
    store.fail, store.values = False, {"api_auth_mode": "off"}
    clock.t = 66
    assert r.mode() == "off"  # explicit rollback within one TTL


def test_force_override():
    store = Store()
    store.fail = True
    r = ModeResolver(store, environ={"LLM_BAWT_API_AUTH_MODE_FORCE": "off"}, clock=Clock())
    assert r.mode() == "off"


# --- middleware -----------------------------------------------------------
def _run(mw, path="/v1/chat/completions", headers=(), kind="http", method="POST"):
    sent, reached = [], []

    async def app(scope, receive, send):
        reached.append(scope.get("state", {}).get("api_principal"))

    async def send(msg):
        sent.append(msg)

    mw.app = app

    scope = {"type": kind, "path": path, "method": method, "client": ("10.0.0.5", 1),
             "headers": [(k.encode(), v.encode()) for k, v in headers]}
    asyncio.run(mw(scope, None, send))
    return sent, reached


def _mw(mode, checkers=(), obs=None):
    store = Store({"api_auth_mode": mode} if mode else {})
    if mode is None:
        store.fail = True
    return ApiAuthMiddleware(None, resolver=ModeResolver(store, environ={}, clock=Clock()),
                             checkers=checkers, root=ROOT, observations=obs)


def _status(sent):
    return sent[0].get("status") or sent[0].get("code")


@pytest.mark.parametrize("mode", ["log", "enforce"])
def test_valid_key_admitted(mode):
    sent, reached = _run(_mw(mode), headers=[("x-llm-bawt-key", keys.header_value("house", root=ROOT))])
    assert not sent and reached == [Principal("house", "key")]


def test_missing_credentials_log_admits_and_counts_enforce_refuses():
    obs = Observations()
    _, reached = _run(_mw("log", obs=obs))
    assert reached == [None] and obs.snapshot()[0]["key"][0] == "anon"
    sent, reached = _run(_mw("enforce"))
    assert _status(sent) == 401 and not reached


@pytest.mark.parametrize("mode", ["log", "enforce"])
def test_bad_or_repeated_key_refused(mode):
    sent, _ = _run(_mw(mode), headers=[("x-llm-bawt-key", "house." + "0" * 64)])
    assert _status(sent) == 401
    good = keys.header_value("house", root=ROOT)
    sent, _ = _run(_mw(mode), headers=[("x-llm-bawt-key", good), ("x-llm-bawt-key", good)])
    assert _status(sent) == 401


def test_off_passes_everything_including_bad_keys():
    sent, reached = _run(_mw("off"), headers=[("x-llm-bawt-key", "garbage")])
    assert not sent and len(reached) == 1


def test_unresolved_503_but_allowlist_and_preflight_open():
    sent, _ = _run(_mw(None))
    assert _status(sent) == 503
    assert _run(_mw(None), path="/health", method="GET")[1]
    assert _run(_mw(None), method="OPTIONS")[1]
    assert _run(_mw("enforce"), path="/integrations/google-home/token")[1]
    assert _status(_run(_mw("enforce"), path="/admin/nextcloud-talk/x")[0]) == 401


def test_websocket_refused_with_4401():
    sent, reached = _run(_mw("enforce"), kind="websocket")
    assert sent == [{"type": "websocket.close", "code": 4401}] and not reached


def test_turn_capability_wins_over_key_and_legacy_checkers_admit():
    bot = lambda h: Principal("bot:al", "bot") if b"x-llm-bawt-task-turn-context" in h else None  # noqa: E731
    svc = lambda h: Principal("bawthub-service", "service") if b"authorization" in h else None  # noqa: E731
    mw = _mw("enforce", checkers=[bot, svc])
    _, reached = _run(mw, headers=[("x-llm-bawt-task-turn-context", "t"),
                                   ("x-llm-bawt-key", keys.header_value("house", root=ROOT))])
    assert reached == [Principal("bot:al", "bot")]
    _, reached = _run(mw, headers=[("authorization", "Bearer x")])
    assert reached == [Principal("bawthub-service", "service")]


def test_observations_cap_and_age():
    clock = Clock()
    obs = Observations(cap=2, max_age=10, clock=clock)
    for i in range(3):
        obs.record(("anon", str(i)))
    assert {tuple(r["key"]) for r in obs.snapshot()} == {("anon", "0"), ("anon", "1"), ("<overflow>",)}
    clock.t = 20
    assert obs.snapshot() == []


def test_resolver_custom_setting_key():
    from llm_bawt.api_auth.mode import ModeResolver
    r = ModeResolver(lambda: {"mcp_auth_mode": "enforce", "api_auth_mode": "off"},
                     environ={}, setting_key="mcp_auth_mode")
    assert r.mode() == "enforce"


def test_wiring_checkers_ignore_missing_headers():
    from llm_bawt.api_auth import wiring
    assert wiring.turn_capability({}) is None
    assert wiring.bridge_token({}) is None


def test_app_installs_api_auth_middleware():
    from llm_bawt.api_auth.middleware import ApiAuthMiddleware
    from llm_bawt.service.api import app
    assert any(m.cls is ApiAuthMiddleware for m in app.user_middleware)


def test_per_instance_key_policy():
    from llm_bawt.api_auth.middleware import ApiAuthMiddleware, Principal
    from llm_bawt.api_auth import keys
    root = lambda: b"r" * 32
    resolver = type("R", (), {"denylist": lambda self: frozenset(), "mode": lambda self: "log"})()
    bot = lambda h: Principal("vex", "bot") if b"x-llm-bawt-task-turn-context" in h else None

    def scope(*hdrs):
        return {"type": "http", "headers": list(hdrs)}

    mw = ApiAuthMiddleware(None, resolver=resolver, checkers=(bot,), root=root,
                           key_principal_allowed=lambda p: p == "llm-bawt-app",
                           refuse_turn_with_key=True)
    good = (b"x-llm-bawt-key", keys.header_value("llm-bawt-app", root=root).encode())
    other = (b"x-llm-bawt-key", keys.header_value("bot-vex", root=root).encode())
    turn = (b"x-llm-bawt-task-turn-context", b"t")
    assert mw.authenticate(scope(good))[0].name == "llm-bawt-app"
    assert mw.authenticate(scope(other)) == (None, "key_not_permitted")
    assert mw.authenticate(scope(good, turn)) == (None, "ambiguous_credentials")
    assert mw.authenticate(scope(turn))[0].kind == "bot"


def test_garbled_auth_row_in_real_store_keeps_last_good_mode():
    """An undecodable api_auth_mode row must not read as absent (= default off)."""
    from types import SimpleNamespace

    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine, select

    from llm_bawt.api_auth.wiring import _GlobalSettings
    from llm_bawt.runtime_settings import RuntimeSetting

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with Session(engine) as s:
        s.add(RuntimeSetting(scope_type="global", scope_id="*", key="api_auth_mode", value_json='"enforce"'))
        s.commit()
    reader = _GlobalSettings()
    reader.store = SimpleNamespace(engine=engine)
    clock = Clock()
    r = ModeResolver(reader, environ={"LLM_BAWT_API_AUTH_MODE": "off"}, clock=clock)
    assert r.mode() == "enforce"
    with Session(engine) as s:
        row = s.exec(select(RuntimeSetting)).one()
        row.value_json = "{not json"
        s.add(row)
        s.commit()
    clock.t += 3600
    assert r.mode() == "enforce"
    # Cold start with the garbled row: unresolved (fail closed), never the off default.
    cold = ModeResolver(reader, environ={"LLM_BAWT_API_AUTH_MODE": "off"}, clock=Clock())
    assert cold.mode() != "off"
