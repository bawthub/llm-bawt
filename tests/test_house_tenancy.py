"""TASK-2118: a bot moves itself into a free house and out of its own.

The decision logic runs on a SQLite stand-in for the profile store; the real
store, its advisory lock and a two-bot race are in test_house_tenancy_pg.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt import house_tenancy
from llm_bawt.house_tenancy import HouseMoveRefused, live_leases, list_houses, move_in, move_out


class Store:
    """Just enough of BotProfileStore: get, upsert on a borrowed session, engine."""

    def __init__(self, **bots):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        self.writes = []
        with self.engine.begin() as conn:
            conn.execute(text("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, bot_type TEXT, "
                              "agent_backend TEXT, house TEXT, house_pinned BOOLEAN)"))
            for slug, fields in bots.items():
                conn.execute(text("INSERT INTO bot_profiles VALUES (:slug, :bot_type, :backend, "
                                  ":house, :pinned)"), {
                    "slug": slug, "bot_type": fields.get("bot_type", "agent"),
                    "backend": fields.get("backend", "claude-code"),
                    "house": fields.get("house"), "pinned": fields.get("pinned", False)})

    def get(self, slug):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM bot_profiles WHERE slug=:s"), {"s": slug}).first()
        return None if row is None else SimpleNamespace(**row._mapping)

    def upsert(self, payload, *, session, require_existing):
        assert require_existing
        self.writes.append(payload)
        session.connection().execute(
            text("UPDATE bot_profiles SET house=:h, house_pinned=:p WHERE slug=:s"),
            {"h": payload["house"], "p": payload["house_pinned"], "s": payload["slug"]})

    def house(self, slug):
        return self.get(slug).house


def leases(*keys):
    return lambda pattern: [f"agent:house:{k}" for k in keys]


LIVE = leases("h1:lease:claude-code", "h2:lease:claude-code", "c1:lease:codex",
              "h3:epoch:claude-code")


def test_live_leases_reads_only_lease_keys():
    assert live_leases(LIVE) == {"h1": {"claude-code"}, "h2": {"claude-code"}, "c1": {"codex"}}


def test_named_move_in_claims_a_free_house_and_never_pins():
    store = Store(alpha={})
    result = move_in(store, "alpha", "h1", LIVE)
    assert (result.house, result.changed) == ("h1", True)
    assert store.house("alpha") == "h1"
    (write,) = store.writes
    assert write["house_pinned"] is False and write["_changed_by"] == "bot:alpha"
    assert write["expected_fields"] == {"house": None}
    assert result.as_dict()["effective"].startswith("next turn")


def test_occupied_house_is_refused_and_any_free_skips_it():
    store = Store(alpha={"house": "h1"}, beta={})
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "beta", "h1", LIVE)
    assert refused.value.code == "occupied" and refused.value.extra["tenants"] == ["alpha"]
    assert store.house("alpha") == "h1" and store.house("beta") is None
    assert move_in(store, "beta", None, LIVE).house == "h2"


def test_pinned_tenant_also_blocks_and_no_free_house_is_refused():
    store = Store(alpha={"house": "h1", "pinned": True}, beta={"house": "h2"}, gamma={})
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "gamma", None, LIVE)
    assert refused.value.code == "no_free_house"
    assert store.writes == []


def test_house_without_a_live_lease_for_the_backend_is_refused():
    store = Store(alpha={})
    for house in ("c1", "h3", "h9"):  # codex-only lease, epoch only, unknown
        with pytest.raises(HouseMoveRefused) as refused:
            move_in(store, "alpha", house, LIVE)
        assert refused.value.code == "house_unavailable"


def test_already_housed_is_idempotent_or_must_move_out_first():
    store = Store(alpha={"house": "h1"})
    assert move_in(store, "alpha", "h1", LIVE).changed is False
    assert move_in(store, "alpha", None, LIVE).house == "h1"
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "alpha", "h2", LIVE)
    assert refused.value.code == "already_housed"
    assert store.writes == []


@pytest.mark.parametrize("fields,backend", [
    ({"backend": "codex"}, "codex"),
    ({"bot_type": "chat", "backend": None}, "chat"),
])
def test_codex_and_chat_bots_are_refused(fields, backend):
    store = Store(bot=fields)
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "bot", "h1", LIVE)
    assert refused.value.code == "unsupported_harness" and refused.value.status_code == 403
    assert backend in str(refused.value)
    # A manually assigned tenancy can't be self-released either.
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET house='h1' WHERE slug='bot'"))
    with pytest.raises(HouseMoveRefused) as out:
        move_out(store, "bot")
    assert out.value.code == "unsupported_harness" and store.house("bot") == "h1"
    assert store.writes == []
    listed = list_houses(store, "bot", LIVE)
    assert listed["you"]["eligible"] is False


def test_invalid_house_and_unknown_bot():
    store = Store(alpha={})
    with pytest.raises(HouseMoveRefused) as bad:
        move_in(store, "alpha", "H 1!", LIVE)
    assert bad.value.code == "invalid_house"
    with pytest.raises(HouseMoveRefused) as missing:
        move_in(store, "ghost", "h1", LIVE)
    assert missing.value.status_code == 404


def test_move_out_releases_only_the_callers_house():
    store = Store(alpha={"house": "h1"}, beta={"house": "h1"})
    result = move_out(store, "alpha")
    assert (result.changed, result.vacated, result.vacated_empty) == (True, "h1", False)
    assert store.house("alpha") is None and store.house("beta") == "h1"
    assert store.writes[-1]["expected_fields"] == {"house": "h1", "house_pinned": False}
    last = move_out(store, "beta")
    assert (last.vacated, last.vacated_empty) == ("h1", True)
    assert move_out(store, "beta").changed is False


def test_pinned_bot_cannot_move_itself_out():
    store = Store(alpha={"house": "h1", "pinned": True})
    with pytest.raises(HouseMoveRefused) as refused:
        move_out(store, "alpha")
    assert refused.value.code == "pinned" and store.house("alpha") == "h1"


def test_list_shows_live_houses_tenants_and_what_is_free_for_the_caller():
    store = Store(alpha={"house": "h1", "pinned": True}, beta={}, gone={"house": "h7"})
    listed = list_houses(store, "beta", LIVE)
    assert listed["you"] == {"bot": "beta", "house": None, "eligible": True}
    by_id = {h["house"]: h for h in listed["houses"]}
    assert by_id["h1"] == {"house": "h1", "live": ["claude-code"],
                           "tenants": [{"bot": "alpha", "pinned": True}], "free": False}
    assert by_id["h2"]["free"] is True
    assert by_id["c1"]["free"] is False  # codex bridge only
    assert by_id["h7"] == {"house": "h7", "live": [], "tenants": [{"bot": "gone"}], "free": False}


def test_tools_take_no_bot_argument():
    import inspect

    from llm_bawt.mcp_server.house_tools import self_house

    assert set(inspect.signature(self_house).parameters) == {"action", "house"}
    for fn in (move_in, move_out, list_houses):
        # The route passes only the signed caller's slug.
        assert list(inspect.signature(fn).parameters)[:2] == ["store", "slug"]
    assert house_tenancy.SELF_MOVE_BACKENDS == {"claude-code"}


# ------------------------------------------------------------------ route


@pytest.fixture
def api(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt import house_lease
    from llm_bawt.service.routes import houses as routes
    from llm_bawt.task_turn_context import TaskTurnContextError

    store = Store(alpha={}, beta={"house": "h1"})
    calls = []

    def open_turn(token):
        if not (token or "").startswith("turn:"):
            raise TaskTurnContextError("no trusted turn")
        return SimpleNamespace(bot_id=token.removeprefix("turn:"))

    monkeypatch.setattr(routes, "open_task_turn_context", open_turn)
    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_scan", LIVE)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config="cfg"))
    monkeypatch.setattr(house_lease, "_refresh_bot",
                        lambda service, llm, slug: calls.append(("refresh", slug)))
    monkeypatch.setattr(house_lease, "request_recycle",
                        lambda config, house: calls.append(("recycle", house)))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    def call(method, path, bot=None, **kw):
        headers = {"X-LLM-Bawt-Task-Turn-Context": f"turn:{bot}"} if bot else {}
        return client.request(method, f"/v1/houses/self{path}", headers=headers, **kw)

    return SimpleNamespace(call=call, store=store, calls=calls)


def test_route_needs_a_signed_turn(api):
    for method, path in (("GET", ""), ("POST", "/move-in"), ("POST", "/move-out")):
        response = api.call(method, path, json={})
        assert response.status_code == 403 and response.json()["detail"]["code"] == "unauthorized"
    assert api.store.writes == []


def test_route_moves_only_the_signed_caller(api):
    # The body can't name another bot: unknown keys are refused.
    assert api.call("POST", "/move-in", "alpha", json={"bot_id": "beta"}).status_code == 422
    refused = api.call("POST", "/move-in", "alpha", json={"house": "h1"})
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "occupied"
    moved = api.call("POST", "/move-in", "alpha", json={})
    assert moved.status_code == 200 and moved.json()["house"] == "h2"
    assert api.calls == [("refresh", "alpha")]
    listed = api.call("GET", "", "alpha").json()
    assert listed["you"]["house"] == "h2"


def test_route_move_out_recycles_an_emptied_house(api):
    out = api.call("POST", "/move-out", "beta")
    assert out.status_code == 200 and out.json()["vacated"] == "h1"
    assert api.calls == [("refresh", "beta"), ("recycle", "h1")]
    again = api.call("POST", "/move-out", "beta").json()
    assert again["changed"] is False and len(api.calls) == 2
