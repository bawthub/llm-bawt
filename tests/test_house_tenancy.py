"""TASK-2118: a bot moves itself into a free house and out of its own.

The decision logic runs on a SQLite stand-in for the profile store; the real
store, the shared assignment lock and the races are in test_house_tenancy_pg.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt import house_lease, house_tenancy
from llm_bawt.house_tenancy import (
    HouseMoveRefused, HouseState, list_houses, move_in, move_out, read_house_state,
)


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

    def set(self, slug, **fields):
        with self.engine.begin() as conn:
            for k, v in fields.items():
                conn.execute(text(f"UPDATE bot_profiles SET {k}=:v WHERE slug=:s"), {"v": v, "s": slug})

    def house(self, slug):
        return self.get(slug).house


class FakeRedis:
    def __init__(self, groups=None, **values):
        self.values = {k.replace("__", ":"): v for k, v in values.items()}
        self.groups = groups or {}
        self.sets = []

    def exists(self, key):
        return key in self.groups

    def xinfo_groups(self, stream):
        return self.groups[stream]

    def scan_iter(self, match):
        import fnmatch

        return [k for k in self.values if fnmatch.fnmatchcase(k, match)]

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.values[key] = value


POOL = {"h1": "claude-code", "h2": "claude-code", "h3": "claude-code", "c1": "codex"}


def state(**overrides):
    base = HouseState(leases={"h1": {"claude-code": "5"}, "h2": {"claude-code": "7"},
                              "c1": {"codex": "1"}, "h9": {"claude-code": "1"}},
                      pool=POOL)
    return lambda: HouseState(**{**base.__dict__, **overrides})


LIVE = state()


def test_read_house_state_reads_leases_recycle_markers_and_running_turns():
    client = FakeRedis(**{
        "agent:house:h1:lease:claude-code": "5", "agent:house:h1:epoch:claude-code": "5",
        "agent:house:h2:recycle:claude-code": "7", "agent:house:BAD!:lease:x": "1",
        "agent:turn:req:r1": "agent:commands:house:h2", "agent:turn:req:r2": "agent:commands",
    })
    got = read_house_state(client, [{"id": "h1", "harness": "claude-code"}])
    assert got.leases == {"h1": {"claude-code": "5"}}
    assert got.recycling == {("h2", "claude-code"): "7"}
    assert got.busy_streams == {"agent:commands:house:h2", "agent:commands"}
    assert got.pool == {"h1": "claude-code"}
    assert read_house_state(client).pool is None


@pytest.mark.parametrize("group,busy", [
    ({"lag": 0, "pending": 0}, False),
    ({"lag": 1, "pending": 0}, True),  # routed, nobody has read it yet
    ({"lag": 0, "pending": 1}, True),  # read, no turn record yet
    ({"lag": None, "pending": 0}, True),  # unknown lag is never drained
])
def test_an_unread_or_unacked_send_holds_the_house(group, busy):
    stream = "agent:commands:house:h1"
    client = FakeRedis(groups={stream: [group]}, **{"agent:house:h1:lease:claude-code": "5"})
    got = read_house_state(client, [{"id": "h1", "harness": "claude-code"}])
    assert got.status("h1", "claude-code", []) == ("draining" if busy else "free")


@pytest.mark.parametrize("overrides,house,status", [
    ({}, "h1", "free"),
    ({}, "c1", "incompatible"),  # configured for codex
    ({}, "h3", "down"),  # configured, no bridge
    ({}, "h9", "unconfigured"),  # a live bridge outside the pool
    ({"pool": None}, "h1", "unconfigured"),
    ({"recycling": {("h1", "claude-code"): "5"}}, "h1", "recycling"),
    ({"recycling": {("h1", "claude-code"): "4"}}, "h1", "free"),  # restarted since
    ({"busy_streams": frozenset({"agent:commands:house:h1"})}, "h1", "draining"),
])
def test_status(overrides, house, status):
    assert state(**overrides)().status(house, "claude-code", []) == status


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
    assert refused.value.code == "occupied"
    assert store.house("alpha") == "h1" and store.house("beta") is None
    assert move_in(store, "beta", None, LIVE).house == "h2"


@pytest.mark.parametrize("overrides,code", [
    ({"recycling": {("h1", "claude-code"): "5"}}, "recycling"),
    ({"busy_streams": frozenset({"agent:commands:house:h1"})}, "draining"),
    ({"leases": {"h2": {"claude-code": "7"}}}, "house_unavailable"),
])
def test_a_house_that_is_not_free_is_refused_and_skipped(overrides, code):
    store = Store(alpha={})
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "alpha", "h1", state(**overrides))
    assert refused.value.code == code and store.writes == []
    assert move_in(store, "alpha", None, state(**overrides)).house == "h2"


def test_only_configured_compatible_houses_are_claimable():
    store = Store(alpha={})
    for house, code in (("h9", "unconfigured"), ("c1", "incompatible"), ("h3", "house_unavailable")):
        with pytest.raises(HouseMoveRefused) as refused:
            move_in(store, "alpha", house, LIVE)
        assert refused.value.code == code
    with pytest.raises(HouseMoveRefused) as none:
        move_in(store, "alpha", None, state(pool=None))
    assert none.value.code == "no_pool" and store.writes == []
    # Any-free never picks the live but unconfigured h9.
    only_h9 = state(leases={"h9": {"claude-code": "1"}})
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "alpha", None, only_h9)
    assert refused.value.code == "no_free_house"


def test_no_free_house_keeps_the_caller_where_it_is():
    store = Store(alpha={"house": "h1", "pinned": True}, beta={"house": "h2"}, gamma={})
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "gamma", None, LIVE)
    assert refused.value.code == "no_free_house"
    assert store.writes == [] and store.house("gamma") is None


def test_already_housed_is_idempotent_or_keeps_its_house():
    store = Store(alpha={"house": "h1"})
    assert move_in(store, "alpha", "h1", LIVE).changed is False
    assert move_in(store, "alpha", None, LIVE).house == "h1"  # retry leaks nothing
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "alpha", "h2", LIVE)  # A->B: refused, A kept
    assert refused.value.code == "already_housed"
    assert store.writes == [] and store.house("alpha") == "h1"


@pytest.mark.parametrize("fields,backend", [
    ({"backend": "codex"}, "codex"),
    ({"bot_type": "chat", "backend": None}, "chat"),
])
def test_unsupported_runtimes_are_refused_both_ways(fields, backend):
    store = Store(bot=fields)
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "bot", "h1", LIVE)
    assert refused.value.code == "unsupported_harness" and refused.value.status_code == 403
    assert backend in str(refused.value)
    assert list_houses(store, "bot", LIVE, frozenset())["you"]["eligible"] is False
    store.set("bot", house="h1")  # a manual tenancy can't be self-released either
    with pytest.raises(HouseMoveRefused) as out:
        move_out(store, "bot")
    assert out.value.code == "unsupported_harness" and store.house("bot") == "h1"
    assert store.writes == []


def test_eligibility_is_the_runtime_not_the_name():
    store = Store(codex={"backend": "claude-code"})  # e.g. a GPT model on claude-code
    assert move_in(store, "codex", "h1", LIVE).house == "h1"


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
    assert move_out(store, "beta").changed is False  # idempotent in the mansion


def test_a_pinned_bot_may_leave_and_its_pin_goes_with_the_tenancy():
    store = Store(alpha={"house": "h1", "pinned": True}, beta={"house": "h2", "pinned": True})
    move_out(store, "alpha")
    assert store.get("alpha").house is None and not store.get("alpha").house_pinned
    assert store.writes[-1]["expected_fields"] == {"house": "h1", "house_pinned": True}
    assert store.get("beta").house_pinned  # nobody else's pin


def test_list_shows_status_tenants_and_masks_private_bots():
    store = Store(alpha={"house": "h1", "pinned": True}, beta={}, shy={"house": "h2"},
                  gone={"house": "h7"})
    listed = list_houses(store, "beta", LIVE, frozenset({"shy"}))
    assert listed["you"] == {"bot": "beta", "house": None, "eligible": True}
    by_id = {h["house"]: h for h in listed["houses"]}
    assert listed["pool_configured"] is True
    assert by_id["h1"] == {"house": "h1", "configured": True, "harness": "claude-code",
                           "live": ["claude-code"],
                           "tenants": [{"bot": "alpha", "pinned": True}], "status": "occupied"}
    assert by_id["h2"]["tenants"] == [{"bot": house_tenancy.PRIVATE_TENANT}]
    assert by_id["c1"]["status"] == "incompatible"  # a codex house
    assert by_id["h3"]["status"] == "down"  # configured, listed though not live
    assert (by_id["h9"]["configured"], by_id["h9"]["status"]) == (False, "unconfigured")
    assert by_id["h7"] == {"house": "h7", "configured": False, "harness": None, "live": [],
                           "tenants": [{"bot": "gone"}], "status": "occupied"}
    # A private caller sees itself.
    mine = {h["house"]: h for h in list_houses(store, "shy", LIVE, frozenset({"shy"}))["houses"]}
    assert mine["h2"]["tenants"] == [{"bot": "shy"}]


@pytest.fixture
def recycler(monkeypatch):
    """request_recycle against a Store's engine and a fake Redis."""
    from llm_bawt.utils import db

    client = FakeRedis(**{"agent:house:h1:lease:claude-code": "5"})
    client.xadd = lambda stream, fields: client.sets.append(("xadd", stream, None))
    monkeypatch.setattr(house_lease, "_publish", lambda config, action: action(client))
    holder = SimpleNamespace(store=None)
    monkeypatch.setattr(db, "get_shared_engine", lambda config: holder.store.engine)
    def reader():
        return read_house_state(client, [{"id": "h1", "harness": "claude-code"}])

    return SimpleNamespace(client=client, holder=holder, read=reader)


def test_recycle_request_marks_the_epoch_before_asking(recycler):
    recycler.holder.store = Store(alpha={})
    assert house_lease.request_recycle(None, "h1") == ["claude-code"]
    assert recycler.client.sets[0] == ("agent:house:h1:recycle:claude-code", "5",
                                       house_tenancy.RECYCLE_MARKER_TTL_S)
    assert recycler.client.sets[1][0] == "xadd"
    assert recycler.read().status("h1", "claude-code", []) == "recycling"
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(recycler.holder.store, "alpha", "h1", recycler.read)
    assert refused.value.code == "recycling"


def test_a_claim_between_release_and_recycle_cancels_the_recycle(recycler):
    # Release, a claim lands, then the releaser's recycle runs: it must not
    # restart the new tenant's house.
    store = recycler.holder.store = Store(alpha={"house": "h1"}, beta={})
    assert move_out(store, "alpha").vacated_empty is True
    assert move_in(store, "beta", "h1", recycler.read).house == "h1"
    assert house_lease.request_recycle(None, "h1") == []
    assert recycler.client.sets == []


def test_unverifiable_tenancy_skips_the_recycle(recycler, monkeypatch):
    from llm_bawt.utils import db

    def broken(config):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_shared_engine", broken)
    assert house_lease.request_recycle(None, "h1") == []
    assert recycler.client.sets == []


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
    monkeypatch.setattr(routes, "_state", LIVE)
    monkeypatch.setattr(house_tenancy, "_private_bots", lambda store: frozenset())
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

    return SimpleNamespace(call=call, store=store, calls=calls, routes=routes)


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
    assert api.calls == [("refresh", "alpha")]  # failures publish nothing
    listed = api.call("GET", "", "alpha").json()
    assert listed["you"]["house"] == "h2"


def test_route_move_out_recycles_an_emptied_house(api):
    out = api.call("POST", "/move-out", "beta")
    assert out.status_code == 200 and out.json()["vacated"] == "h1"
    assert api.calls == [("refresh", "beta"), ("recycle", "h1")]
    again = api.call("POST", "/move-out", "beta").json()
    assert again["changed"] is False and len(api.calls) == 2


def test_route_refuses_when_redis_is_unreadable(api, monkeypatch):
    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(api.routes, "_state", down)
    response = api.call("POST", "/move-in", "alpha", json={})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "unavailable"
    assert api.store.writes == []


def test_list_and_move_in_carry_each_houses_preview_url_and_branch(api, monkeypatch):
    previews = {"h1": {"preview_host": "https://fixture-one.example", "preview": None},
                "h2": {"preview_host": "https://fixture-three.example",
                       "preview": {"up": True, "branch": "fixture/TASK-1", "commit": "abc"}}}
    monkeypatch.setattr(api.routes, "_previews",
                        lambda houses: {h: previews[h] for h in houses if h in previews})

    listed = {row["house"]: row for row in api.call("GET", "", "alpha").json()["houses"]}
    assert listed["h2"]["preview_host"] == "https://fixture-three.example"
    assert listed["h2"]["preview"]["branch"] == "fixture/TASK-1"
    assert listed["h1"]["preview"] is None
    moved = api.call("POST", "/move-in", "alpha", json={}).json()
    assert moved["house"] == "h2" and moved["preview_host"] == "https://fixture-three.example"


def test_an_unreadable_preview_still_lists_houses(api, monkeypatch):
    def down(_houses):
        raise RuntimeError("pool unreadable")

    monkeypatch.setattr(api.routes, "_previews", down)
    response = api.call("GET", "", "alpha")
    assert response.status_code == 200 and response.json()["houses"]
