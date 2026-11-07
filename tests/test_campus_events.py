"""TASK-2120: Campus location/house events and snapshot shaping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from llm_bawt import campus_events as ce
from llm_bawt.house_watcher import (DOWN, RECREATING, UNKNOWN, UP, HouseStates, frames_for,
                                    merge_flags)
from llm_bawt.service.routes.campus import house_rows

AGENT = {"bot_type": "agent", "agent_backend": "claude-code"}
LEASE_H1 = "agent:house:h1:lease:claude-code"


def fields(house=None, pinned=False, **over):
    return {**AGENT, "house": house, "house_pinned": pinned, **over}


# --- assignment frames -------------------------------------------------------

def test_move_in_and_out_report_routes_and_reasons():
    live = {LEASE_H1: True}
    move_in = ce.location_frame("b", fields(), fields("h1"), "", live)
    assert move_in == {"_type": "bot_location_changed", "kind": "assignment", "bot_id": "b",
                       "from": {"runtime": "claude-code-bridge"},
                       "to": {"runtime": "house", "house": "h1"}, "reason": "move_in",
                       "version": None}
    move_out = ce.location_frame("b", fields("h1"), fields(), "", live)
    assert move_out["reason"] == "move_out" and move_out["to"] == {"runtime": "claude-code-bridge"}


def test_named_reason_wins_and_route_reasons_are_not_accepted_for_writes():
    assert ce.location_frame("b", fields("h1"), fields(), "idle_evict", {})["reason"] == "idle_evict"
    assert ce.location_frame("b", fields("h1"), fields(), "house_up", {})["reason"] == "move_out"


def test_pin_only_is_pinned_change_with_same_house():
    frame = ce.location_frame("b", fields("h1"), fields("h1", pinned=True), "", {LEASE_H1: True})
    assert frame["reason"] == "pinned_change"
    assert frame["from"]["house"] == frame["to"]["house"] == "h1"


def test_assignment_to_dead_house_keeps_bridge_route_but_records_house():
    frame = ce.location_frame("b", fields(), fields("h1"), "", {LEASE_H1: False})
    assert frame["to"] == {"runtime": "claude-code-bridge", "house": "h1", "house_live": False}


def test_no_location_change_no_frame():
    assert ce.location_frame("b", fields(), fields(), "", {}) is None
    # Chat bots stay in the app whatever their house field says.
    chat = {"bot_type": "chat", "agent_backend": None, "house_pinned": False}
    assert ce.location_frame("d", {**chat, "house": None}, {**chat, "house": "h1"}, "", {}) is None


def test_create_and_delete_only_for_housed_profiles():
    assert ce.location_frame("b", {}, fields(), "", {}) is None
    created = ce.location_frame("b", {}, fields("h1"), "", {LEASE_H1: True})
    assert created["from"] is None and created["reason"] == "move_in"
    assert ce.location_frame("b", fields(), {}, "move_out", {}) is None
    deleted = ce.location_frame("b", fields("h1"), {}, "move_out", {LEASE_H1: True})
    assert deleted["to"] is None and deleted["reason"] == "move_out"


# --- commit binding ----------------------------------------------------------

@pytest.fixture
def sent(monkeypatch):
    out: list = []
    monkeypatch.setattr(ce, "emit_tenancy_changes", lambda pending, config=None: out.append(pending))
    return out


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    with Session(engine) as s:
        s.execute(text("select 1"))
        yield s


def test_frame_sent_only_after_commit(session, sent):
    ce.note_tenancy_change(session, "b", fields(), fields("h1"), None)
    assert sent == []
    session.commit()
    assert [p[0] for p in sent[0]] == ["b"]
    session.commit()  # listener removed: a later commit of this session sends nothing more
    assert len(sent) == 1


def test_rollback_and_noop_send_nothing(session, sent):
    ce.note_tenancy_change(session, "b", fields(), fields("h1"), None)
    session.rollback()
    session.commit()
    ce.note_tenancy_change(session, "b", fields("h1"), fields("h1"), None)
    session.commit()
    assert sent == []


def test_borrowed_session_batches_every_write_until_its_commit(session, sent):
    ce.note_tenancy_change(session, "a", fields(), fields("h1"), "move_in")
    ce.note_tenancy_change(session, "b", fields("h2"), fields(), "move_out")
    session.commit()
    assert [(p[0], p[3]) for p in sent[0]] == [("a", "move_in"), ("b", "move_out")]


def test_publish_stamps_shared_seq_identity_and_ts():
    class FakeRedis:
        def __init__(self):
            self.n, self.added = 0, []

        def incr(self, key):
            assert key == ce.SEQ_KEY
            self.n += 1
            return self.n

        def xadd(self, stream, data, **_):
            self.added.append((stream, data))

    client = FakeRedis()
    ce.publish(client, [{"_type": "x"}, {"_type": "y"}])
    import json
    frames = [json.loads(d["payload"]) for _, d in client.added]
    assert [f["seq"] for f in frames] == [1, 2]
    assert len({f["event_id"] for f in frames}) == 2 and all(f["ts"] for f in frames)
    assert {s for s, _ in client.added} == {"events:system:__config__"}


# --- house transitions -------------------------------------------------------

def test_lease_lifecycle_transitions():
    s = HouseStates()
    assert s.on_set("h1", "claude-code", "7") == [("h1", None, UP)]
    assert s.on_set("h1", "claude-code", "7") == []  # heartbeat
    assert s.on_gone("h1", "claude-code", released=True) == [("h1", UP, RECREATING)]
    assert s.on_set("h1", "claude-code", "8") == [("h1", RECREATING, UP)]
    assert s.on_gone("h1", "claude-code", released=False) == [("h1", UP, DOWN)]
    assert s.on_gone("h1", "claude-code", released=False) == []


def test_restart_between_heartbeats_is_recreated():
    s = HouseStates()
    s.on_set("h1", "claude-code", "7")
    assert s.on_set("h1", "claude-code", "9") == [("h1", UP, RECREATING), ("h1", RECREATING, UP)]


def test_reseed_reports_what_changed_while_disconnected():
    s = HouseStates()
    s.reseed({("h1", "claude-code"): "1", ("h2", "claude-code"): "1"})
    assert s.reseed({("h2", "claude-code"): "2"}) == [
        ("h1", UP, DOWN), ("h2", UP, RECREATING), ("h2", RECREATING, UP)]


def test_house_frames_move_tenant_routes_without_profile_write():
    tenant = SimpleNamespace(slug="b", house="h1", house_pinned=True, **AGENT)
    frames = frames_for([("h1", UP, DOWN)], lambda house: [tenant])
    assert frames[0] == {"_type": "house_state_changed", "house_id": "h1", "from": UP,
                         "to": DOWN, "tenants": ["b"], "hidden_tenants": 0}
    assert frames[1]["kind"] == "route" and frames[1]["reason"] == "house_down_fallback"
    assert frames[1]["to"] == {"runtime": "claude-code-bridge", "house": "h1",
                               "house_live": False, "pinned": True}
    back = frames_for([("h1", DOWN, UP)], lambda house: [tenant])[1]
    assert back["reason"] == "house_up" and back["to"]["runtime"] == "house"
    # recreating -> down is not a route change: both are "no live lease".
    assert len(frames_for([("h1", RECREATING, DOWN)], lambda house: [tenant])) == 1


def test_first_sighting_up_is_from_down_and_routes_tenants():
    tenant = SimpleNamespace(slug="b", house="h1", house_pinned=False, **AGENT)
    frames = frames_for([("h1", None, UP)], lambda house: [tenant])
    assert frames[0]["from"] == DOWN and frames[0]["tenants"] == ["b"]
    assert frames[1]["reason"] == "house_up" and frames[1]["to"]["runtime"] == "house"
    # First seen down: nothing changed from what a snapshot showed.
    assert frames_for([("h2", None, DOWN)], lambda house: [tenant]) == []


def test_hidden_tenants_are_counted_never_named():
    rows = [SimpleNamespace(slug=s, house="h1", house_pinned=False, **AGENT) for s in ("a", "secretbot")]
    frames = frames_for([("h1", UP, DOWN)], lambda house: rows, lambda slug: slug != "secretbot")
    assert frames[0]["tenants"] == ["a"] and frames[0]["hidden_tenants"] == 1
    assert [f.get("bot_id") for f in frames[1:]] == ["a"]
    assert "secretbot" not in str(frames)


def test_route_frames_carry_the_row_version():
    from datetime import datetime, timezone
    stamp = datetime(2026, 11, 6, tzinfo=timezone.utc)
    row = SimpleNamespace(slug="b", house="h1", house_pinned=False, updated_at=stamp, **AGENT)
    frame = frames_for([("h1", UP, DOWN)], lambda house: [row])[1]
    assert frame["version"] == ce.version_of(row) == int(stamp.timestamp() * 1_000_000)


def apply_frames(held: dict, frames: list[dict]) -> dict:
    """The documented client rule: assignment needs a greater version, route an equal one."""
    for frame in frames:
        version = held.get("version", 0)
        if frame["version"] > version or (frame["kind"] == ce.ROUTE and frame["version"] == version):
            held = frame
    return held


def test_stale_assignment_route_cannot_undo_a_newer_lease_route():
    """Assignment at V read h1 live, then paused; h1 expired and its route frame at V went first."""
    from datetime import datetime, timezone
    stamp = datetime(2026, 11, 6, tzinfo=timezone.utc)
    row = SimpleNamespace(slug="b", house="h1", house_pinned=False, updated_at=stamp, **AGENT)
    v = ce.version_of(row)
    snapshot = {"kind": ce.ASSIGNMENT, "version": v - 1, "to": {"runtime": "claude-code-bridge"}}
    route = frames_for([("h1", UP, DOWN)], lambda house: [row])[1]
    late = ce.location_frame("b", fields(), fields("h1"), "", {LEASE_H1: True}, v)
    held = apply_frames(snapshot, [route, late])
    assert held["to"]["runtime"] == "claude-code-bridge" and held["to"]["house_live"] is False
    # In order, the later lease news still applies at the same version.
    assert apply_frames(snapshot, [late, route]) == route
    back = frames_for([("h1", DOWN, UP)], lambda house: [row])[1]
    assert apply_frames(snapshot, [late, route, back])["to"]["runtime"] == "house"


def test_delivery_redacts_frames_of_bots_hidden_since_publication():
    shown = lambda slug: slug != "gone"  # noqa: E731
    frame = {"_type": "bot_location_changed", "bot_id": "gone", "seq": 3}
    assert ce.redact_for_delivery(frame, shown) is None
    assert ce.redact_for_delivery({**frame, "bot_id": "ok"}, shown)["bot_id"] == "ok"
    house = {"_type": "house_state_changed", "tenants": ["gone", "ok"], "hidden_tenants": 1}
    assert ce.redact_for_delivery(house, shown) == {
        "_type": "house_state_changed", "tenants": ["ok"], "hidden_tenants": 2}
    # Unreadable policy: fail closed.
    assert ce.redact_for_delivery({**frame, "bot_id": "ok"}, None) is None
    assert ce.redact_for_delivery(house, None)["tenants"] == []
    other = {"_type": "bot_profile_changed", "bot_id": "gone"}
    assert ce.redact_for_delivery(other, None) == other


# --- publication: visibility and committed versions ----------------------------

class _Redis:
    def __init__(self, live=()):
        self.live, self.added, self.n = set(live), [], 0

    def get(self, key):
        return "1" if key in self.live else None

    def incr(self, key):
        self.n += 1
        return self.n

    def xadd(self, stream, data, **_):
        import json
        self.added.append(json.loads(data["payload"]))

    def close(self):
        pass


def test_hidden_bots_are_not_published(monkeypatch):
    client = _Redis({LEASE_H1})
    monkeypatch.setattr(ce, "_redis", lambda config=None: client)
    monkeypatch.setattr(ce, "visibility", lambda config=None: lambda slug: slug != "secret")
    ce.emit_tenancy_changes([("secret", fields(), fields("h1"), "", 5),
                             ("open", fields(), fields("h1"), "", 6)])
    assert [(f["bot_id"], f["version"]) for f in client.added] == [("open", 6)]


def test_late_publication_is_older_by_version(monkeypatch):
    """Commit order h1 (v1) then h2 (v2), published h2 first: versions still order them."""
    client = _Redis({LEASE_H1, "agent:house:h2:lease:claude-code"})
    monkeypatch.setattr(ce, "_redis", lambda config=None: client)
    monkeypatch.setattr(ce, "visibility", lambda config=None: lambda slug: True)
    ce.emit_tenancy_changes([("b", fields("h1"), fields("h2"), "", 2)])
    ce.emit_tenancy_changes([("b", fields(), fields("h1"), "", 1)])
    held = apply_frames({}, client.added)
    assert held["to"]["house"] == "h2"
    assert [f["seq"] for f in client.added] == [1, 2]


# --- active location ------------------------------------------------------------

class _TurnRedis:
    def __init__(self, records):
        self.records = records

    async def get(self, key):
        return self.records.get(key)


def _bot(slug, house=None):
    return SimpleNamespace(slug=slug, house=house, house_pinned=False, **AGENT)


def test_active_uses_the_executing_request_record_only():
    import asyncio

    from llm_bawt.service.routes.campus import _active
    records = {
        # An older housed thread's records must not decide a new mansion turn.
        "agent:turn:sess:claude-code:b:nick": "agent:commands:house:h1",
        "agent:turn:req:old": "agent:commands:house:h1",
        "agent:turn:req:r2": "agent:commands:house:h2",
        "agent:turn:req:r3": "agent:commands",
    }
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 11, 6, tzinfo=timezone.utc)
    rows = [_bot("a", "h2"), _bot("b"), _bot("c", "h1"), _bot("d", "h1"), _bot("e"), _bot("f"),
            _bot("g")]
    for row in rows:
        row.updated_at = t0
    rows[-1].updated_at = t0 + timedelta(seconds=5)  # g released its house mid-turn
    turns = {"a": {"kind": "executing", "request_id": "r2"},
             "b": {"kind": "executing", "request_id": "new", "created_at": t0 + timedelta(1)},
             "g": {"kind": "executing", "request_id": "housed", "created_at": t0 + timedelta(seconds=1)},
             "c": {"kind": "executing", "request_id": "r3"},
             "d": {"kind": "executing", "request_id": "gone"},
             "e": {"kind": "idle"},
             "f": {"kind": "ambiguous"}}
    out = asyncio.run(_active(_TurnRedis(records), rows, turns))
    assert out["a"] == {"runtime": "house", "house": "h2"}
    assert out["b"] == "unknown"  # no record: the profile doesn't prove where it went
    assert out["g"] == "unknown"
    assert out["c"] == {"runtime": "claude-code-bridge"}  # recorded fallback
    assert out["d"] == "unknown"  # housed but no record: not trustworthy
    assert out["e"] is None and out["f"] == "unknown"


# --- watcher subscriptions -------------------------------------------------------

def test_every_subscription_ack_rereads_leases(monkeypatch):
    """redis-py resubscribes inside listen(); each ack must re-read, no timer."""
    import asyncio

    from llm_bawt import house_watcher as hw

    reads = iter([{("h1", "claude-code"): "1"}, {}])
    emitted: list = []

    class PubSub:
        async def psubscribe(self, *patterns):
            pass

        async def listen(self):
            yield {"type": "psubscribe"}
            yield {"type": "psubscribe"}  # automatic reconnect
            raise asyncio.CancelledError

        async def aclose(self):
            pass

    class Client:
        connection_pool = SimpleNamespace(connection_kwargs={})
        flag_reads = 0

        def pubsub(self):
            return PubSub()

        async def config_get(self, name):
            Client.flag_reads += 1
            return {name: "K$gx"}

        async def aclose(self):
            pass

    import redis.asyncio as aioredis
    monkeypatch.setattr(aioredis, "from_url", lambda *a, **k: Client())
    watcher = hw.HouseWatcher(SimpleNamespace(REDIS_URL="redis://x"))

    async def read(client):
        return next(reads)

    async def emit(transitions):
        emitted.append(transitions)

    monkeypatch.setattr(watcher, "_read_leases", read)
    monkeypatch.setattr(watcher, "_emit", emit)

    async def no_previews(client):
        return {}

    monkeypatch.setattr(hw, "read_previews", no_previews)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(watcher.run())
    assert emitted == [[("h1", UP, DOWN)]] and Client.flag_reads == 2
    assert watcher.states.state["h1"] == DOWN


def test_merge_flags_keeps_existing():
    assert set(merge_flags("")) == set("K$gx")
    assert set(merge_flags("Elg")) == set("Elg") | set("K$x")
    assert merge_flags("KA") == "AK"


# --- snapshot houses ---------------------------------------------------------

POOL = [{"id": "h1", "harness": "claude-code", "preview_host": "https://dev1.example.test"},
        {"id": "h2", "harness": "claude-code", "preview_host": "https://dev2.example.test"}]


def test_hidden_tenants_keep_occupancy_without_identity():
    rows = house_rows(POOL, {}, {}, {}, {"h2": 1})
    assert [(r["id"], r["tenants"], r["hidden_tenants"]) for r in rows] == [
        ("h1", [], 0), ("h2", [], 1)]


def test_configured_empty_and_down_houses_are_listed():
    rows = house_rows(POOL, {"h1": {"claude-code"}}, {}, {})
    assert [(r["id"], r["state"], r["configured"]) for r in rows] == [
        ("h1", UP, True), ("h2", DOWN, True)]
    assert rows[1]["preview_host"] == "https://dev2.example.test"


def test_watcher_distinguishes_recycle_from_death():
    rows = house_rows(POOL, {}, {"h1": RECREATING}, {})
    assert rows[0]["state"] == RECREATING and rows[1]["state"] == DOWN


def test_unreadable_redis_is_unknown_not_down():
    rows = house_rows(POOL, None, {}, {"h1": ["b"]})
    assert {r["state"] for r in rows} == {UNKNOWN}
    assert rows[0]["tenants"] == ["b"]


def test_unconfigured_observed_house_is_flagged():
    rows = house_rows(None, {"h9": {"codex"}}, {}, {"h3": ["b"]})
    assert [(r["id"], r["configured"], r["state"], r["harness"]) for r in rows] == [
        ("h3", False, DOWN, None), ("h9", False, UP, "codex")]


def test_pool_file_exposes_only_id_harness_and_origin(tmp_path, monkeypatch):
    import json

    from llm_bawt.house_pool_config import POOL_FILE_ENV, configured_pool

    monkeypatch.delenv(POOL_FILE_ENV, raising=False)
    assert configured_pool() is None
    empty = tmp_path / "empty.json"
    empty.write_text("")
    monkeypatch.setenv(POOL_FILE_ENV, str(empty))
    assert configured_pool() is None  # compose mounts /dev/null when unset
    pool = tmp_path / "houses.json"
    pool.write_text(json.dumps({
        "image": "img", "base_image": "base", "preview_network": "net",
        "service_token_volume": "vol", "frontend_env_file": "/secret/frontend.env",
        "houses": [{"id": "h1", "harness": "claude-code", "origin": "https://dev1.example.test",
                    "memory": "4g"}]}))
    monkeypatch.setenv(POOL_FILE_ENV, str(pool))
    assert configured_pool() == [{"id": "h1", "harness": "claude-code",
                                  "preview_host": "https://dev1.example.test"}]
    # A self-hosted tenant's pool file holds only houses (TASK-2129).
    pool.write_text(json.dumps({"houses": [
        {"id": "h1", "harness": "claude-code", "origin": "http://10.0.0.20:3101", "memory": "3g"}]}))
    assert configured_pool() == [{"id": "h1", "harness": "claude-code",
                                  "preview_host": "http://10.0.0.20:3101"}]
    pool.write_text("{not json")
    assert configured_pool() is None
